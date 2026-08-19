import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsLoss(object):
    r"""An abstract class for loss functions."""

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt, **kwargs):
        r"""Calculate the loss.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass

    def _get_batch_size(self, pred, gt, **kwargs):
        """Return the effective batch size for loss averaging. Override in subclasses."""
        return pred.size()[0]

    def _update_loss(self, pred, gt, **kwargs):
        loss = self.compute_loss(pred, gt, **kwargs)
        # Record loss on GPU — non-finite entries are filtered at score time
        self.record.append(loss.detach())
        self.bs.append(self._get_batch_size(pred, gt, **kwargs))
        return loss

    def _average_loss(self):
        record = torch.stack(self.record)
        bs = torch.tensor(self.bs, device=record.device, dtype=record.dtype)
        finite = torch.isfinite(record)
        record = torch.where(finite, record, record.new_tensor(0.0))
        bs = torch.where(finite, bs, bs.new_tensor(0.0))
        total = bs.sum()
        if total == 0:
            return 0.0
        return float((record * bs).sum().div(total).item())

    def _reinit(self):
        self.record = []
        self.bs = []


class KabschAlignedCoordLoss(AbsLoss):
    r"""Kabsch-aligned coordinate loss averaged over graphs.

    Uses Kabsch SVD alignment to find optimal rotation and translation,
    then computes coordinate loss after alignment. This approach is more
    principled than pairwise distance matching and eliminates the need
    for pair sampling.

    Args:
        loss_type: Type of loss to compute ('huber', 'l2', 'l1').
        huber_delta: Delta parameter for Huber loss.
        eps: Small constant for numerical stability.
    """

    def __init__(self, loss_type: str = "huber", huber_delta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.eps = eps

    def _get_batch_size(self, pred, gt, batch, **kwargs):
        """Track number of graphs (not nodes) since loss is averaged per-graph."""
        return int(batch.max()) + 1

    def compute_loss(self, pred, gt, batch, mask=None, **kwargs):
        """
        pred:  [N_nodes, 3] predicted coords
        gt:    [N_nodes, 3] true coords
        batch: [N_nodes]    graph indices
        mask:  [N_nodes] or [N_nodes, 3] optional mask (0/1)
        """
        from monroe.train.kabsch import kabsch_align

        pos_pred = pred.float()
        pos_true = gt.float()
        graph_id = batch

        device, dtype = pos_pred.device, pos_pred.dtype
        G = int(graph_id.max().item()) + 1

        # Derive / reduce mask to 1D per-node. A node is valid iff all its
        # coordinates pass the mask (or, when mask is not supplied, iff all
        # coords are finite). Auto-deriving from NaN matches the Sparse*Loss
        # convention and lets PCBA molecules (with all-NaN POS rows in
        # a combined multi-source batch) flow through as zero-weight with no
        # extra wiring at the call site.
        if mask is None:
            mask_1d = ~torch.isnan(gt).any(dim=-1)
        elif mask.dim() > 1:
            mask_1d = mask.all(dim=-1)
        else:
            mask_1d = mask
        w = mask_1d.to(device=device, dtype=dtype)

        # Replace invalid ground-truth coords with zeros before Kabsch. In IEEE
        # 754 ``0.0 * NaN == NaN``, so the w=0 weights alone are not enough to
        # prevent NaN contamination of the SVD inputs.
        pos_true = torch.where(
            mask_1d.unsqueeze(-1), pos_true, torch.zeros_like(pos_true)
        )

        pos_aligned, R, t = kabsch_align(pos_pred, pos_true, graph_id, w=w, eps=self.eps)

        # Compute loss
        if self.loss_type == "huber":
            per_atom = F.smooth_l1_loss(
                pos_aligned, pos_true, beta=self.huber_delta, reduction="none"
            ).sum(dim=-1)
        elif self.loss_type == "l2":
            per_atom = ((pos_aligned - pos_true) ** 2).sum(dim=-1)
        elif self.loss_type == "l1":
            per_atom = (pos_aligned - pos_true).abs().sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Aggregate: use the reduced 1D mask weights
        if w.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # Weighted mean over atoms within each graph, then mean over graphs
        num = torch.zeros((G,), device=per_atom.device, dtype=per_atom.dtype).index_add_(
            0, graph_id, per_atom * w
        )
        den = torch.zeros((G,), device=per_atom.device, dtype=per_atom.dtype).index_add_(
            0, graph_id, w
        ).clamp_min(1.0)
        return (num / den).mean()



class SparseHuberLoss(AbsLoss):
    r"""Huber loss for sparse targets with NaN masking.

    Uses PyTorch's nn.HuberLoss with reduction='none', then applies
    mask for sparse labels.

    Args:
        delta: Threshold where loss transitions from quadratic to linear.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1:
            pred = pred.squeeze(1)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)
        # Replace NaN targets with 0 to avoid NaN in loss, then mask out
        gt_safe = torch.where(mask, gt, torch.zeros_like(gt))
        per_elem = self.loss_fn(pred, gt_safe)
        masked = torch.where(mask, per_elem, torch.zeros_like(per_elem))
        return masked.sum() / mask.sum().clamp(min=1)


class SparseLogHuberLoss(AbsLoss):
    r"""Log-transformed Huber loss for sparse positive targets.

    Applies log(1 + |x|) * sign(x) transform to both predictions and targets
    before computing Huber loss using PyTorch's nn.HuberLoss.
    Suitable for positive-valued targets where relative errors matter more
    than absolute errors.

    Args:
        delta: Threshold where loss transitions from quadratic to linear.
        eps: Small constant for numerical stability.
    """

    def __init__(self, delta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)
        self.eps = eps

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1:
            pred = pred.squeeze(1)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)
        gt_safe = torch.where(mask, gt, torch.zeros_like(gt))
        gt_log = torch.log1p(torch.abs(gt_safe) + self.eps) * torch.sign(gt_safe + self.eps)
        pred_log = torch.log1p(torch.abs(pred) + self.eps) * torch.sign(pred + self.eps)
        per_elem = self.loss_fn(pred_log, gt_log)
        masked = torch.where(mask, per_elem, torch.zeros_like(per_elem))
        return masked.sum() / mask.sum().clamp(min=1)


class SparseLogRatioSquaredLoss(AbsLoss):
    r"""Log-ratio squared loss for values in [0, 1].

    Computes squared error directly in logit space. Network outputs raw logits,
    ground truth is transformed to logit space using clamping for stability.

    This is numerically stable under bfloat16/autocast because we avoid
    applying log to sigmoid outputs (which can saturate to 0 or 1).

    Args:
        eps: Clamping margin to avoid log(0) for ground truth logit transform.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1:
            pred = pred.squeeze(1)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)
        # Replace NaN with 0.5 (maps to logit 0, giving safe squared error)
        gt_safe = torch.where(mask, gt, gt.new_tensor(0.5))
        gt_clamped = torch.clamp(gt_safe, self.eps, 1.0 - self.eps)
        gt_logit = torch.log(gt_clamped) - torch.log1p(-gt_clamped)
        diff = torch.clamp(pred - gt_logit, -20.0, 20.0)
        per_elem = diff ** 2
        masked = torch.where(mask, per_elem, torch.zeros_like(per_elem))
        return masked.sum() / mask.sum().clamp(min=1)


class SparseBetaNLLLoss(AbsLoss):
    r"""Beta distribution negative log-likelihood for [0, 1] targets.
    
    Models the target as coming from a Beta distribution. Network outputs
    alpha and beta concentration parameters (or mean and precision).
    Suitable for bounded continuous targets like QED, fsp3.
    
    The network should output 2 values per sample: [mean_logit, log_precision]
    - mean is obtained via sigmoid(mean_logit) 
    - precision (alpha + beta) is obtained via softplus(log_precision) + min_precision
    
    Numerically stable: alpha, beta are clamped to [min_alpha, max_alpha] to prevent
    lgamma from producing extreme values or NaN gradients.
    
    Args:
        eps: Small constant for clamping ground truth.
        min_precision: Minimum precision (alpha + beta) for the distribution.
        min_alpha: Minimum value for alpha and beta to prevent lgamma instability.
        max_alpha: Maximum value to prevent overflow in lgamma.
    """

    def __init__(self, eps: float = 1e-4, min_precision: float = 2.0,
                 min_alpha: float = 1.0, max_alpha: float = 100.0,
                 use_softplus: bool = True):
        super().__init__()
        self.eps = eps
        self.min_precision = min_precision
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.use_softplus = use_softplus

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        # pred should be [N, 2]: (mean_logit, log_precision)
        if pred.dim() == 1:
            raise ValueError("SparseBetaNLLLoss expects pred of shape [N, 2]")
        if pred.size(-1) != 2:
            raise ValueError(f"SparseBetaNLLLoss expects 2 outputs, got {pred.size(-1)}")
        
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[mask]  # [M, 2]
        gt_valid = gt[mask]      # [M]
        
        # Extract mean and precision from network output
        # Clamp mean away from 0 and 1 to ensure alpha, beta stay reasonable
        mean = torch.sigmoid(pred_valid[:, 0])
        mean = torch.clamp(mean, self.min_alpha / self.max_alpha, 1.0 - self.min_alpha / self.max_alpha)
        
        precision = F.softplus(pred_valid[:, 1]) + self.min_precision
        precision = torch.clamp(precision, self.min_precision, self.max_alpha * 2)
        
        # Convert to alpha, beta with clamping
        alpha = torch.clamp(mean * precision, self.min_alpha, self.max_alpha)
        beta = torch.clamp((1.0 - mean) * precision, self.min_alpha, self.max_alpha)
        
        # Clamp target to avoid log(0)
        gt_clamped = torch.clamp(gt_valid, self.eps, 1.0 - self.eps)
        
        # Beta NLL: -log Beta(y; alpha, beta)
        # = -(alpha-1)*log(y) - (beta-1)*log(1-y) + lgamma(alpha) + lgamma(beta) - lgamma(alpha+beta)
        log_y = torch.log(gt_clamped)
        log_1my = torch.log(1.0 - gt_clamped)
        
        nll = (
            -(alpha - 1.0) * log_y 
            - (beta - 1.0) * log_1my 
            + torch.lgamma(alpha) 
            + torch.lgamma(beta) 
            - torch.lgamma(alpha + beta)
        )
        
        # STCH scalarization requires positive losses (it normalizes by a nadir
        # vector and optionally takes log).  softplus preserves the full dynamic
        # range for positive NLL while smoothly mapping negative NLL to small
        # positive values.  The legacy +3.0 constant offset works but compresses
        # the signal that STCH sees, reducing optimization pressure on this task.
        if self.use_softplus:
            return F.softplus(nll.mean())
        return nll.mean() + 3.0



class SparseOrdinalLoss(AbsLoss):
    r"""Ordinal regression loss for count/discrete ordered targets.

    Uses cumulative link model: P(Y <= k) = sigmoid(theta_k - f(x))
    where theta_k are learned thresholds and f(x) is network output.

    For simplicity, we use a single output and compute binary cross-entropy
    for each threshold crossing, treating it as an ordered regression problem.

    Args:
        num_classes: Maximum number of ordinal classes (max count + 1).
            Derived from max(n_unique) across ordinal tasks in pm6_stats.json.
            Using a fixed K avoids per-batch .item() sync from gt.max().
    """

    def __init__(self, num_classes: int = None):
        super().__init__()
        self.num_classes = num_classes

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1:
            pred = pred.squeeze(1)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[mask]
        gt_valid = gt[mask].float()

        K = self.num_classes
        if K is None:
            raise ValueError("num_classes must be set (derived from pm6_stats.json in build_task_dict)")

        # Closed-form cumulative BCE using softplus(-x) - softplus(x) = -x:
        #   BCE_total(p, g) = sum_k softplus(p - k) - g*p + g*(g-1)/2
        # Accumulate softplus sum in O(N) memory instead of O(N*K).
        sp_sum = pred_valid.new_zeros(pred_valid.shape)
        for k in range(K - 1):
            sp_sum.add_(F.softplus(pred_valid - k))

        per_sample = sp_sum - gt_valid * pred_valid + gt_valid * (gt_valid - 1) * 0.5
        return per_sample.mean() / (K - 1)


def off_diag_cov_loss(emb: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty (VICReg-style decorrelation).

    Penalizes correlation between embedding dimensions to prevent dimensional
    collapse. Computed in fp32 for numerical stability.

    Args:
        emb: [B, D] graph-level embeddings.

    Returns:
        Scalar loss: mean squared off-diagonal covariance.
    """
    B, D = emb.shape
    if B < 2:
        return emb.new_tensor(0.0)
    emb = emb.float()
    emb = emb - emb.mean(dim=0)
    cov = (emb.T @ emb) / (B - 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / (D * (D - 1))


class SparseFocalLoss(AbsLoss):
    r"""Focal loss for binary classification with NaN masking.

    Focal loss (Lin et al., 2017) downweights well-classified examples and focuses
    gradient on hard ones. Designed for extreme class imbalance — PCBA has 99:1
    negative:positive ratio.

        L = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p, and alpha_t balances positive/negative contribution.

    Args:
        gamma: Focusing parameter. gamma=0 recovers BCE. Default 2.0 (RetinaNet recipe).
        alpha: Positive-class weight in [0, 1]. None disables class balancing.
            Default 0.25 (RetinaNet recipe: 0.25 for positives, 0.75 for negatives).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, pred, gt, mask=None, **kwargs):
        # pred: [B, C] logits; gt: [B, C] binary targets (0/1/NaN for missing)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if mask is None:
            mask = ~torch.isnan(gt)
        elif mask.device != pred.device:
            mask = mask.to(pred.device)

        # Replace NaN targets with 0 to keep arithmetic finite; mask zeros them out.
        gt_safe = torch.where(mask, gt, torch.zeros_like(gt))

        # Compute BCE per-element (numerically stable via logits)
        bce = F.binary_cross_entropy_with_logits(pred, gt_safe, reduction='none')

        # p_t = p if y=1 else 1-p; equivalently, p_t = exp(-bce)
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = gt_safe * self.alpha + (1.0 - gt_safe) * (1.0 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce
        masked = torch.where(mask, loss, torch.zeros_like(loss))
        return masked.sum() / mask.sum().clamp(min=1)
