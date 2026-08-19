import torch


class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task.

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """

    def __init__(self):
        self.record = []
        self.bs = []

    @staticmethod
    def _resolve_mask(gt, mask=None):
        """Return a boolean mask and its count, defaulting to non-NaN entries."""
        if mask is None:
            mask = ~torch.isnan(gt)
        return mask, mask.sum()

    def update_fun(self, pred, gt, **kwargs):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass

    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends)."""
        self.record = []
        self.bs = []


class PairwiseDistanceDummyMetric(AbsMetric):
    """Lightweight placeholder when loss already measures pairwise distances."""

    def update_fun(self, pred, gt, batch, mask=None, **kwargs):
        # intentionally no-op to avoid double computation with the loss
        return

    def score_fun(self):
        return [0.0]


class SparseL1(AbsMetric):
    """Calculate MAE for sparse targets."""

    def __init__(self):
        super().__init__()

    def update_fun(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if mask is None:
            mask = ~torch.isnan(gt)
        n_valid = mask.sum()
        abs_err = torch.where(mask, torch.abs(pred - gt), pred.new_tensor(0.0))
        self.record.append(abs_err.sum().detach())
        self.bs.append(n_valid.detach())

    def score_fun(self):
        if not self.record:
            return [0.0]
        total_err = torch.stack(self.record).sum().item()
        total_samples = torch.stack(self.bs).sum().item()
        return [total_err / total_samples if total_samples > 0 else 0.0]


class SparseLogMAE(SparseL1):
    """MAE computed after inverting the log transform.

    For log_huber loss: predictions are in log(1+|y|)*sign(y) space,
    we invert to original space before computing MAE.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def update_fun(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if mask is None:
            mask = ~torch.isnan(gt)
        n_valid = mask.sum()

        # Invert log transform: exp(|x|) - 1, restore sign
        pred_orig = (torch.exp(torch.abs(pred)) - 1) * torch.sign(pred + self.eps)
        abs_err = torch.where(mask, (pred_orig - gt).abs(), pred.new_tensor(0.0))
        self.record.append(abs_err.sum().detach())
        self.bs.append(n_valid.detach())


class SparseLogitMAE(SparseL1):
    """MAE for log_ratio_squared loss (values in [0,1]).

    Predictions are logits, apply sigmoid before computing MAE against [0,1] targets.
    """

    def update_fun(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if mask is None:
            mask = ~torch.isnan(gt)
        n_valid = mask.sum()
        abs_err = torch.where(mask, (torch.sigmoid(pred) - gt).abs(), pred.new_tensor(0.0))
        self.record.append(abs_err.sum().detach())
        self.bs.append(n_valid.detach())


class SparseBetaMAE(AbsMetric):
    """MAE for beta_nll loss using the predicted mean.

    Network outputs [mean_logit, log_precision], we use sigmoid(mean_logit)
    as the point prediction.
    """

    def __init__(self):
        super().__init__()

    def update_fun(self, pred, gt, mask=None, **kwargs):
        # pred is [N, 2]: (mean_logit, log_precision)
        if pred.dim() == 1:
            return  # Invalid shape
        if mask is None:
            mask = ~torch.isnan(gt)
        n_valid = mask.sum()
        pred_mean = torch.sigmoid(pred[:, 0])
        abs_err = torch.where(mask, (pred_mean - gt).abs(), pred_mean.new_tensor(0.0))
        self.record.append(abs_err.sum().detach())
        self.bs.append(n_valid.detach())

    def score_fun(self):
        if not self.record:
            return [0.0]
        total_err = torch.stack(self.record).sum().item()
        total_samples = torch.stack(self.bs).sum().item()
        return [total_err / total_samples if total_samples > 0 else 0.0]


class SparseOrdinalAcc(AbsMetric):
    """Accuracy for ordinal regression.

    Prediction is a latent score; P(Y > k) = sigmoid(pred - k).
    We predict the class by finding argmax of P(Y = k).

    Args:
        num_classes: Maximum number of ordinal classes (max count + 1).
            Derived from max(n_unique) across ordinal tasks in pm6_stats.json.
            Using a fixed K avoids per-batch .item() sync from gt.max().
    """

    def __init__(self, num_classes: int = None):
        super().__init__()
        self.num_classes = num_classes

    def update_fun(self, pred, gt, mask=None, **kwargs):
        if pred.dim() > 1:
            pred = pred.squeeze(1)
        if mask is None:
            mask = ~torch.isnan(gt)
        n_valid = mask.sum()

        pred_valid = pred[mask]
        gt_valid = gt[mask].long()

        K = self.num_classes
        if K is None:
            raise ValueError("num_classes must be set (derived from pm6_stats.json in build_task_dict)")

        # Thresholds 0, 1, ..., K-2
        thresholds = torch.arange(K - 1, device=pred.device, dtype=pred.dtype)

        # P(Y > k) = sigmoid(pred - k)  for k in [0, K-2]
        # Shape: [N, K-1]
        probs_gt = torch.sigmoid(pred_valid.unsqueeze(1) - thresholds.unsqueeze(0))

        # P(Y = k) for k in [0, K-1]
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k in [1, K-2]
        # P(Y = K-1) = P(Y > K-2)
        probs = torch.zeros(pred_valid.size(0), K, device=pred.device, dtype=pred.dtype)
        probs[:, 0] = 1 - probs_gt[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = probs_gt[:, k - 1] - probs_gt[:, k]
        probs[:, K - 1] = probs_gt[:, K - 2]

        pred_classes = probs.argmax(dim=1)
        correct = (pred_classes == gt_valid).sum()

        self.record.append(correct.detach())
        self.bs.append(n_valid.detach())

    def score_fun(self):
        if not self.record:
            return [0.0]
        total_correct = torch.stack(self.record).sum().item()
        total_samples = torch.stack(self.bs).sum().item()
        return [total_correct / total_samples if total_samples > 0 else 0.0]


class SparseFocalAUROC(AbsMetric):
    """Mean AUROC across assays for PCBA-style multi-task binary classification.

    AUROC cannot be accumulated incrementally (requires full ranked list), so this
    metric stores raw predictions and labels on CPU across batches and computes
    per-assay AUROC at ``score_fun()`` time. Assays with fewer than ``min_positives``
    positive samples are skipped (AUROC is ill-defined).

    Args:
        min_positives: Skip assays with fewer positives at score time. Default 10.
    """

    def __init__(self, min_positives: int = 10):
        super().__init__()
        self.min_positives = min_positives
        # Override: store raw [B, C] pred/gt chunks on CPU (float32, float32-with-NaN).
        self.pred_chunks = []
        self.gt_chunks = []

    def update_fun(self, pred, gt, mask=None, **kwargs):
        # pred, gt: [B, C] where C is the number of assays
        self.pred_chunks.append(pred.detach().to(dtype=torch.float32, device="cpu"))
        self.gt_chunks.append(gt.detach().to(dtype=torch.float32, device="cpu"))

    def score_fun(self):
        if not self.pred_chunks:
            return [0.0]
        preds = torch.cat(self.pred_chunks, dim=0).numpy()  # [N, C]
        gts = torch.cat(self.gt_chunks, dim=0).numpy()      # [N, C]

        import numpy as np
        from sklearn.metrics import roc_auc_score

        C = preds.shape[1]
        aurocs = []
        for c in range(C):
            col_gt = gts[:, c]
            col_pred = preds[:, c]
            valid = ~np.isnan(col_gt)
            if valid.sum() < 2:
                continue
            y_true = col_gt[valid]
            y_score = col_pred[valid]
            n_pos = int((y_true == 1).sum())
            n_neg = int((y_true == 0).sum())
            if n_pos < self.min_positives or n_neg < self.min_positives:
                continue
            try:
                aurocs.append(float(roc_auc_score(y_true, y_score)))
            except ValueError:
                continue
        return [float(np.mean(aurocs)) if aurocs else 0.0]

    def reinit(self):
        super().reinit()
        self.pred_chunks = []
        self.gt_chunks = []




