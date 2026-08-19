"""Batched Kabsch alignment for variable-size graphs stored in flat tensors."""

import torch


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def kabsch_align(
    pos_pred: torch.Tensor,
    pos_true: torch.Tensor,
    graph_id: torch.Tensor,
    w: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kabsch SVD alignment of predicted coordinates onto true coordinates.

    Finds the optimal rotation R and translation t (per graph) that minimize
    the weighted squared distance between R @ pos_pred + t and pos_true.

    Args:
        pos_pred: Predicted positions [M, 3].
        pos_true: Ground-truth positions [M, 3].
        graph_id: Graph index for each node [M] (long tensor).
        w: Per-node weights [M]. Defaults to uniform weights.
        eps: Numerical stability constant.

    Returns:
        pos_aligned: Aligned predicted positions [M, 3].
        R: Rotation matrices [G, 3, 3].
        t: Translation vectors [G, 3].
    """
    pos_pred = pos_pred.float()
    pos_true = pos_true.float()
    device = pos_pred.device
    M = pos_pred.shape[0]
    G = int(graph_id.max().item()) + 1

    if w is None:
        w = torch.ones(M, device=device, dtype=torch.float32)
    else:
        w = w.float()

    # Per-graph weight sums and normalized weights
    w_sum = torch.zeros(G, device=device, dtype=torch.float32)
    w_sum.index_add_(0, graph_id, w)
    w_sum = w_sum.clamp_min(eps)
    w_norm = w / w_sum[graph_id]

    # Per-graph centroids
    c_pred = torch.zeros(G, 3, device=device, dtype=torch.float32)
    c_true = torch.zeros(G, 3, device=device, dtype=torch.float32)
    c_pred.index_add_(0, graph_id, w_norm[:, None] * pos_pred)
    c_true.index_add_(0, graph_id, w_norm[:, None] * pos_true)

    X = pos_pred - c_pred[graph_id]
    Y = pos_true - c_true[graph_id]

    # Per-graph covariance H = X^T W Y, shape [G, 3, 3]
    H = torch.zeros(G, 3, 3, device=device, dtype=torch.float32)
    outer = (w_norm[:, None] * X).unsqueeze(2) * Y.unsqueeze(1)
    H.index_add_(0, graph_id, outer)

    # Regularize H to prevent rank-deficient matrices (near-linear/planar molecules)
    # which cause NaN in SVD backward due to 1/(sigma_i^2 - sigma_j^2) terms blowing up.
    # Use non-equal diagonal so singular values are always distinct (equal diagonal
    # produces degenerate s_i == s_j for single-atom or coincident-atom graphs).
    _reg = torch.tensor([3e-6, 2e-6, 1e-6], device=device, dtype=torch.float32)
    H = H + torch.diag(_reg).unsqueeze(0)

    # Batched SVD
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    # Reflection correction: flip last singular vector when det(V @ U^T) < 0
    R = V @ U.transpose(-2, -1)
    detR = torch.linalg.det(R)
    flip = (detR < 0).float()
    D = torch.ones(G, 3, device=device, dtype=torch.float32)
    D[:, -1] = 1.0 - 2.0 * flip
    R = (V * D.unsqueeze(1)) @ U.transpose(-2, -1)

    # Translation: t = c_true - R @ c_pred
    t = c_true - torch.einsum("gij,gj->gi", R, c_pred)

    # Apply per-atom: aligned = R_g @ pos_pred + t_g
    pos_aligned = torch.einsum("mij,mj->mi", R[graph_id], pos_pred) + t[graph_id]

    return pos_aligned, R, t
