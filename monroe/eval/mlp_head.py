"""Per-task MLP head evaluation on frozen encoder embeddings.

MiniMol-style approach: extract embeddings once, train a lightweight MLP per task
with skip connection, batch norm, and early stopping.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from monroe.utils import printf


class MLPTaskHead(nn.Module):
    """MLP head with skip connection (concat input + MLP output)."""

    def __init__(self, in_dim: int, is_classification: bool, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        readout = [nn.Linear(in_dim * 2, 1)]
        if is_classification:
            readout.append(nn.Sigmoid())
        self.readout = nn.Sequential(*readout)
        self.is_classification = is_classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.mlp(x)], dim=-1)
        return self.readout(h).squeeze(-1)


def train_mlp_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    is_classification: bool,
    seed: int = 42,
    lr: float = 3e-4,
    epochs: int = 50,
    patience: int = 5,
    batch_size: int = 64,
    dropout: float = 0.1,
    val_fraction: float = 0.2,
    device: torch.device | None = None,
) -> tuple[MLPTaskHead, float, float]:
    """Train an MLP head on precomputed embeddings.

    Args:
        X_train: Embeddings array (N, D)
        y_train: Labels array (N,)
        is_classification: Whether this is a classification task
        seed: Random seed for train/val split
        lr: Learning rate
        epochs: Max training epochs
        patience: Early stopping patience
        batch_size: Training batch size
        dropout: Dropout rate in MLP
        val_fraction: Fraction of training data for validation
        device: Torch device

    Returns:
        (trained_model, target_mean, target_std) — mean/std are 0/1 for classification
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(seed)
    n = len(X_train)
    n_val = max(1, int(n * val_fraction))
    indices = rng.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx].copy()
    X_val = X_train[val_idx]
    y_val = y_train[val_idx].copy()

    # Standardize regression targets
    y_mean, y_std = 0.0, 1.0
    if not is_classification:
        y_mean = float(np.mean(y_tr))
        y_std = float(np.std(y_tr))
        if y_std < 1e-8:
            y_std = 1.0
        y_tr = (y_tr - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X_tr) > batch_size,
    )

    in_dim = X_train.shape[1]
    model = MLPTaskHead(in_dim, is_classification, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    loss_fn = F.binary_cross_entropy if is_classification else F.mse_loss

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, y_mean, y_std


def predict_mlp(
    model: MLPTaskHead,
    X: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run MLP prediction, inverse-scaling regression outputs.

    Returns (predictions, probabilities_or_None).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    all_preds = []
    for start in range(0, len(X_t), batch_size):
        xb = X_t[start:start + batch_size].to(device)
        with torch.no_grad():
            pred = model(xb).cpu().numpy()
        all_preds.append(pred)
    preds = np.concatenate(all_preds)

    if model.is_classification:
        probs = preds.copy()
        binary = (preds > 0.5).astype(float)
        return binary, probs
    else:
        preds = preds * y_std + y_mean
        return preds, None
