"""Dataset-level prediction heads for multi-source pre-training.

Unlike PM6's ``BatchedDecoders`` (one decoder per scalar task, batched for speed),
these heads produce many outputs per molecule (e.g. 1,328 PCBA assays) from a
single projection. PCBA uses a pure linear probe
(``num_layers=0``: one ``Linear(hidden_dim, n_assays)``) — mathematically
identical to ``n_assays`` independent ``Linear(hidden_dim, 1)`` heads stacked.
Each head is called sequentially in the forward pass; per-head losses are
aggregated via ``STCH`` alongside PM6 task losses.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class DatasetHead(nn.Module):
    """MLP/linear-probe head producing wide output from graph-level embeddings.

    Architecture:
        [Linear -> LayerNorm -> ReLU -> Dropout] * num_layers -> Linear

    With ``num_layers=0``, the head is a pure linear probe (single
    ``Linear(in_dim, output_dim)``). For PCBA this is the intended form —
    per-assay logit weights are the rows of the single weight matrix.

    Args:
        in_dim: Input (graph-level embedding) dimensionality.
        output_dim: Number of output logits (e.g. 1328 for PCBA).
        hidden_dim: Hidden layer width (unused when num_layers=0).
        num_layers: Number of hidden blocks before the output linear. 0 = linear probe.
        dropout: Dropout probability after ReLU in each hidden block.
    """

    def __init__(
        self,
        in_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(prev, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> [B, output_dim]"""
        return self.net(x)


def build_dataset_heads(
    task_dict: Dict[str, dict],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    dataset_task_names: tuple = ("pcba",),
    pcba_num_layers: Optional[int] = None,
) -> nn.ModuleDict:
    """Create a ``ModuleDict`` of ``DatasetHead`` instances for dataset-level tasks.

    Head ``output_dim`` is read from ``task_dict[name]['n_outputs']``.

    ``pcba_num_layers`` (default ``None`` = fall back to ``num_layers``) overrides
    the hidden-block depth for the ``pcba`` head only. Typical usage: pass
    ``pcba_num_layers=0`` to build PCBA as a pure linear probe while keeping PM6
    decoder depth at whatever ``num_layers`` was set to.

    PCBA supports two modes:
      * **aggregate** — ``task_dict`` contains a single ``"pcba"`` entry whose
        ``n_outputs`` is the full assay count. One head, one task.
      * **per-assay** — ``task_dict`` contains N ``"pcba/<assay>"`` entries, each
        with ``n_outputs=1`` and a ``pcba_assay_idx`` column index. They share a
        single underlying linear probe (one ``DatasetHead`` with ``n_outputs=N``);
        ``AbsWeighting.forward`` slices its output into per-task predictions.
    """
    heads: Dict[str, nn.Module] = {}

    # PCBA — handle both aggregate ("pcba") and per-assay ("pcba/<assay>") modes.
    pcba_info = task_dict.get("pcba")
    pcba_per_assay = [info for name, info in task_dict.items()
                      if info.get("pcba_assay_idx") is not None]
    if pcba_info is not None or pcba_per_assay:
        pcba_output_dim = (
            pcba_info["n_outputs"]
            if pcba_info is not None
            else len(pcba_per_assay)
        )
        heads["pcba"] = DatasetHead(
            in_dim=in_dim,
            output_dim=pcba_output_dim,
            hidden_dim=hidden_dim,
            num_layers=(
                pcba_num_layers if pcba_num_layers is not None else num_layers
            ),
            dropout=dropout,
        )

    # Respect ``dataset_task_names`` allow-list: if a name's been explicitly
    # disabled from the allow-list, drop the head we may have just built.
    heads = {k: v for k, v in heads.items() if k in dataset_task_names}
    return nn.ModuleDict(heads)
