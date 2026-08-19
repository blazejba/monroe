from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monroe.model.ckpt import load_ckpt, load_training_ckpt
from monroe.model.featurizer import build_single_graph
from monroe.model.grit import GritTransformer
from monroe.model.heads import DatasetHead, build_dataset_heads

__all__ = [
    "GritTransformer", "load_ckpt", "load_training_ckpt", "build_single_graph",
    "get_decoders", "copy_encoder_weights", "BatchedDecoders",
    "DatasetHead", "build_dataset_heads",
]


class _BatchedLinearBlock(nn.Module):
    """N independent MLPs executed as batched matmuls.

    Architecture per decoder: [Linear -> LayerNorm -> ReLU -> Dropout] x num_layers -> Linear.
    All N decoders run in a single batched operation.
    """

    def __init__(self, n_decoders, in_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.n_decoders = n_decoders
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.num_layers = num_layers

        sizes = [in_dim] + [hidden_dim] * num_layers

        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        self.ln_w = nn.ParameterList()
        self.ln_b = nn.ParameterList()

        for i in range(num_layers):
            # First layer uses F.linear (needs [out, in] layout per decoder);
            # subsequent layers use bmm — store as [N, in, out] to avoid transpose.
            if i == 0:
                w = torch.empty(n_decoders, sizes[i + 1], sizes[i])
            else:
                w = torch.empty(n_decoders, sizes[i], sizes[i + 1])
            fan_in, fan_out = sizes[i], sizes[i + 1]
            a = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(w, -a, a)
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.zeros(n_decoders, sizes[i + 1])))
            self.ln_w.append(nn.Parameter(torch.ones(n_decoders, sizes[i + 1])))
            self.ln_b.append(nn.Parameter(torch.zeros(n_decoders, sizes[i + 1])))

        # Store as [N, last_dim, output_dim] so bmm doesn't need transpose
        last_dim = hidden_dim if num_layers > 0 else in_dim
        out_w = torch.empty(n_decoders, last_dim, output_dim)
        fan_in, fan_out = last_dim, output_dim
        a = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(out_w, -a, a)
        self.out_W = nn.Parameter(out_w)
        self.out_b = nn.Parameter(torch.zeros(n_decoders, output_dim))

    def forward(self, x):
        """x: [B, in_dim] -> [n_decoders, B, output_dim]"""
        B = x.size(0)

        for i in range(self.num_layers):
            if i == 0:
                # First layer: shared input -> single large matmul
                W_flat = self.W[0].reshape(-1, x.size(-1))
                b_flat = self.b[0].reshape(-1)
                h = F.linear(x, W_flat, b_flat)  # [B, N*H]
                h = h.view(B, self.n_decoders, self.hidden_dim)
                h = h.transpose(0, 1).contiguous()  # [N, B, H]
            else:
                h = torch.bmm(h, self.W[i]) + self.b[i].unsqueeze(1)

            # LayerNorm per decoder (over hidden dim)
            # Compute in float32 (F.layer_norm is autocast-promoted to fp32;
            # manual impl must do this explicitly to avoid NaN in bf16 backward)
            inp_dtype = h.dtype
            h_f32 = h.float()
            mean = h_f32.mean(dim=-1, keepdim=True)
            var = h_f32.var(dim=-1, keepdim=True, unbiased=False)
            h = ((h_f32 - mean) * torch.rsqrt(var + 1e-5)).to(inp_dtype)
            h = h * self.ln_w[i].unsqueeze(1) + self.ln_b[i].unsqueeze(1)

            h = F.relu(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)

        # Output layer
        if self.num_layers == 0:
            # No hidden layers: x is [B, in_dim], use F.linear like first hidden layer
            W_flat = self.out_W.transpose(1, 2).reshape(-1, x.size(-1))  # [N*out, in]
            b_flat = self.out_b.reshape(-1)
            out = F.linear(x, W_flat, b_flat)  # [B, N*out]
            out = out.view(B, self.n_decoders, self.output_dim)
            out = out.transpose(0, 1).contiguous()  # [N, B, out]
        else:
            out = torch.bmm(h, self.out_W) + self.out_b.unsqueeze(1)
        return out  # [N, B, output_dim]


class BatchedDecoders(nn.Module):
    """All task decoders batched by (task_type, n_outputs) for efficient execution.

    Tasks are grouped by their type and output dimensionality. Within each group,
    all decoders run as a single batched matmul instead of N sequential calls.
    """

    def __init__(self, task_dict, in_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()

        group_map: Dict[tuple, list] = OrderedDict()
        for task, info in task_dict.items():
            if "n_outputs" not in info:
                continue
            key = (info.get("task_type", "graph_level"), info["n_outputs"])
            if key not in group_map:
                group_map[key] = []
            group_map[key].append(task)

        self._task_to_group: Dict[str, tuple] = {}
        self._group_tasks: Dict[str, list] = {}
        self._group_task_type: Dict[str, str] = {}

        self.groups = nn.ModuleDict()
        for (task_type, n_outputs), task_names in group_map.items():
            gkey = f"{task_type}_{n_outputs}"
            self._group_tasks[gkey] = task_names
            self._group_task_type[gkey] = task_type
            for i, task in enumerate(task_names):
                self._task_to_group[task] = (gkey, i)

            self.groups[gkey] = _BatchedLinearBlock(
                n_decoders=len(task_names),
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                output_dim=n_outputs,
                num_layers=num_layers,
                dropout=dropout,
            )

    @property
    def num_tasks(self):
        return len(self._task_to_group)

    def forward(self, graph_level, node_level):
        """Run all decoder groups, return per-task prediction dict."""
        if node_level is not None and node_level.dtype != graph_level.dtype:
            node_level = node_level.to(dtype=graph_level.dtype)

        out = {}
        for gkey, block in self.groups.items():
            task_type = self._group_task_type[gkey]
            if task_type in ("graph_level", "representation"):
                x = graph_level
            else:
                if node_level is None:
                    continue
                x = node_level

            batched_out = block(x)  # [N_group, B, output_dim]
            for i, task in enumerate(self._group_tasks[gkey]):
                out[task] = batched_out[i]

        return out


def get_decoders(
    task_dict: Dict[str, Any],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.1,
    normalization: Optional[str] = "layernorm",
) -> BatchedDecoders:
    return BatchedDecoders(
        task_dict=task_dict,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def copy_encoder_weights(
    src_model: nn.Module,
    tgt_model: nn.Module,
) -> nn.Module:
    # Exclude both PM6 decoders and new dataset-level heads; copy encoder only.
    excluded_prefixes = ("decoders.", "dataset_heads.")
    src_state = src_model.state_dict()
    filtered_state = {
        k: v for k, v in src_state.items() if not any(k.startswith(p) for p in excluded_prefixes)
    }
    tgt_model.load_state_dict(filtered_state, strict=False)
    return tgt_model
