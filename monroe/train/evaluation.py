"""Evaluation utilities for training."""

import torch
from torch.amp import autocast

from monroe.train.dataset import TaskInfo, targets_to_dict, to_bfloat16_batch


def compute_kabsch_rmsd(pred, gt, batch):
    """Computes RMSD after Kabsch alignment.

    Args:
        pred: Predicted positions [Total_Nodes, 3].
        gt: Ground truth positions [Total_Nodes, 3].
        batch: Graph index for each node [Total_Nodes].

    Returns:
        Tuple of (rmsd_per_graph, graph_atom_counts).
    """
    from monroe.train.kabsch import kabsch_align

    pos_true = gt.float()
    graph_id = batch
    device = pos_true.device
    G = int(graph_id.max().item()) + 1

    try:
        pos_aligned, R, t = kabsch_align(pred, gt, graph_id)

        # Compute squared distances
        sq_diff = ((pos_aligned - pos_true) ** 2).sum(dim=-1)

        # Aggregate to per-graph RMSD
        graph_sq_diff_sum = torch.zeros((G,), device=device, dtype=torch.float32)
        graph_sq_diff_sum.index_add_(0, graph_id, sq_diff)

        graph_atom_counts = torch.zeros((G,), device=device, dtype=torch.float32)
        graph_atom_counts.index_add_(0, graph_id, torch.ones_like(sq_diff))

        rmsd_per_graph = torch.sqrt(graph_sq_diff_sum / graph_atom_counts.clamp_min(1.0))

        return rmsd_per_graph, graph_atom_counts
    except Exception:
        return torch.full((G,), float("nan"), device=device), torch.zeros(
            (G,), device=device
        )


def eval_model(model, val_loader, tracker, task_info: TaskInfo, structure_loss=False):
    """Run evaluation on validation set.

    Args:
        model: The model to evaluate (may be DDP-wrapped).
        val_loader: Validation DataLoader.
        tracker: Tracker instance for computing metrics.
        task_info: TaskInfo with task configuration.
        structure_loss: Whether to include structure prediction loss.

    Returns:
        Tuple of (mean_loss, mean_losses_dict, mean_metrics_dict).
    """
    model.eval()
    base_model = getattr(model, "module", model)

    with torch.no_grad():
        for batch, y_graph in val_loader:
            pos_gt = batch.pop("pos")
            node_targets = batch.pop("node_targets", None)

            batch = to_bfloat16_batch(batch, base_model.device, non_blocking=True)
            node_idxs = torch.arange(batch.num_nodes, device=base_model.device)
            y_graph = y_graph.to(base_model.device, non_blocking=True)
            pos_gt = pos_gt.to(base_model.device, non_blocking=True)

            targets = targets_to_dict(
                y_graph, task_info.graph_tasks, task_info.graph_target_indices
            )

            if task_info.node_tasks:
                if node_targets is None:
                    raise ValueError(
                        "Node targets missing from batch while node-level tasks are enabled."
                    )
                node_targets = node_targets.to(base_model.device, non_blocking=True)
                targets.update(
                    targets_to_dict(
                        node_targets,
                        task_info.node_tasks,
                        task_info.node_target_indices,
                    )
                )

            if structure_loss:
                targets["structure_pred"] = pos_gt

            # Multi-source tasks (pcba, and pcba/<assay> in per-assay mode)
            # live in task_dict but PM6 val shards don't carry their labels.
            # Inject all-NaN targets so the tracker sees a consistent key set;
            # losses/metrics mask everything to zero-valid and report 0.0.
            multisource_bases = ("pcba",)
            for name in multisource_bases:
                info = task_info.task_dict.get(name)
                if info is None or name in targets:
                    continue
                targets[name] = torch.full(
                    (batch.num_graphs, info["n_outputs"]),
                    float("nan"),
                    device=base_model.device,
                )
            # Per-assay PCBA mode: each assay is its own task entry.
            for task_name, info in task_info.task_dict.items():
                if info.get("pcba_assay_idx") is None or task_name in targets:
                    continue
                targets[task_name] = torch.full(
                    (batch.num_graphs, info["n_outputs"]),
                    float("nan"),
                    device=base_model.device,
                )

            with autocast(dtype=torch.bfloat16, device_type="cuda"):
                preds = model(batch, node_idxs)
                task_losses = tracker.update(preds, targets, batch.batch[node_idxs])
                nan_tasks = [
                    task
                    for task, loss in zip(task_info.task_dict.keys(), task_losses)
                    if torch.isnan(loss)
                ]
                if nan_tasks:
                    raise ValueError(
                        f"NaN occurred in tasks: {nan_tasks} during evaluation."
                    )

    return tracker.score()


def eval_pcba_val(model, pcba_val_loader, tracker, task_info: TaskInfo):
    """PCBA-only validation pass — real PCBA targets, NaN-injected everything else.

    Call after ``eval_model`` with a ``tracker.reset()`` in between, so the PCBA
    AUROC is computed purely over PCBA val molecules. All non-PCBA tasks see
    all-NaN targets and register 0 loss / 0 metric (ignored at log time).

    ``pcba_val_loader`` comes from ``build_pcba_val_dataloader``. Its ``y_graph``
    tensors are [B, n_pcba_assays] — the full Y_graph of the PCBA val shard,
    so they land verbatim on the ``pcba`` task (or get sliced per-assay when
    per-assay STCH mode is active).
    """
    model.eval()
    base_model = getattr(model, "module", model)
    pcba_slices = getattr(base_model, "pcba_assay_slices", {}) or {}

    with torch.no_grad():
        for batch, y_graph in pcba_val_loader:
            batch.pop("pos")
            batch.pop("node_targets", None)
            batch = to_bfloat16_batch(batch, base_model.device, non_blocking=True)
            node_idxs = torch.arange(batch.num_nodes, device=base_model.device)
            y_graph = y_graph.to(base_model.device, non_blocking=True)

            targets = {}
            # NaN-inject PM6 graph tasks (their n_outputs is always 1 in practice).
            for task_name in task_info.graph_tasks:
                n_out = task_info.task_dict[task_name]["n_outputs"]
                targets[task_name] = torch.full(
                    (batch.num_graphs, n_out),
                    float("nan"),
                    device=base_model.device,
                )
            # NaN-inject node tasks.
            for task_name in task_info.node_tasks:
                n_out = task_info.task_dict[task_name]["n_outputs"]
                targets[task_name] = torch.full(
                    (batch.num_nodes, n_out),
                    float("nan"),
                    device=base_model.device,
                )
            if "structure_pred" in task_info.task_dict:
                targets["structure_pred"] = torch.full(
                    (batch.num_nodes, 3),
                    float("nan"),
                    device=base_model.device,
                )
            # Real PCBA targets (aggregate mode). In per-assay mode this key
            # isn't in task_dict, so we split instead.
            if "pcba" in task_info.task_dict:
                targets["pcba"] = y_graph
            for task_name, idx in pcba_slices.items():
                targets[task_name] = y_graph[:, idx:idx + 1]

            with autocast(dtype=torch.bfloat16, device_type="cuda"):
                preds = model(batch, node_idxs)
                tracker.update(preds, targets, batch.batch[node_idxs])

    return tracker.score()
