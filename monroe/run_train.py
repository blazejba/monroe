"""Training script for Monroe molecular foundation model."""

import gc
import json
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast

from monroe.config import dict_to_namespace, parse_args, to_dict
from monroe.model import build_dataset_heads, get_decoders
from monroe.model.ckpt import _build_encoder, load_training_ckpt, save_ckpt
from monroe.mtl import build_mtl_model_class
from monroe.train.dataset import (
    ShardLoader,
    TaskInfo,
    add_multisource_tasks,
    build_pcba_val_dataloader,
    build_resident_dataloaders,
    build_target_indices,
    build_task_dict,
    concat_multi_source,
    dataloader_factory,
    load_resident_datasets,
    targets_to_dict,
    to_bfloat16_batch,
)
from monroe.train.distributed import (
    DistributedConfig,
    setup_distributed,
    wrap_model_distributed,
)
from monroe.train.ema import ModelEMA
from monroe.train.evaluation import eval_model, eval_pcba_val
from monroe.train.loss import off_diag_cov_loss
from monroe.train.optimizer import optimizer_factory
from monroe.train.tracker import Tracker, init_wandb
from monroe.utils import count_parameters, printf

# -----------------------------------------------------------------------------
# Multi-source helpers
# -----------------------------------------------------------------------------
#
# Multi-source (PM6 + PCBA) enters the training loop at three points:
#
#   1. Task registration: _maybe_add_multisource (below) appends pcba entries
#      to task_dict when --pcba-dir is set. These tasks route through
#      dataset_heads in build_model (not BatchedDecoders).
#
#   2. Resident data loading: load_resident_datasets (monroe/train/dataset.py)
#      loads each configured dataset once into shared memory on local rank 0,
#      broadcasts the SHM metadata to the other ranks, and hands back a
#      ResidentDataset per dataset. Called once in main() before the PM6
#      shard loop.
#
#   3. Per-batch merging: concat_multi_source (monroe/train/dataset.py) fuses
#      the PM6 PyG Batch with each resident mini-batch and produces a combined
#      Batch plus a y_dict keyed by dataset name, each NaN-padded to the
#      combined batch size. Sparse losses mask the NaN rows automatically.


def _peek_resident_y_cols(shard_dir: str | None) -> list | None:
    """Read the assay/gene column names of a resident dataset's Y_graph."""
    if not shard_dir:
        return None
    import glob as _glob
    cols_files = sorted(_glob.glob(f"{shard_dir}/*.Y_graph_cols.json"))
    if not cols_files:
        raise FileNotFoundError(f"No Y_graph_cols.json found under {shard_dir}")
    with open(cols_files[0]) as f:
        return json.load(f)


def _peek_resident_n_outputs(shard_dir: str | None) -> int | None:
    """Read the column count of a resident dataset's Y_graph without loading arrays."""
    cols = _peek_resident_y_cols(shard_dir)
    return None if cols is None else len(cols)


# Dataset-level tasks served by dataset_heads rather than BatchedDecoders.
MULTISOURCE_TASK_NAMES = ("pcba",)


def _maybe_add_multisource(task_dict: dict, hp) -> dict:
    """Add PCBA entries to task_dict if its data dir is configured."""
    pcba_dir = getattr(hp.pretrain, "pcba_dir", None)
    if not pcba_dir:
        return task_dict

    pcba_per_assay = bool(getattr(hp.pretrain, "pcba_per_assay_stch", False))
    pcba_assay_names = _peek_resident_y_cols(pcba_dir) if pcba_per_assay else None
    pcba_n = _peek_resident_n_outputs(pcba_dir)
    return add_multisource_tasks(
        task_dict,
        pcba_n_assays=pcba_n,
        pcba_per_assay=pcba_per_assay,
        pcba_assay_names=pcba_assay_names,
    )


def _collect_pcba_assay_slices(task_dict: dict) -> dict:
    """Return ``{task_name: pcba_assay_idx}`` for per-assay PCBA mode; empty otherwise."""
    return {
        name: info["pcba_assay_idx"]
        for name, info in task_dict.items()
        if info.get("pcba_assay_idx") is not None
    }


def _load_with_logging(model, state_dict, is_main: bool) -> None:
    """Load state dict with ``strict=False``, logging missing/unexpected keys."""
    result = model.load_state_dict(state_dict, strict=False)
    if not is_main:
        return
    if result.missing_keys:
        printf(f"Load checkpoint: {len(result.missing_keys)} missing keys "
               f"(e.g. {result.missing_keys[:3]}). These will remain at init.")
    if result.unexpected_keys:
        printf(f"Load checkpoint: {len(result.unexpected_keys)} unexpected keys "
               f"(e.g. {result.unexpected_keys[:3]}). Ignored.")


# -----------------------------------------------------------------------------
# Model Building
# -----------------------------------------------------------------------------


def build_model(hp, task_dict, device):
    """Build the MTL model with encoder, PM6 decoders, and optional dataset heads."""

    def encoder_factory():
        return _build_encoder(to_dict(hp)).to(device)

    # PM6 per-task decoders go through BatchedDecoders; PCBA tasks are filtered
    # out of task_dict here because they use standalone DatasetHead modules.
    # In per-assay PCBA mode the filter also excludes every ``pcba/<assay>`` task,
    # since those share the single ``pcba`` head (sliced per-task in forward).
    pm6_task_dict = {
        name: info for name, info in task_dict.items()
        if name not in MULTISOURCE_TASK_NAMES
        and info.get("pcba_assay_idx") is None
    }
    decoders = get_decoders(
        task_dict=pm6_task_dict,
        in_dim=hp.encoder.hidden_dim,
        **to_dict(hp.pretrain.decoder),
    ).to(device)

    # Dataset-level heads (currently PCBA only). ModuleDict() if no
    # multi-source tasks in task_dict. PCBA uses pcba_head_layers (default 0 =
    # linear probe).
    dataset_heads = build_dataset_heads(
        task_dict=task_dict,
        in_dim=hp.encoder.hidden_dim,
        hidden_dim=hp.pretrain.decoder.hidden_dim,
        num_layers=hp.pretrain.decoder.num_layers,
        dropout=hp.pretrain.decoder.dropout,
        pcba_num_layers=getattr(hp.pretrain, "pcba_head_layers", None),
    ).to(device)

    # Runtime sanity: if PCBA is present and the operator asked for a linear
    # probe (pcba_head_layers == 0), confirm the head really is a single Linear.
    if "pcba" in dataset_heads and getattr(hp.pretrain, "pcba_head_layers", 1) == 0:
        children = list(dataset_heads["pcba"].net.children())
        assert len(children) == 1 and isinstance(children[0], torch.nn.Linear), (
            f"PCBA linear-probe head expected a single Linear child, got "
            f"{[type(c).__name__ for c in children]}"
        )

    MTLModel = build_mtl_model_class(hp.pretrain.mtl.weighting, hp.pretrain.n_shards)

    model = MTLModel(
        task_types={name: info["task_type"] for name, info in task_dict.items()},
        encoder_cls=encoder_factory,
        decoders=decoders,
        device=device,
        dataset_heads=dataset_heads,
        pcba_assay_slices=_collect_pcba_assay_slices(task_dict),
    )

    optimizer, scheduler = optimizer_factory(
        model=model,
        optim_params=to_dict(hp.pretrain.optim_param),
        scheduler_params=to_dict(hp.pretrain.scheduler_param),
    )

    return model, optimizer, scheduler


def setup_validation(hp, task_dict, dist_cfg: DistributedConfig):
    """Set up validation dataloader and task info."""
    val_loader = dataloader_factory(
        task_dict=task_dict,
        dataset_dir=hp.pretrain.data_dir,
        split="val",
        batch_node_budget=hp.pretrain.train_node_budget,
        n_workers=hp.pretrain.n_workers,
        seed=hp.pretrain.seed,
        subset_ratio=hp.pretrain.subset_ratio,
        walk_len=hp.encoder.walk_len,
        q=0.0,  # only use bad conformers for validation
        use_node_labels=hp.pretrain.use_node_labels,
        load_to_memory=hp.pretrain.load_to_memory,
        world_size=dist_cfg.world_size,
        rank=dist_cfg.rank,
    )

    # PM6 graph tasks: column-lookups into Y_graph. Multi-source tasks (pcba and
    # per-assay pcba/<assay> entries) get their predictions from dataset_heads
    # and their targets from extra_targets; they must NOT appear in
    # graph_tasks/graph_target_indices.
    graph_tasks = [
        name for name, info in task_dict.items()
        if info.get("task_type") == "graph_level"
        and name not in MULTISOURCE_TASK_NAMES
        and info.get("pcba_assay_idx") is None
    ]
    node_tasks = [
        name for name, info in task_dict.items() if info.get("task_type") == "node_level" and name != "structure_pred"
    ]

    task_info = TaskInfo(
        task_dict=task_dict,
        graph_tasks=graph_tasks,
        node_tasks=node_tasks,
        graph_target_indices=build_target_indices(graph_tasks, val_loader.dataset.Y_graph_cols),
        node_target_indices=(build_target_indices(node_tasks, val_loader.dataset.Y_node_cols) if node_tasks else []),
    )

    return val_loader, task_info


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_batch(
    model,
    base_model,
    batch,
    y_graph,
    optimizer,
    scheduler,
    tracker,
    task_info: TaskInfo,
    hp,
    dist_cfg: DistributedConfig,
    shard_idx: int,
    batch_idx: int,
    ema: ModelEMA | None = None,
    extra_targets: dict | None = None,
):
    """Process a single training batch.

    Args:
        y_graph: PM6 graph-level target tensor. When ``extra_targets`` is provided,
            this tensor has been NaN-padded to cover the combined batch size.
        extra_targets: Optional dict mapping dataset name (``pcba``) to a
            ``[B_total, n_outputs]`` target tensor, NaN-padded outside that
            dataset's row range.
    """
    pos_gt = batch.pop("pos")
    node_targets = batch.pop("node_targets", None)

    batch = to_bfloat16_batch(batch, dist_cfg.device, non_blocking=True)
    node_idxs = torch.arange(batch.num_nodes, device=base_model.device)
    y_graph = y_graph.to(base_model.device, non_blocking=True)

    targets = targets_to_dict(y_graph, task_info.graph_tasks, task_info.graph_target_indices)

    if task_info.node_tasks:
        if node_targets is None:
            raise ValueError("Node targets missing from batch while node-level tasks are enabled.")
        node_targets = node_targets.to(base_model.device, non_blocking=True)
        targets.update(targets_to_dict(node_targets, task_info.node_tasks, task_info.node_target_indices))

    if hp.pretrain.structure_loss:
        # pos_gt is NaN for non-PM6 molecules; KabschAlignedCoordLoss masks them.
        targets["structure_pred"] = pos_gt.to(base_model.device, non_blocking=True)

    if extra_targets:
        pcba_slices = getattr(base_model, "pcba_assay_slices", {}) or {}
        for name, y in extra_targets.items():
            if name == "pcba" and pcba_slices:
                # Per-assay PCBA STCH mode: split the wide [B_total, n_assays]
                # target tensor into per-task [B_total, 1] views. NaN padding
                # in non-PCBA rows survives the slice (NaN stays NaN).
                y_gpu = y.to(base_model.device, non_blocking=True)
                for task_name, idx in pcba_slices.items():
                    targets[task_name] = y_gpu[:, idx:idx + 1]
            else:
                targets[name] = y.to(base_model.device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        preds = model(batch, node_idxs)
        train_losses = tracker.update(preds, targets, batch.batch[node_idxs])

    weighting_kwargs = to_dict(hp.pretrain.mtl.weighting_args)
    # Inject precomputed pref vector (stored outside hp to avoid JSON serialization issues)
    if hasattr(train_one_batch, "_stch_pref_tensor") and train_one_batch._stch_pref_tensor is not None:
        weighting_kwargs["STCH_pref_vector"] = train_one_batch._stch_pref_tensor

    # Build extra_losses (added directly, bypassing MTL scalarization)
    extras: list[torch.Tensor] = []

    # Embedding decorrelation loss
    decorr_weight = getattr(hp.pretrain, "emb_decorr_weight", 0.0)
    if decorr_weight > 0:
        decorr = off_diag_cov_loss(base_model._last_graph_emb) * decorr_weight
        extras.append(decorr.reshape(1))

    task_names = list(task_info.task_dict.keys())

    if extras:
        weighting_kwargs["extra_losses"] = torch.cat(extras)

    if dist_cfg.is_distributed:
        with torch.no_grad():
            # Compute valid counts on GPU without per-task .item() syncs
            counts = []
            for name in task_names:
                t = targets[name]
                if torch.is_floating_point(t):
                    counts.append((~torch.isnan(t)).sum().to(dtype=train_losses.dtype))
                else:
                    counts.append(train_losses.new_tensor(t.shape[0]))
            task_counts = torch.stack(counts)
            loss_sums = train_losses.detach() * task_counts
            # Fuse two all_reduce calls into one
            payload = torch.stack([loss_sums, task_counts])
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            global_losses = payload[0] / payload[1].clamp_min(1)
            weighting_kwargs["global_losses"] = global_losses

    loss = base_model.backward(train_losses, **weighting_kwargs)

    # NaN-safe training: skip optimizer step if loss or gradients are corrupted
    # Synchronize across DDP ranks so all ranks agree on skip/step
    has_nan = _has_nan(loss, base_model, shard_idx, batch_idx)
    if dist_cfg.is_distributed:
        nan_flag = torch.tensor(1.0 if has_nan else 0.0, device=dist_cfg.device)
        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
        has_nan = nan_flag.item() > 0

    if has_nan:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(base_model)


def _has_nan(loss, model, shard_idx, batch_idx) -> bool:
    """Check for NaN in loss/gradients, warn if found.

    Uses at most 2 GPU→CPU syncs (one for loss, one for all gradients combined).
    """
    if not torch.isfinite(loss).item():
        warnings.warn(
            f"NaN/Inf loss at shard={shard_idx}, batch={batch_idx}. "
            f"Skipping optimizer step to prevent weight corruption."
        )
        return True

    # Single-sync gradient check: accumulate finiteness across all params on GPU
    all_finite = loss.new_tensor(True, dtype=torch.bool)
    for param in model.parameters():
        if param.grad is not None:
            all_finite = all_finite & torch.isfinite(param.grad).all()

    if not all_finite.item():
        warnings.warn(
            f"NaN/Inf gradients at shard={shard_idx}, batch={batch_idx}. "
            f"Skipping optimizer step to prevent weight corruption."
        )
        return True

    return False


def train_one_shard(
    model,
    base_model,
    shard_idx: int,
    hp,
    task_info: TaskInfo,
    dist_cfg: DistributedConfig,
    tracker,
    optimizer,
    scheduler,
    shard_loader: ShardLoader,
    data_shard_id: int | None = None,
    next_shard_id: int | None = None,
    ema: ModelEMA | None = None,
    residents: dict | None = None,
):
    """Train on a single data shard.

    Args:
        shard_idx: Position counter (0, 1, 2, ...) used for model.shard, loss buffer, logging.
        data_shard_id: Actual shard index to load from disk (may differ from shard_idx when
            shards are shuffled). Defaults to shard_idx for backwards compatibility.
        next_shard_id: Data shard ID to prefetch next, or None if last shard.
        residents: Optional dict of ``ResidentDataset`` (PCBA). When present, each
            PM6 batch is paired with fixed-count mini-batches from each resident dataset.
    """
    if data_shard_id is None:
        data_shard_id = shard_idx

    if dist_cfg.is_distributed:
        dist.barrier()

    shm_metadata = shard_loader.get_shard_metadata(data_shard_id, next_shard_id=next_shard_id)

    model.train()
    base_model.shard = shard_idx

    train_loader = dataloader_factory(
        task_dict=task_info.task_dict,
        dataset_dir=hp.pretrain.data_dir,
        split="train",
        batch_node_budget=hp.pretrain.train_node_budget,
        shard_id=data_shard_id,
        n_workers=hp.pretrain.n_workers,
        seed=hp.pretrain.seed,
        subset_ratio=hp.pretrain.subset_ratio,
        walk_len=hp.encoder.walk_len,
        use_node_labels=hp.pretrain.use_node_labels,
        load_to_memory=hp.pretrain.load_to_memory,
        world_size=dist_cfg.world_size,
        rank=dist_cfg.rank,
        shm_metadata=shm_metadata,
    )

    # Build one-shot resident dataloaders sized to match PM6's batch count.
    resident_loaders = {}
    if residents:
        mols_per_batch = {
            "pcba": getattr(hp.pretrain, "pcba_mols_per_batch", 0),
        }
        num_batches = len(train_loader)
        resident_loaders = build_resident_dataloaders(
            residents=residents,
            mols_per_batch=mols_per_batch,
            num_batches=num_batches,
            world_size=dist_cfg.world_size,
            rank=dist_cfg.rank,
            n_workers=0,  # Data is already in SHM; workers add overhead with no I/O benefit.
            seed=(hp.pretrain.seed + shard_idx) if hp.pretrain.seed is not None else None,
        )

    if dist_cfg.is_distributed:
        dist.barrier()

    shard_loader.cleanup_previous_shard(shm_metadata)

    resident_iters = {name: iter(loader) for name, loader in resident_loaders.items()}

    for batch_idx, (batch, y_graph) in enumerate(train_loader):
        extra_targets = None
        if resident_iters:
            extra = {}
            for name, it in resident_iters.items():
                try:
                    extra[name] = next(it)
                except StopIteration:
                    # Resident sampler exhausted before PM6 did (NodeBudget-
                    # BatchSampler's __len__ and actual __iter__ count can drift
                    # by 1-2 per shard). Restart the resident iterator and
                    # continue; sampling is random so cycling has no correctness
                    # impact.
                    resident_iters[name] = iter(resident_loaders[name])
                    extra[name] = next(resident_iters[name])
            batch, y_dict = concat_multi_source(batch, y_graph, extra)
            y_graph = y_dict["pm6"]
            extra_targets = {k: v for k, v in y_dict.items() if k != "pm6"}

        train_one_batch(
            model=model,
            base_model=base_model,
            batch=batch,
            y_graph=y_graph,
            optimizer=optimizer,
            scheduler=scheduler,
            tracker=tracker,
            task_info=task_info,
            hp=hp,
            dist_cfg=dist_cfg,
            shard_idx=shard_idx,
            batch_idx=batch_idx,
            ema=ema,
            extra_targets=extra_targets,
        )

    train_loss, per_task_losses, train_metrics = tracker.score()
    tracker.reset()
    base_model.update_train_loss_buffer(shard_idx, per_task_losses)

    del train_loader
    gc.collect()

    return train_loss, per_task_losses, train_metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    hp = parse_args()
    dist_cfg = setup_distributed()
    compile_model = getattr(hp, "compile", False)

    # Seed RNGs
    np.random.seed(hp.pretrain.seed + dist_cfg.rank)
    random.seed(hp.pretrain.seed + dist_cfg.rank)

    # Load or build model
    ema_state_dict = None
    if hp.load_ckpt:
        hp_dict, weights_sd, training_state, ema_state_dict = load_training_ckpt(
            ckpt_path=hp.exp_dir,
            device=dist_cfg.device,
        )
        hp = dict_to_namespace(hp_dict)
        if not hasattr(hp, "distributed"):
            hp.distributed = dict_to_namespace(
                {
                    "world_size": dist_cfg.world_size,
                    "rank": dist_cfg.rank,
                    "local_rank": dist_cfg.local_rank,
                }
            )
        task_dict = build_task_dict(
            use_node_labels=hp.pretrain.use_node_labels,
            structure_loss=hp.pretrain.structure_loss,
            beta_nll_softplus=getattr(hp.pretrain, "beta_nll_softplus", True),
            per_task_ordinal_k=getattr(hp.pretrain, "per_task_ordinal_k", False),
            exclude_loss_types=getattr(hp.pretrain, "exclude_loss_types", None),
            exclude_task_states=getattr(hp.pretrain, "exclude_task_states", None),
        )
        task_dict = _maybe_add_multisource(task_dict, hp)
        model, optimizer, scheduler = build_model(hp, task_dict, dist_cfg.device)
        # strict=False lets us resume PM6-only checkpoints into multi-source models
        # (the new dataset_heads.* keys will be randomly initialized and logged).
        _load_with_logging(model, weights_sd, dist_cfg.is_main_process)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        model.shard = training_state["shard"]
        model.train_loss_buffer = training_state["train_loss_buffer"].to(dist_cfg.device)
        weighting_state = training_state.get("weighting_state", {})
        if weighting_state:
            model.deserialize_weighting_state(weighting_state)
        run_id = training_state.get("run_id")
        start_shard = model.shard + 1
    else:
        task_dict = build_task_dict(
            use_node_labels=hp.pretrain.use_node_labels,
            structure_loss=hp.pretrain.structure_loss,
            beta_nll_softplus=hp.pretrain.beta_nll_softplus,
            per_task_ordinal_k=getattr(hp.pretrain, "per_task_ordinal_k", False),
            exclude_loss_types=getattr(hp.pretrain, "exclude_loss_types", None),
            exclude_task_states=getattr(hp.pretrain, "exclude_task_states", None),
        )
        task_dict = _maybe_add_multisource(task_dict, hp)
        model, optimizer, scheduler = build_model(hp, task_dict, dist_cfg.device)

        # Optionally load weights from a different experiment (e.g., curriculum pretraining)
        # Loads all compatible keys (encoder + any matching decoders) with strict=False
        load_encoder_from = getattr(hp, "load_encoder_from", None)
        if load_encoder_from:
            _, src_weights, _, src_ema = load_training_ckpt(load_encoder_from, device=dist_cfg.device)
            src_state = src_ema if src_ema is not None else src_weights
            # Load all keys that exist in the target model (encoder + compatible decoders)
            tgt_keys = set(model.state_dict().keys())
            compatible_state = {k: v for k, v in src_state.items() if k in tgt_keys}
            model.load_state_dict(compatible_state, strict=False)
            if dist_cfg.is_main_process:
                n_enc = sum(1 for k in compatible_state if k.startswith("encoder."))
                n_dec = sum(1 for k in compatible_state if k.startswith("decoders."))
                printf(f"Loaded weights from {load_encoder_from} ({n_enc} encoder, {n_dec} decoder params)")
            del src_weights, src_ema, src_state, compatible_state

        run_id = None
        start_shard = 0

    # Set up EMA (before compile — deepcopy of compiled modules causes recompilation
    # which can take different amounts of time per rank, triggering NCCL timeouts)
    ema_decay = getattr(hp.pretrain, "ema_decay", 0.0)
    ema = None
    if ema_decay > 0:
        ema = ModelEMA(model, decay=ema_decay)
        if ema_state_dict is not None:
            ema.load_state_dict(ema_state_dict)
        if dist_cfg.is_main_process:
            printf(f"EMA enabled with decay={ema_decay}")

    # Compile dense subgraphs for kernel fusion (before DDP wrapping)
    if compile_model:
        torch._dynamo.config.force_parameter_static_shapes = False

        for layer in model.encoder.grit_layers:
            layer._post_attn_ffn = torch.compile(layer._post_attn_ffn, dynamic=True)
        for gkey in model.decoders.groups:
            model.decoders.groups[gkey] = torch.compile(model.decoders.groups[gkey], dynamic=True)
        # Compile dataset-level heads the same way (dense Linear+LayerNorm pipeline).
        for name in list(model.dataset_heads.keys()):
            model.dataset_heads[name] = torch.compile(model.dataset_heads[name], dynamic=True)
        if dist_cfg.is_main_process:
            printf(
                f"torch.compile enabled for GRIT layers, decoders, "
                f"and {len(model.dataset_heads)} dataset heads"
            )

    # Wrap for distributed training
    model = wrap_model_distributed(model, dist_cfg, sync_batchnorm=hp.sync_batchnorm)
    base_model = model.module if hasattr(model, "module") else model

    # Set up validation
    val_loader, task_info = setup_validation(hp, task_dict, dist_cfg)

    # Optional PCBA val eval — separate one-shot DataLoader over the PCBA val
    # CSR shard. Runs as a second val pass with real PCBA targets, logged under
    # "val_pcba" / "val_pcba_ema". Without this, PCBA val metrics are 0
    # (the PM6 val pass injects all-NaN PCBA targets for tracker consistency).
    pcba_val_loader = None
    pcba_val_dir = getattr(hp.pretrain, "pcba_val_dir", None)
    pcba_in_task_dict = (
        "pcba" in task_dict
        or any(info.get("pcba_assay_idx") is not None for info in task_dict.values())
    )
    if pcba_val_dir and pcba_in_task_dict:
        pcba_val_loader = build_pcba_val_dataloader(
            shard_dir=pcba_val_dir,
            batch_node_budget=hp.pretrain.train_node_budget,
            walk_len=hp.encoder.walk_len,
            world_size=dist_cfg.world_size,
            rank=dist_cfg.rank,
            n_workers=0,
        )
        if dist_cfg.is_main_process:
            printf(f"PCBA val loader built from {pcba_val_dir}")

    # Log model info
    enc_params, dec_params, n_decs = count_parameters(base_model)
    if dist_cfg.is_main_process:
        print(json.dumps(to_dict(hp), indent=4))
        printf(f"Encoder params: {enc_params}, per-decoder params: {dec_params}, # decoders: {n_decs}")

    # Initialize wandb and tracker
    init_wandb(hp, run_id, enc_params, dec_params, n_decs, dist_cfg.is_main_process)
    tracker = Tracker(task_dict, device=base_model.device)
    shard_loader = ShardLoader(hp, dist_cfg)

    # Load resident datasets once into SHM. These live for the duration of training
    # and are never cleaned up (distinct from the rotating PM6 shards). Empty dict
    # if no --pcba-dir flag was passed.
    residents = load_resident_datasets(hp, dist_cfg)
    if residents and dist_cfg.is_main_process:
        sizes = {name: len(r.dataset) for name, r in residents.items()}
        printf(f"Resident datasets loaded: {sizes}")

    # Build STCH preference vector from --STCH-pref (e.g. "ordinal:0.3,huber:1.0")
    stch_pref_tensor = None
    stch_pref_str = getattr(hp.pretrain.mtl.weighting_args, "STCH_pref", None)
    if stch_pref_str:
        pref_by_type = {}
        for part in stch_pref_str.split(","):
            ltype, val = part.strip().split(":")
            pref_by_type[ltype.strip()] = float(val.strip())
        pref_vec = []
        for tname, tinfo in task_dict.items():
            ltype = tinfo.get("loss_type", "unknown")
            pref_vec.append(pref_by_type.get(ltype, 1.0))
        stch_pref_tensor = torch.tensor(pref_vec, device=base_model.device)
        train_one_batch._stch_pref_tensor = stch_pref_tensor
        if dist_cfg.is_main_process:
            printf(f"STCH preference vector: {dict(zip(task_dict.keys(), pref_vec))}")

    # Initial evaluation (before any training)
    if not hp.load_ckpt:
        val_loss, val_losses, val_metrics = eval_model(
            model,
            val_loader,
            tracker,
            task_info,
            structure_loss=hp.pretrain.structure_loss,
        )
        tracker.log(
            split="val",
            shard=0,
            update_dict={"val_loss": val_loss},
            log_to_wandb=dist_cfg.is_main_process,
            mean_losses=val_losses,
            mean_metrics=val_metrics,
        )
        if pcba_val_loader is not None:
            pcba_val_loss, pcba_val_losses, pcba_val_metrics = eval_pcba_val(
                model, pcba_val_loader, tracker, task_info
            )
            tracker.log(
                split="val_pcba",
                shard=0,
                update_dict={"val_pcba_loss": pcba_val_loss},
                log_to_wandb=dist_cfg.is_main_process,
                mean_losses=pcba_val_losses,
                mean_metrics=pcba_val_metrics,
            )
        if ema is not None:
            ema_loss, ema_losses, ema_metrics = eval_model(
                ema.model,
                val_loader,
                tracker,
                task_info,
                structure_loss=hp.pretrain.structure_loss,
            )
            tracker.log(
                split="val_ema",
                shard=0,
                update_dict={"val_ema_loss": ema_loss},
                log_to_wandb=dist_cfg.is_main_process,
                mean_losses=ema_losses,
                mean_metrics=ema_metrics,
            )
            if pcba_val_loader is not None:
                ema_pcba_loss, ema_pcba_losses, ema_pcba_metrics = eval_pcba_val(
                    ema.model, pcba_val_loader, tracker, task_info
                )
                tracker.log(
                    split="val_pcba_ema",
                    shard=0,
                    update_dict={"val_pcba_ema_loss": ema_pcba_loss},
                    log_to_wandb=dist_cfg.is_main_process,
                    mean_losses=ema_pcba_losses,
                    mean_metrics=ema_pcba_metrics,
                )
        if dist_cfg.is_main_process:
            printf(f"Pre-training eval, loss: {val_loss:.2f}")

        # Save randomly initialized checkpoint before training starts
        if hp.save_ckpt and dist_cfg.is_main_process:
            save_ckpt(
                model,
                hp,
                optimizer,
                scheduler,
                ema_state_dict=ema.state_dict() if ema is not None else None,
            )
            printf("Saved randomly initialized checkpoint")

    # Shuffle shard order (deterministic across all DDP ranks via dedicated RNG)
    shard_order = list(range(hp.pretrain.n_shards))
    shard_rng = random.Random(hp.pretrain.seed)
    shard_rng.shuffle(shard_order)
    if dist_cfg.is_main_process:
        printf(f"Shard order (first 10): {shard_order[:10]}...")

    # Prefetch first shard
    if start_shard < hp.pretrain.n_shards:
        shard_loader.prefetch_first_shard(shard_order[start_shard])

    # Main training loop
    for shard_pos in range(start_shard, hp.pretrain.n_shards):
        data_shard_id = shard_order[shard_pos]
        next_shard_id = shard_order[shard_pos + 1] if shard_pos + 1 < hp.pretrain.n_shards else None

        train_loss, per_task_losses, train_metrics = train_one_shard(
            model=model,
            base_model=base_model,
            shard_idx=shard_pos,
            hp=hp,
            task_info=task_info,
            dist_cfg=dist_cfg,
            tracker=tracker,
            optimizer=optimizer,
            scheduler=scheduler,
            shard_loader=shard_loader,
            data_shard_id=data_shard_id,
            next_shard_id=next_shard_id,
            ema=ema,
            residents=residents,
        )

        task_weights = base_model.get_task_weights()
        base_model.reset_task_weights()

        lr = scheduler.get_last_lr()[0]

        # Evaluate periodically
        val_results = None
        ema_val_results = None
        pcba_val_results = None
        ema_pcba_val_results = None
        if shard_pos % hp.pretrain.eval_freq == 0:
            val_results = eval_model(
                model,
                val_loader,
                tracker,
                task_info,
                structure_loss=hp.pretrain.structure_loss,
            )
            if pcba_val_loader is not None:
                tracker.reset()
                pcba_val_results = eval_pcba_val(
                    model, pcba_val_loader, tracker, task_info
                )
            if ema is not None:
                tracker.reset()
                ema_val_results = eval_model(
                    ema.model,
                    val_loader,
                    tracker,
                    task_info,
                    structure_loss=hp.pretrain.structure_loss,
                )
                if pcba_val_loader is not None:
                    tracker.reset()
                    ema_pcba_val_results = eval_pcba_val(
                        ema.model, pcba_val_loader, tracker, task_info
                    )

        tracker.log_shard_results(
            shard_idx=shard_pos,
            train_loss=train_loss,
            per_task_losses=per_task_losses,
            train_metrics=train_metrics,
            lr=lr,
            is_main_process=dist_cfg.is_main_process,
            val_results=val_results,
            ema_val_results=ema_val_results,
            pcba_val_results=pcba_val_results,
            ema_pcba_val_results=ema_pcba_val_results,
            task_weights=task_weights,
        )

        # Save checkpoint
        if hp.save_ckpt and dist_cfg.is_main_process and shard_pos != 0:
            save_ckpt(
                model,
                hp,
                optimizer,
                scheduler,
                ema_state_dict=ema.state_dict() if ema is not None else None,
            )

    # Cleanup
    if dist_cfg.is_distributed:
        dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
