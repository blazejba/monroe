"""Distributed training utilities."""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    world_size: int
    rank: int
    local_rank: int
    is_distributed: bool
    device: torch.device
    is_main_process: bool
    is_local_main: bool


def setup_distributed() -> DistributedConfig:
    """Initialize distributed training environment.

    Reads WORLD_SIZE, RANK, LOCAL_RANK from environment (set by torchrun).
    Initializes process group if distributed.

    Returns:
        DistributedConfig with all distributed training state.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_distributed:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
        )

    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    return DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device,
        is_main_process=(not is_distributed) or (rank == 0),
        is_local_main=(not is_distributed) or (local_rank == 0),
    )


def wrap_model_distributed(model, dist_cfg: DistributedConfig, sync_batchnorm: bool = False):
    """Wrap model for distributed training if needed.

    Args:
        model: The model to wrap.
        dist_cfg: Distributed configuration.
        sync_batchnorm: If True, convert BatchNorm layers to SyncBatchNorm.

    Returns:
        The wrapped model (or original if not distributed).
    """
    if not dist_cfg.is_distributed:
        return model

    if sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_cfg.local_rank] if dist_cfg.device.type == "cuda" else None,
        output_device=dist_cfg.local_rank if dist_cfg.device.type == "cuda" else None,
        find_unused_parameters=False,
        static_graph=True,
    )
