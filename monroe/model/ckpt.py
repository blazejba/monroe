import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

import wandb
from monroe.config import to_dict
from monroe.model.constants import (
    EDGE_FEAT_LIST_ONE_HOT,
    NODE_FEAT_LIST_FLOAT,
    NODE_FEAT_LIST_ONE_HOT,
)
from monroe.model.grit import GritTransformer


def _resolve_ckpt_dir(path: str) -> Path:
    """Resolve a checkpoint path to the directory containing config/weights.

    If the path itself contains config.json, use it directly.
    Otherwise, find the latest checkpoint-* subdirectory.
    Accepts a file path (uses its parent directory).
    """
    ckpt_dir = Path(path)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    if not (ckpt_dir / "config.json").exists():
        checkpoints = sorted(
            list(ckpt_dir.glob("checkpoint-*")),
            key=lambda x: int(x.name.split("-")[-1]),
        )
        if checkpoints:
            ckpt_dir = checkpoints[-1]
    return ckpt_dir


def _load_state_dict(weights_path: Path, device=None) -> dict:
    """Load a state dict, stripping the torch.compile ``_orig_mod.`` prefix."""
    kwargs = {"weights_only": False}
    if device is not None:
        kwargs["map_location"] = device
    state_dict = torch.load(weights_path, **kwargs)
    # Strip _orig_mod. prefix inserted by torch.compile so that
    # weights load correctly into non-compiled modules.
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def _load_config_and_weights(ckpt_dir: Path, device=None, use_ema: bool = False):
    """Load config dict and state dict from a checkpoint directory.

    Args:
        ckpt_dir: Path to checkpoint directory.
        device: Target torch device for map_location.
        use_ema: If True, load ema_weights.pt instead of weights.pt.
    """
    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / ("ema_weights.pt" if use_ema else "weights.pt")

    if use_ema and not weights_path.exists():
        raise FileNotFoundError(f"EMA weights not found at {weights_path}")

    with config_path.open("r") as f:
        hp_dict = json.load(f)

    return hp_dict, _load_state_dict(weights_path, device=device)


def _build_encoder(hp_dict: dict) -> GritTransformer:
    """Construct a GritTransformer encoder from a saved config dict."""
    encoder_cfg = dict(hp_dict["encoder"])
    # Backward compat: old checkpoints have "edge_rbf_dim", new have "rbf_dim"
    if "edge_rbf_dim" in encoder_cfg and "rbf_dim" not in encoder_cfg:
        encoder_cfg["rbf_dim"] = encoder_cfg.pop("edge_rbf_dim")
    elif "edge_rbf_dim" in encoder_cfg:
        encoder_cfg.pop("edge_rbf_dim")
    # Strip config keys that are no longer constructor params (now always enabled)
    for key in ["node_float_rbf", "node_float_missing"]:
        encoder_cfg.pop(key, None)
    return GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **encoder_cfg,
    )


def load_ckpt(path: str, use_ema: bool = False):
    """Load a checkpoint for inference (encoder only).

    ``path`` is a checkpoint directory containing ``config.json`` plus
    ``weights.pt`` (or ``ema_weights.pt`` when ``use_ema=True``) — the
    training-output layout, which the bundled model in ``checkpoint/`` also uses.
    """
    ckpt_dir = _resolve_ckpt_dir(path)
    hp_dict, state_dict = _load_config_and_weights(ckpt_dir, use_ema=use_ema)

    class Monroe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _build_encoder(hp_dict)

    model = Monroe()
    model.load_state_dict(state_dict, strict=False)

    return model.encoder


def load_training_ckpt(ckpt_path: str, device: torch.device):
    """Load checkpoint for training resume.

    Args:
        ckpt_path: Path to checkpoint directory or experiment directory.
        device: Target torch device.

    Returns:
        Tuple of (hp_dict, weights_state_dict, training_state, ema_state_dict).
        ema_state_dict is None if no EMA weights were saved.
    """
    ckpt_dir = _resolve_ckpt_dir(ckpt_path)
    hp_dict, weights_state_dict = _load_config_and_weights(ckpt_dir, device=device)

    state_path = ckpt_dir / "state.pt"
    training_state = torch.load(state_path, map_location=device, weights_only=False)

    ema_path = ckpt_dir / "ema_weights.pt"
    ema_state_dict = None
    if ema_path.exists():
        ema_state_dict = torch.load(ema_path, map_location=device, weights_only=False)

    return hp_dict, weights_state_dict, training_state, ema_state_dict


def save_ckpt(
    model: torch.nn.Module,
    hparams: SimpleNamespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    overwrite: bool = False,
    ema_state_dict: dict | None = None,
):
    """
    Saves three objects: config.json, weights.pt and state.pt.
    - config.json contains the hparams
    - weights.pt contains the model weights
    - state.pt contains the optimizer, scheduler, epoch, and weighting state
    """
    base_model = model.module if hasattr(model, "module") else model

    if overwrite:
        checkpoint_dir = Path(hparams.exp_dir)
    else:
        checkpoint_dir = Path(hparams.exp_dir) / f"checkpoint-{base_model.shard}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / 'config.json'
    weights_path = checkpoint_dir / 'weights.pt'
    state_path = checkpoint_dir / 'state.pt'

    with config_path.open('w') as f:
        json.dump(to_dict(hparams), f, indent=2)

    torch.save(base_model.state_dict(), weights_path)

    weighting_state = base_model.serialize_weighting_state()

    run_id = None
    if getattr(wandb, "run", None) is not None and wandb.run is not None:
        run_id = wandb.run.id

    state = {
        'run_id': run_id,
        'shard': base_model.shard,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'weighting_state': weighting_state,
        'train_loss_buffer': base_model.train_loss_buffer.detach().cpu(),
    }

    torch.save(state, state_path)

    if ema_state_dict is not None:
        ema_path = checkpoint_dir / "ema_weights.pt"
        torch.save(ema_state_dict, ema_path)

    keep_ckpts = getattr(hparams, "keep_ckpts", None)
    if keep_ckpts is None:
        keep_ckpts = 0 if getattr(hparams, "keep_all_ckpts", False) else -1
    keep_warmup_shards = int(getattr(hparams, "keep_warmup_shards", 0) or 0)

    checkpoints = sorted(list(checkpoint_dir.parent.glob('checkpoint-*')), key=lambda x: int(x.name.split('-')[-1]))
    for ckpt in checkpoints[:-1]:
        ckpt_epoch = int(ckpt.name.split('-')[-1])
        # During warmup, only the latest ckpt is retained — drop everything older.
        if keep_warmup_shards > 0 and ckpt_epoch < keep_warmup_shards:
            shutil.rmtree(ckpt)
            continue
        # After warmup (or always when warmup is 0), apply the keep_ckpts policy.
        if keep_ckpts == 0:
            continue                    # 0 = keep all (legacy semantic in this branch)
        if keep_ckpts > 0 and ckpt_epoch % keep_ckpts == 0:
            continue                    # multiple of keep_ckpts → retain
        shutil.rmtree(ckpt)
