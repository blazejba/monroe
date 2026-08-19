import argparse
import glob
import os
from datetime import datetime
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from monroe.utils import str2bool


def _count_train_nodes(data_dir: str, n_shards: int) -> int:
    """Count total nodes across training shards by reading node_ptr files.

    When n_shards exceeds available shards, cycles through them (multi-epoch).
    """
    ptrs = sorted(glob.glob(os.path.join(data_dir, "*.node_ptr.npy")))
    # Last shard is reserved for validation
    train_ptrs = ptrs[:-1]
    if len(train_ptrs) == 0:
        raise ValueError(f"No training shards found in {data_dir}")
    # Count nodes per shard once
    nodes_per_shard = [int(np.load(f, mmap_mode="r")[-1]) for f in train_ptrs]
    n_available = len(train_ptrs)
    full_epochs, remainder = divmod(n_shards, n_available)
    total = full_epochs * sum(nodes_per_shard) + sum(nodes_per_shard[:remainder])
    return total


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def parse_args():
    parser = argparse.ArgumentParser()
    ## misc
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wandb", default=True, type=str2bool)
    parser.add_argument("--exp-dir", type=str, default=None)
    parser.add_argument("--load-ckpt", action="store_true")
    parser.add_argument("--load-encoder-from", type=str, default=None,
                        help="Load encoder weights (not decoders/optimizer) from another experiment dir")
    parser.add_argument("--save-ckpt", default=True, type=str2bool)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--keep-ckpts", default=10, type=int)
    parser.add_argument("--keep-warmup-shards", default=0, type=int,
                        help="If >0, retain only the latest checkpoint while shard < N; "
                             "after that, the regular --keep-ckpts policy applies. "
                             "Use to skip saving early ckpts that aren't useful.")
    parser.add_argument("--compile", default=True, type=str2bool)
    parser.add_argument("--sync-batchnorm", default=True, type=str2bool)

    ## encoder
    parser.add_argument("--model-encoder-hidden-dim", default=720, type=int)
    parser.add_argument("--model-encoder-emb-dim", default=128, type=int)
    parser.add_argument("--model-encoder-walk-len", default=16, type=int)
    parser.add_argument("--model-encoder-num-layers", default=10, type=int)
    parser.add_argument("--model-encoder-dropout", default=0.05, type=float)
    parser.add_argument("--model-encoder-num-heads", default=10, type=int)
    parser.add_argument("--model-encoder-rbf-dim", "--model-encoder-edge-rbf-dim", default=32, type=int, dest="model_encoder_rbf_dim")
    parser.add_argument("--use-stereo-edges", default=True, type=str2bool)
    parser.add_argument("--zero-vn-edge-rbf", default=False, type=str2bool, help="Zero out edge RBF for virtual node edges (removes frame-dependent distances)")

    ## pretraining
    ### data
    parser.add_argument("--subset-ratio", default=1.0, type=float)
    # Flag mirrors --pcba-dir; dest stays data_dir so resuming older checkpoints
    # (whose saved config uses that key) keeps working.
    parser.add_argument("--pm6-dir", dest="data_dir", default="data/pm6", type=str,
                        help="PM6 CSR shard directory (the main pre-training source).")
    parser.add_argument("--load-to-memory", default=True, type=str2bool)
    parser.add_argument("--n-shards", default=100, type=int)
    parser.add_argument("--n-workers", default=16, type=int)
    parser.add_argument("--use-node-labels", action="store_true")
    ### multi-source (resident datasets loaded once into shared memory)
    parser.add_argument("--pcba-dir", default=None, type=str,
                        help="PCBA CSR shard dir (train split). Enables PCBA focal-loss head.")
    parser.add_argument("--pcba-val-dir", default=None, type=str,
                        help="Optional PCBA val CSR shard dir. Logs a true val_pcba AUROC.")
    parser.add_argument("--pcba-mols-per-batch", default=100, type=int,
                        help="PCBA molecules sampled per training step, split across ranks.")
    parser.add_argument("--pcba-head-layers", default=0, type=int,
                        help="Hidden blocks in the PCBA head. 0 = linear probe.")
    parser.add_argument("--pcba-per-assay-stch", default=False, type=str2bool,
                        help="Give each PCBA assay its own STCH task instead of one aggregate slot.")
    ### training
    parser.add_argument("--eval-freq", default=2, type=int)
    parser.add_argument("--train-node-budget", default=100_000, type=int)

    parser.add_argument("--structure-loss", default=True, type=str2bool)
    parser.add_argument("--beta-nll-softplus", default=True, type=str2bool)
    parser.add_argument("--per-task-ordinal-k", default=True, type=str2bool,
                        help="Use per-task K for ordinal loss (old LibMTL behavior) instead of global K=82")
    parser.add_argument("--exclude-loss-types", default="", type=str,
                        help="Comma-separated loss types to exclude (e.g. 'ordinal' or 'ordinal,beta_nll')")
    parser.add_argument("--exclude-task-states", default="", type=str,
                        help="Comma-separated electronic states to exclude (e.g. 'anion,cation,T0')")
    ### decoder
    parser.add_argument("--model-decoder-channels", default=340, type=int)
    parser.add_argument("--model-decoder-num-layers", default=1, type=int)
    parser.add_argument("--model-decoder-dropout", default=0.2, type=float)
    parser.add_argument("--model-decoder-normalization", default="layernorm", type=str)
    ### optimizer and scheduler
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-shards", default=10, type=int)
    parser.add_argument("--cosine-alpha", default=1.0, type=float)
    ### mtl
    parser.add_argument("--weighting", default="STCH", type=str)
    #### STCH
    parser.add_argument("--STCH-mu", default=1.5, type=float)
    parser.add_argument("--STCH-mu-end", default=0.5, type=float)
    parser.add_argument("--STCH-warmup-shard", default=4, type=int)
    parser.add_argument("--STCH-ramp", default=True, type=str2bool)
    parser.add_argument("--STCH-log", default=True, type=str2bool)
    parser.add_argument("--STCH-nadir-refresh", default=0, type=int, help="Recalibrate nadir every N shards (0=never, default old behavior)")
    parser.add_argument("--STCH-pref", default=None, type=str,
                        help="Per-loss-type preference weights, e.g. 'ordinal:0.3,huber:1.0'. "
                             "Unspecified types default to 1.0. Lower = less preferred.")
    #### DWA
    parser.add_argument("--DWA-T", default=2.0, type=float)
    ### regularization
    parser.add_argument("--emb-decorr-weight", default=0.0, type=float,
                        help="Weight for off-diagonal covariance penalty on pooled embeddings (0=disabled)")
    ### ema
    parser.add_argument("--ema-decay", default=0.0, type=float)

    args = parser.parse_args()

    if args.weighting == "EW":
        weighting_args = {}
    elif args.weighting == "UW":
        weighting_args = {}
    elif args.weighting == "RLW":
        weighting_args = {}
    elif args.weighting == "STCH":
        weighting_args = {
            "STCH_mu": args.STCH_mu,
            "STCH_mu_end": args.STCH_mu_end,
            "STCH_warmup_shard": args.STCH_warmup_shard,
            "STCH_ramp": args.STCH_ramp,
            "STCH_log": args.STCH_log,
            "STCH_total_shards": args.n_shards,
            "STCH_nadir_refresh": args.STCH_nadir_refresh,
            "STCH_pref": args.STCH_pref,
        }
    elif args.weighting == "DWA":
        weighting_args = {"DWA_T": args.DWA_T}
    else:
        raise ValueError(f"Unsupported weighting method: {args.weighting}")

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist_world_size = args.world_size or env_world_size
    dist_rank = args.rank if args.rank is not None else env_rank
    dist_local_rank = args.local_rank if args.local_rank is not None else env_local_rank
    dist_world_size = max(1, dist_world_size)

    optim_param = {"lr": args.lr, "weight_decay": args.weight_decay}

    train_node_budget_global = args.train_node_budget
    train_node_budget = train_node_budget_global
    if dist_world_size > 1:
        train_node_budget = max(1, int(ceil(train_node_budget / dist_world_size)))

    total_train_nodes = _count_train_nodes(args.data_dir, args.n_shards)
    total_steps = max(1, int(total_train_nodes // train_node_budget_global))
    assert args.warmup_shards < args.n_shards, "Warmup shards must be less than total shards"
    warmup_steps = int((args.warmup_shards / args.n_shards) * total_steps)

    if args.exp_dir is None:
        datatime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        exp_dir = Path("results") / f"{args.weighting}_{datatime}"
    else:
        exp_dir = args.exp_dir

    hparams_dict = {
        "wandb": args.wandb,
        "exp_dir": str(exp_dir),
        "load_ckpt": args.load_ckpt,
        "load_encoder_from": args.load_encoder_from,
        "save_ckpt": args.save_ckpt,
        "keep_ckpts": args.keep_ckpts,
        "keep_warmup_shards": args.keep_warmup_shards,
        "compile": args.compile,
        "sync_batchnorm": args.sync_batchnorm,
        "distributed": {
            "world_size": dist_world_size,
            "rank": dist_rank,
            "local_rank": dist_local_rank,
        },
        "encoder": {
            "hidden_dim": args.model_encoder_hidden_dim,
            "num_layers": args.model_encoder_num_layers,
            "num_heads": args.model_encoder_num_heads,
            "emb_dim": args.model_encoder_emb_dim,
            "walk_len": args.model_encoder_walk_len,
            "rbf_dim": args.model_encoder_rbf_dim,
            "dropout": args.model_encoder_dropout,
            "use_stereo_edges": args.use_stereo_edges,
            "zero_vn_edge_rbf": args.zero_vn_edge_rbf,
        },
        "pretrain": {
            "load_to_memory": args.load_to_memory,
            "structure_loss": args.structure_loss,
            "beta_nll_softplus": args.beta_nll_softplus,
            "per_task_ordinal_k": args.per_task_ordinal_k,
            "exclude_loss_types": [s.strip() for s in args.exclude_loss_types.split(",") if s.strip()],
            "exclude_task_states": [s.strip() for s in args.exclude_task_states.split(",") if s.strip()],
            "seed": args.seed,
            "data_dir": args.data_dir,
            "pcba_dir": args.pcba_dir,
            "pcba_val_dir": args.pcba_val_dir,
            "pcba_mols_per_batch": args.pcba_mols_per_batch,
            "pcba_head_layers": args.pcba_head_layers,
            "pcba_per_assay_stch": args.pcba_per_assay_stch,
            "n_shards": args.n_shards,
            "n_workers": args.n_workers,
            "subset_ratio": args.subset_ratio,
            "eval_freq": args.eval_freq,
            "train_node_budget": train_node_budget,
            "train_node_budget_global": train_node_budget_global,
            "ema_decay": args.ema_decay,
            "emb_decorr_weight": args.emb_decorr_weight,
            "use_node_labels": args.use_node_labels,
            "decoder": {
                "hidden_dim": args.model_decoder_channels,
                "num_layers": args.model_decoder_num_layers,
                "dropout": args.model_decoder_dropout,
                "normalization": args.model_decoder_normalization,
            },
            "mtl": {
                "weighting": args.weighting,
                "weighting_args": weighting_args,
            },
            "optim_param": optim_param,
            "scheduler_param": {
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "cosine_alpha": args.cosine_alpha,
            },
        },
    }

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    return dict_to_namespace(hparams_dict)
