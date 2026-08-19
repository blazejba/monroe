#!/bin/bash
#SBATCH --job-name=MONROE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   Single GPU (default):
#     sbatch scripts/training/train.sbatch
#
#   Multi GPU: override --gres at submission time
#     sbatch --gres=gpu:4 scripts/training/train.sbatch
#
#   Override any training arg via passthrough:
#     sbatch scripts/training/train.sbatch --lr 1e-4 --n-shards 50
#
#   Train on PM6 + PCBA (the published setup; see README):
#     sbatch scripts/training/train.sbatch --pcba-dir data/pcba_monroe/train/consolidated
#
#   Resume from checkpoint:
#     sbatch scripts/training/train.sbatch --load-ckpt --exp-dir results/STCH_20250601
#
# All training args are defined in monroe/config.py (run with --help to see them).
#
# No SLURM? Run torchrun directly:
#   torchrun --nproc_per_node 1 monroe/run_train.py --pm6-dir data/pm6

set -euo pipefail

# Repo root: the directory you submit from (SLURM copies this script elsewhere, so
# its own path is not usable). Override by exporting MONROE_ROOT before submitting.
MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
DATA_DIR="${MONROE_DATA_DIR:-$MONROE_ROOT/data/pm6}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"

# If your cluster uses environment modules, load CUDA here, e.g.:
#   module load cuda/12.8

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
N_WORKERS="${SLURM_CPUS_PER_GPU:-16}"
# Derive a per-job master port so concurrent jobs don't collide.
MASTER_PORT=$(expr 10000 + $(echo -n "${SLURM_JOB_ID:-$$}" | tail -c 4))

srun -u --ntasks=1 \
    torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
    monroe/run_train.py \
    --pm6-dir "$DATA_DIR" \
    --n-workers "$N_WORKERS" \
    "$@"
