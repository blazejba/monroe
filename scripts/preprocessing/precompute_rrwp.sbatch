#!/bin/bash
#SBATCH --job-name=MONROE-rrwp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-101%3
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# GPU array job for RRWP precomputation, one shard per array task.
#
# Usage (PM6):
#   sbatch scripts/preprocessing/precompute_rrwp.sbatch
#
# Usage (PCBA):
#   sbatch --array=0-19 scripts/preprocessing/precompute_rrwp.sbatch \
#       --shard-dir data/pcba_monroe/train/built --max-nodes 500
#
# No SLURM? Omit --shard-prefix and one process does every shard:
#   python scripts/preprocessing/precompute_rrwp.py --shard-dir data/pm6/ --walk-len 16

set -euo pipefail

MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"   # venv activation + thread caps (see .env.example)

# If your cluster uses environment modules, load CUDA here, e.g.:
#   module load cuda/12.8

if [ "$#" -eq 0 ]; then
    set -- --shard-dir data/pm6/ --max-nodes 200
fi

srun -u python -u scripts/preprocessing/precompute_rrwp.py \
    --shard-prefix "$SLURM_ARRAY_TASK_ID" \
    --walk-len 16 \
    "$@"
