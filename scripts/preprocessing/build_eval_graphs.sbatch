#!/bin/bash
#SBATCH --job-name=eval-graphs
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --hint=nomultithread
#SBATCH --time=0-8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# CPU array job that precomputes the molecular graphs the evaluation benchmarks need.
#
# Usage:
#   sbatch scripts/preprocessing/build_eval_graphs.sbatch \
#       --benchmark polaris --output-path data/polaris/
#   sbatch scripts/preprocessing/build_eval_graphs.sbatch \
#       --benchmark moleculeace --output-path data/moleculeace/
#
# No SLURM? Run it unsharded:
#   python -m monroe.preprocessing.build_eval_graphs --benchmark polaris --output-path data/polaris/

set -euo pipefail

MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"   # venv activation + thread caps (see .env.example)

# All args are passed through to build_eval_graphs.
srun -u python -u -m monroe.preprocessing.build_eval_graphs \
  --num-shards 10 \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  "$@"
