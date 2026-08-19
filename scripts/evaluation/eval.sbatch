#!/bin/bash
#SBATCH --job-name=MONROE-EVAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# SLURM launcher for evaluation (MoleculeACE + Polaris).
#
# Usage:
#   sbatch scripts/evaluation/eval.sbatch --ckpt-path results/STCH_20250601
#   sbatch scripts/evaluation/eval.sbatch --ckpt-path ... --use-ema
#   sbatch scripts/evaluation/eval.sbatch --ckpt-path ... --skip-moleculeace
#
# All eval args are defined in monroe/run_eval.py (run with --help to see them).
#
# No SLURM? Run it directly:
#   python -m monroe.run_eval --ckpt-path results/STCH_20250601

set -euo pipefail

MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"   # venv activation + wandb/thread env (see .env.example)

# If your cluster uses environment modules, load CUDA here, e.g.:
#   module load cuda/12.8

srun -u python -m monroe.run_eval "$@"
