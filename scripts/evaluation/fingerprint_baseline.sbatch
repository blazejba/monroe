#!/bin/bash
#SBATCH --job-name=MONROE-GBM
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Classical fingerprint/descriptor GBM baseline on Polaris + MoleculeACE (CPU only).
#
# Usage:
#   sbatch scripts/evaluation/fingerprint_baseline.sbatch
#
# The 5 seeds run sequentially inside one job so the results JSON is never
# written concurrently.
#
# No SLURM? Run it directly:
#   python scripts/evaluation/fingerprint_baseline.py

set -euo pipefail

MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"

srun -u python scripts/evaluation/fingerprint_baseline.py
