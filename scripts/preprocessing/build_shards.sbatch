#!/bin/bash
#SBATCH --job-name=pm6-build
#SBATCH --array=1-10%10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --hint=nomultithread
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# CPU array job for the CSR build step (the slow stage of preprocessing).
#
# Usage (PM6):
#   sbatch scripts/preprocessing/build_shards.sbatch
#
# Usage (PCBA — pass the dirs explicitly):
#   sbatch scripts/preprocessing/build_shards.sbatch \
#       --shards-dir data/pcba_monroe/train/shards \
#       --out-dir data/pcba_monroe/train/built \
#       --smiles-only
#
# No SLURM? Drop --exit-after-one and one process builds every shard in turn:
#   python -m monroe.preprocessing.build_shards --shards-dir ... --out-dir ...

set -euo pipefail

MONROE_ROOT="${MONROE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

cd "$MONROE_ROOT"
source "$MONROE_ROOT/.env"   # venv activation + thread caps (see .env.example)

# Defaults target the PM6 pipeline; any args passed to sbatch override them.
if [ "$#" -eq 0 ]; then
    set -- --shards-dir data/pm6_resharded/shards \
           --out-dir data/pm6_monroe/ \
           --use-node-features
fi

srun -u python -u -m monroe.preprocessing.build_shards "$@" --exit-after-one
