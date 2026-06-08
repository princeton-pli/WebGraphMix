#!/bin/bash
# Example SLURM job for GPU centrality computation (Katz centrality shown).
# Requires cuGraph / RAPIDS environment. Adjust partition and modules for your cluster.
#
# Usage (local, from repo root):
#   export REPO_ROOT=$(pwd)
#   python pipeline/graph/centrality/compute_katz.py
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export REPO_ROOT

cd "${REPO_ROOT}/pipeline/graph/centrality"
mkdir -p logs results

if command -v module &>/dev/null; then
  module purge 2>/dev/null || true
  module load anaconda3/2025.6 2>/dev/null || true
fi

if command -v conda &>/dev/null; then
  conda activate reasoning-scaling-law-centrality-calculations 2>/dev/null || true
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

python compute_katz.py
