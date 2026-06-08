#!/bin/bash
# Shared environment for WebGraphMix training and evaluation.
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/corpus_200b}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# Used by exp_data JSON paths (${DATA_ROOT}, ${REPO_ROOT}, ${HF_HOME})
export DATA_ROOT REPO_ROOT HF_HOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/dclm:${PYTHONPATH:-}"

cd "${REPO_ROOT}/dclm"
mkdir -p logs outputs

# Optional SLURM helpers (no-op outside Princeton clusters)
if command -v module &>/dev/null; then
  module purge 2>/dev/null || true
  module load anaconda3/2025.6 2>/dev/null || true
fi

if command -v conda &>/dev/null; then
  # conda activate/deactivate scripts reference unset vars; nounset (-u) aborts silently.
  set +u
  conda activate webgraphmix 2>/dev/null || conda activate reasoning-scaling-law-big-graph 2>/dev/null || true
  set -u
fi

get_free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
