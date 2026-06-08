#!/bin/bash
# Upload precomputed WebGraphMix artifacts to HuggingFace (maintainers only).
# Requires: huggingface-cli login, network access, and local artifact files.
#
# Usage:
#   ./experiments/artifacts/upload.sh centrality
#   ./experiments/artifacts/upload.sh checkpoints
#   ./experiments/artifacts/upload.sh all
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CENTRALITY_DIR="${REPO_ROOT}/pipeline/graph/centrality/results"
HF_CENTRALITY_REPO="${HF_CENTRALITY_REPO:-PrincetonPLI/cc-centrality-scores}"
HF_CHECKPOINTS_REPO="${HF_CHECKPOINTS_REPO:-PrincetonPLI/WebGraphMix-openlm-1B}"

upload_centrality() {
  local src="${CENTRALITY_DIR}"
  if [[ ! -f "${src}/host_graph_scores_betweenness_k1400000.json" ]]; then
    echo "Missing centrality scores in ${src}. Run centrality scripts or copy from cluster."
    exit 1
  fi
  huggingface-cli upload "${HF_CENTRALITY_REPO}" "${src}" . --repo-type dataset
}

upload_checkpoints() {
  local src="${REPO_ROOT}/dclm/checkpoints"
  if [[ ! -d "${src}" ]] || [[ -z "$(ls -A "${src}" 2>/dev/null)" ]]; then
    echo "Missing checkpoints in ${src}."
    echo "Copy epoch_11.pt for headline models, or run: python dclm/convert_openlm_to_hf_1b.py"
    exit 1
  fi
  huggingface-cli upload "${HF_CHECKPOINTS_REPO}" "${src}" . --repo-type model
}

case "${1:-all}" in
  centrality) upload_centrality ;;
  checkpoints) upload_checkpoints ;;
  all) upload_centrality; upload_checkpoints ;;
  *)
    echo "Usage: $0 {centrality|checkpoints|all}"
    exit 1
    ;;
esac

echo "Upload complete."
