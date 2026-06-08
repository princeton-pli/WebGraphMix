#!/bin/bash
# Download precomputed WebGraphMix artifacts from HuggingFace.
#
# Usage:
#   ./experiments/artifacts/download.sh centrality
#   ./experiments/artifacts/download.sh checkpoints
#   ./experiments/artifacts/download.sh all
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CENTRALITY_DIR="${REPO_ROOT}/pipeline/graph/centrality/results"
HF_CENTRALITY_REPO="${HF_CENTRALITY_REPO:-PrincetonPLI/cc-centrality-scores}"
HF_CHECKPOINTS_REPO="${HF_CHECKPOINTS_REPO:-PrincetonPLI/WebGraphMix-openlm-1B}"

download_centrality() {
  mkdir -p "${CENTRALITY_DIR}"
  huggingface-cli download "${HF_CENTRALITY_REPO}" \
    --local-dir "${CENTRALITY_DIR}" \
    --repo-type dataset
  echo "Centrality scores saved to pipeline/graph/centrality/results/"
}

download_checkpoints() {
  mkdir -p "${REPO_ROOT}/dclm/checkpoints"
  huggingface-cli download "${HF_CHECKPOINTS_REPO}" \
    --local-dir "${REPO_ROOT}/dclm/checkpoints" \
    --repo-type model
  echo "Checkpoints saved to dclm/checkpoints/"
}

case "${1:-all}" in
  centrality) download_centrality ;;
  checkpoints) download_checkpoints ;;
  all) download_centrality; download_checkpoints ;;
  *)
    echo "Usage: $0 {centrality|checkpoints|all}"
    exit 1
    ;;
esac
