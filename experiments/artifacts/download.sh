#!/bin/bash
# Download precomputed WebGraphMix artifacts from HuggingFace.
#
# Usage:
#   ./experiments/artifacts/download.sh centrality
#   ./experiments/artifacts/download.sh checkpoints
#   ./experiments/artifacts/download.sh all
#
# Checkpoints land under dclm/checkpoints/<name>/epoch_11.pt where <name> matches
# exp_data/models/*.json and experiments/eval/mmlu_and_lowvar.sh filters.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CHECKPOINTS_DIR="${REPO_ROOT}/dclm/checkpoints"
CENTRALITY_DIR="${REPO_ROOT}/pipeline/graph/centrality/results"
HF_CENTRALITY_REPO="${HF_CENTRALITY_REPO:-PrincetonPLI/cc-centrality-scores}"
HF_CHECKPOINTS_REPO="${HF_CHECKPOINTS_REPO:-PrincetonPLI/WebGraphMix-openlm-1B}"

HEADLINE_MODELS=(
  random_selection
  dclm_fasttext_only
  betweenness_alpha0.5
  betweenness_alpha0.5_mult_div_dclm_fasttext
)

download_centrality() {
  mkdir -p "${CENTRALITY_DIR}"
  huggingface-cli download "${HF_CENTRALITY_REPO}" \
    --local-dir "${CENTRALITY_DIR}" \
    --repo-type dataset
  echo "Centrality scores saved to pipeline/graph/centrality/results/"
}

verify_checkpoints() {
  local missing=0
  local name ckpt_path

  if [[ ! -f "${CHECKPOINTS_DIR}/open_lm_1b_eval_params.txt" ]]; then
    echo "Missing shared eval params: ${CHECKPOINTS_DIR}/open_lm_1b_eval_params.txt"
    missing=1
  fi

  for name in "${HEADLINE_MODELS[@]}"; do
    ckpt_path="${CHECKPOINTS_DIR}/${name}/epoch_11.pt"
    if [[ ! -f "${ckpt_path}" ]]; then
      echo "Missing checkpoint: ${ckpt_path}"
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    echo "Some checkpoint artifacts are missing. Re-run: $0 checkpoints"
    exit 1
  fi

  echo "All headline 1B checkpoints present under ${CHECKPOINTS_DIR}/"
  echo ""
  echo "Evaluate (from repo root, with conda env active):"
  echo "  export REPO_ROOT=${REPO_ROOT}"
  echo "  ./experiments/eval/mmlu_and_lowvar.sh   # default: betweenness_alpha0.5"
  echo ""
  echo "Or pick a model (name matches HF folder):"
  for name in "${HEADLINE_MODELS[@]}"; do
    echo "  ./experiments/eval/mmlu_and_lowvar.sh ${name}"
  done
}

download_checkpoints() {
  mkdir -p "${CHECKPOINTS_DIR}"
  huggingface-cli download "${HF_CHECKPOINTS_REPO}" \
    --local-dir "${CHECKPOINTS_DIR}" \
    --repo-type model
  verify_checkpoints
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
