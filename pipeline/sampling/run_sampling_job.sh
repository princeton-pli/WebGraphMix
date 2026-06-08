#!/bin/bash
# Example SLURM job for importance-based document sampling.
# Adjust paths and token budget for your experiment.
#
# Usage (local):
#   CORPUS_ROOT=corpus_200b \
#   OUTPUT_ROOT=importance_sampled_documents/centrality_only/mix/50top/betweenness_corpus_32b \
#   ANNOTATIONS_DIR=corpus_200b/annotations_for_importance_based_document_sampling_topk/betweenness \
#   NUM_TOKENS=32000000000 \
#   ./run_sampling_job.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/corpus_200b}"
CORPUS_ROOT="${CORPUS_ROOT:-${DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:?Set ANNOTATIONS_DIR}"
NUM_TOKENS="${NUM_TOKENS:-32000000000}"
NUM_PROC="${NUM_PROC:-16}"

mkdir -p "${REPO_ROOT}/logs"

python "${REPO_ROOT}/pipeline/sampling/select_training_data.py" \
  "${CORPUS_ROOT}" \
  "${OUTPUT_ROOT}" \
  --data_dirs documents \
  --annotations_dir "${ANNOTATIONS_DIR}" \
  --tokens_dir tokens \
  --num_tokens "${NUM_TOKENS}" \
  --num_proc "${NUM_PROC}"
