#!/bin/bash
# Tokenize and shuffle importance-sampled documents with tokshuf-rs.
#
# Usage:
#   INPUT_DIR=/path/to/sampled/documents \
#   OUTPUT_DIR=/path/to/tokenized/output \
#   LOCAL_CELL_DIR=/path/to/tokshuf_tmp \
#   ./dclm/rust_processing/tokshuf-rs/run_rust_tokenizer.sh
#
# Note on whitespace: tokshuf-rs expects documents with a trailing newline per record.
# If tokenization fails on large corpora, ensure each jsonl line ends with '\n' and
# re-run after `cargo build --release` in this directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/corpus_200b}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the sampled documents/ folder}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the tokenized output directory}"
LOCAL_CELL_DIR="${LOCAL_CELL_DIR:-${REPO_ROOT}/tokshuf_tmp}"
TOKENIZER="${TOKENIZER:-EleutherAI/gpt-neox-20b}"

export HF_HOME
mkdir -p "${LOCAL_CELL_DIR}" "${OUTPUT_DIR}"

cd "$(dirname "$0")"
cargo run --release -- \
  --input "${INPUT_DIR}" \
  --local-cell-dir "${LOCAL_CELL_DIR}" \
  --output "${OUTPUT_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --seqlen 2049 \
  --wds-chunk-size 8192 \
  --num-local-cells 512 \
  --threads "${THREADS:-32}"
