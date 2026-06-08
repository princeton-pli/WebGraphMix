#!/bin/bash
# Train 1B random-sampling baseline (Table 1).
# Usage: NPROC=4 ./experiments/train/random.sh
source "$(dirname "$0")/../artifacts/common.sh"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"

torchrun --nproc-per-node "${NPROC}" --master_port "${MASTER_PORT}" -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/baseline_random_corpus_32b.json \
  --logs baseline_random_corpus_32b_training_logs \
  --attn-name torch_attn \
  --acc 16 \
  --num-checkpoints 10 \
  --torchcompile
