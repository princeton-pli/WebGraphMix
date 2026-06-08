#!/bin/bash
# Train 1B WebGraphMix headline: 50/50 betweenness top/bottom mix (Table 1, 41.4%).
# Usage: NPROC=4 ./experiments/train/betweenness_50top.sh
source "$(dirname "$0")/../artifacts/common.sh"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"

torchrun --nproc-per-node "${NPROC}" --master_port "${MASTER_PORT}" -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/betweenness_50top_corpus_32b.json \
  --logs betweenness_50top_corpus_32b_training_logs \
  --attn-name torch_attn \
  --acc 16 \
  --num-checkpoints 10 \
  --torchcompile
