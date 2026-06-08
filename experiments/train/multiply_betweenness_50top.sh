#!/bin/bash
# Train 1B WebGraphMix+ headline: multiply/divide betweenness 50% top mix (Table 1, 43.8%).
# Usage: NPROC=4 ./experiments/train/multiply_betweenness_50top.sh
source "$(dirname "$0")/../artifacts/common.sh"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"

torchrun --nproc-per-node "${NPROC}" --master_port "${MASTER_PORT}" -m training.train -- \
  --scale 1b_1x_fast \
  --data-config exp_data/datasets/tokenized/centrality_dclmfilter_multiply/regular_bottomk/betweenness_50top_corpus_32b.json \
  --logs centralitydclmfiltermultiply_betweenness_50top_corpus_32b_training_logs \
  --attn-name torch_attn \
  --acc 16 \
  --num-checkpoints 10 \
  --torchcompile
