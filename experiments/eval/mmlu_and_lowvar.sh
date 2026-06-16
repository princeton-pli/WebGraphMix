#!/bin/bash
# Evaluate a trained 1B checkpoint on DCLM CORE v2 (mmlu_and_lowvar, 23 tasks).
#
# Usage:
#   MODEL_UUID=969815a0-652d-4227-a51e-020fbdda0357 ./experiments/eval/mmlu_and_lowvar.sh
#   ./experiments/eval/mmlu_and_lowvar.sh betweenness_alpha0.5
#   ./experiments/eval/mmlu_and_lowvar.sh random_selection
#
# Model names match HuggingFace checkpoint folders under dclm/checkpoints/.
# Use >=2 GPUs (default) so FSDP can shard the 1B model; NUM_GPUS=1 risks OOM (SIGKILL).
source "$(dirname "$0")/../artifacts/common.sh"

MODEL_FILTER="${1:-}"
if [[ -n "${MODEL_UUID:-}" ]]; then
  FILTER_ARG="-f uuid=${MODEL_UUID}"
elif [[ -n "${MODEL_FILTER}" ]]; then
  FILTER_ARG="-f name=${MODEL_FILTER}"
else
  FILTER_ARG="-f name=betweenness_alpha0.5"
fi

python tools/eval_expdb.py \
  --num_gpus "${NUM_GPUS:-2}" \
  --no_skip \
  --output_dir exp_data/evals/ \
  --eval_yaml eval/mmlu_and_lowvar.yaml \
  ${FILTER_ARG} \
  --skip_perplexity
