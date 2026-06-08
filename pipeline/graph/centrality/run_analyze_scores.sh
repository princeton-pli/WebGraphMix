#!/bin/bash
#SBATCH --job-name=analyze_scores
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=logs/analyze_scores_slurm_%j.out
#SBATCH --error=logs/analyze_scores_slurm_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export REPO_ROOT

cd "${REPO_ROOT}/pipeline/graph/centrality"
mkdir -p logs

if command -v module &>/dev/null; then
  module load anaconda3/2025.6 2>/dev/null || true
fi

if command -v conda &>/dev/null; then
  conda activate reasoning-scaling-law-big-graph 2>/dev/null || true
fi

python3 analyze_scores.py
