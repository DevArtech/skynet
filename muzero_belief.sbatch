#!/bin/bash
#SBATCH --job-name=muzero_belief
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=runs/muzero_belief/muzero_belief_%j.out
#SBATCH --error=runs/muzero_belief/muzero_belief_%j.err

set -euo pipefail

# Allow override: ENV=/path/to/venv_or_conda_env sbatch ...
ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"

# Use submission directory so relative output paths are stable.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Sanity check CUDA visibility in job logs.
"$PYTHON_BIN" -u -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

OUTPUT_DIR=${OUTPUT_DIR:-runs/muzero_belief}
EVAL_EPISODES=${EVAL_EPISODES:-40}
SELFPLAY_EPISODES_PER_ITER=${SELFPLAY_EPISODES_PER_ITER:-8}
SELFPLAY_SIMS=${SELFPLAY_SIMS:-64}
EVAL_SIMS=${EVAL_SIMS:-64}
MAX_MOVES_PER_EPISODE=${MAX_MOVES_PER_EPISODE:-800}
WINNER_LOSS_WEIGHT=${WINNER_LOSS_WEIGHT:-0.1}
RANK_LOSS_WEIGHT=${RANK_LOSS_WEIGHT:-0.1}

"$PYTHON_BIN" -u train_belief_muzero.py \
  --device cuda \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-sims "$EVAL_SIMS" \
  --selfplay-episodes-per-iter "$SELFPLAY_EPISODES_PER_ITER" \
  --selfplay-sims "$SELFPLAY_SIMS" \
  --max-moves-per-episode "$MAX_MOVES_PER_EPISODE" \
  --winner-loss-weight "$WINNER_LOSS_WEIGHT" \
  --rank-loss-weight "$RANK_LOSS_WEIGHT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
