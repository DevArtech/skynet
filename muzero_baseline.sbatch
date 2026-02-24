#!/bin/bash
#SBATCH --job-name=muzero_baseline
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/muzero_baseline/muzero_baseline_%j.out
#SBATCH --error=runs/muzero_baseline/muzero_baseline_%j.err

set -euo pipefail

# Allow override: ENV=/path/to/venv_or_conda_env sbatch ...
ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"

# Use submission directory so relative output paths are stable.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

OUTPUT_DIR=${OUTPUT_DIR:-runs/muzero_baseline}
mkdir -p "$OUTPUT_DIR"
exec >> "$OUTPUT_DIR/muzero_baseline_${SLURM_JOB_ID}.out" 2>> "$OUTPUT_DIR/muzero_baseline_${SLURM_JOB_ID}.err"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Sanity check CUDA visibility in job logs.
"$PYTHON_BIN" -u -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
EVAL_EPISODES=${EVAL_EPISODES:-40}
SELFPLAY_EPISODES_PER_ITER=${SELFPLAY_EPISODES_PER_ITER:-8}
SELFPLAY_SIMS=${SELFPLAY_SIMS:-64}
EVAL_SIMS=${EVAL_SIMS:-64}
EVAL_BOT_EPISODES_PER_BOT=${EVAL_BOT_EPISODES_PER_BOT:-20}
MAX_MOVES_PER_EPISODE=${MAX_MOVES_PER_EPISODE:-800}
DISCOUNT=${DISCOUNT:-0.997}
TRUNCATION_PENALTY=${TRUNCATION_PENALTY:-1.0}
TIME_PENALTY_PER_STEP=${TIME_PENALTY_PER_STEP:-0}
TIME_PENALTY_MAX_PER_EPISODE=${TIME_PENALTY_MAX_PER_EPISODE:--1}
OPPONENT_CHECKPOINT_FRACTION=${OPPONENT_CHECKPOINT_FRACTION:-0.2}
HEURISTIC_BOT_FRACTION=${HEURISTIC_BOT_FRACTION:-0.3}
HEURISTIC_BOT_NAMES=${HEURISTIC_BOT_NAMES:-greedy_value_replacement,information_first_flip,column_hunter,risk_aware_unknown_replacement,end_round_aggro,anti_discard}
HEURISTIC_BOT_EPSILON=${HEURISTIC_BOT_EPSILON:-0.02}
RESUME=${RESUME:-0}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    RESUME|RESUME=1|--resume)
      RESUME=1
      ;;
    RESUME_CHECKPOINT=*)
      RESUME_CHECKPOINT="${arg#RESUME_CHECKPOINT=}"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

TRAIN_ARGS=(
  --device cuda
  --eval-episodes "$EVAL_EPISODES"
  --eval-bot-episodes-per-bot "$EVAL_BOT_EPISODES_PER_BOT"
  --eval-sims "$EVAL_SIMS"
  --selfplay-episodes-per-iter "$SELFPLAY_EPISODES_PER_ITER"
  --selfplay-sims "$SELFPLAY_SIMS"
  --discount "$DISCOUNT"
  --truncation-penalty "$TRUNCATION_PENALTY"
  --time-penalty-per-step "$TIME_PENALTY_PER_STEP"
  --time-penalty-max-per-episode "$TIME_PENALTY_MAX_PER_EPISODE"
  --opponent-checkpoint-fraction "$OPPONENT_CHECKPOINT_FRACTION"
  --heuristic-bot-fraction "$HEURISTIC_BOT_FRACTION"
  --heuristic-bot-names "$HEURISTIC_BOT_NAMES"
  --heuristic-bot-epsilon "$HEURISTIC_BOT_EPSILON"
  --max-moves-per-episode "$MAX_MOVES_PER_EPISODE"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

"$PYTHON_BIN" -u train_muzero.py "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
