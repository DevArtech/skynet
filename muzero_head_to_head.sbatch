#!/bin/bash
#SBATCH --job-name=muzero_h2h
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=runs/muzero_arch_compare/muzero_h2h_%j.out
#SBATCH --error=runs/muzero_arch_compare/muzero_h2h_%j.err

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

BASELINE_DIR=${BASELINE_DIR:-runs/muzero_baseline/checkpoints}
BELIEF_DIR=${BELIEF_DIR:-runs/muzero_belief/checkpoints}
OUTPUT_DIR=${OUTPUT_DIR:-runs/muzero_arch_compare}
MATCHUP=${MATCHUP:-baseline_vs_belief}
TARGET_ITER=${TARGET_ITER:-}
GAMES=${GAMES:-1000}
SIMS=${SIMS:-200}
MAX_MOVES_PER_GAME=${MAX_MOVES_PER_GAME:-2000}
SEED=${SEED:-123}
ENV_MODE=${ENV_MODE:-decision}
LOG_EVERY=${LOG_EVERY:-25}
ABLATE_BELIEF_HEAD=${ABLATE_BELIEF_HEAD:-0}
NEUTRALIZE_TIME_PENALTY=${NEUTRALIZE_TIME_PENALTY:-0}
TIME_PENALTY_PER_STEP=${TIME_PENALTY_PER_STEP:-0.0002}

mkdir -p "$OUTPUT_DIR"

selected_iter=$(
  BASELINE_DIR="$BASELINE_DIR" BELIEF_DIR="$BELIEF_DIR" MATCHUP="$MATCHUP" TARGET_ITER="$TARGET_ITER" python - <<'PY'
import os
import pathlib
import re
import sys

pat = re.compile(r"^checkpoint_iter_(\d+)\.pt$")

def iters(folder: str) -> set[int]:
    p = pathlib.Path(folder)
    if not p.exists():
        return set()
    out: set[int] = set()
    for child in p.glob("checkpoint_iter_*.pt"):
        m = pat.match(child.name)
        if m:
            out.add(int(m.group(1)))
    return out

target = os.environ.get("TARGET_ITER", "").strip()
matchup = os.environ.get("MATCHUP", "baseline_vs_belief").strip()

belief_iters = iters(os.environ["BELIEF_DIR"])
if matchup == "baseline_vs_belief":
    baseline_iters = iters(os.environ["BASELINE_DIR"])
    common = baseline_iters & belief_iters
    if target:
        print(int(target))
        sys.exit(0)
    if not common:
        sys.exit(1)
    print(max(common))
    sys.exit(0)

if matchup == "belief_vs_ablated":
    if target:
        print(int(target))
        sys.exit(0)
    if not belief_iters:
        sys.exit(1)
    print(max(belief_iters))
    sys.exit(0)

sys.exit(2)
PY
) || {
  echo "ERROR: Unable to resolve checkpoint iteration."
  echo "  matchup:  $MATCHUP"
  echo "  baseline: $BASELINE_DIR"
  echo "  belief:   $BELIEF_DIR"
  exit 2
}

BASELINE_CKPT="$BASELINE_DIR/checkpoint_iter_${selected_iter}.pt"
BELIEF_CKPT="$BELIEF_DIR/checkpoint_iter_${selected_iter}.pt"
RUN_OUT="$OUTPUT_DIR/aligned_iter_${selected_iter}"

if [[ "$MATCHUP" == "baseline_vs_belief" ]]; then
  if [[ ! -f "$BASELINE_CKPT" || ! -f "$BELIEF_CKPT" ]]; then
    echo "ERROR: Requested iteration is not available for both models."
    echo "  requested_iter: $selected_iter"
    echo "  baseline_ckpt_exists: $([[ -f "$BASELINE_CKPT" ]] && echo yes || echo no)"
    echo "  belief_ckpt_exists:   $([[ -f "$BELIEF_CKPT" ]] && echo yes || echo no)"
    exit 3
  fi
elif [[ "$MATCHUP" == "belief_vs_ablated" ]]; then
  if [[ ! -f "$BELIEF_CKPT" ]]; then
    echo "ERROR: Requested belief checkpoint is missing."
    echo "  requested_iter: $selected_iter"
    echo "  belief_ckpt_exists:   $([[ -f "$BELIEF_CKPT" ]] && echo yes || echo no)"
    exit 3
  fi
else
  echo "ERROR: Unsupported MATCHUP='$MATCHUP'"
  echo "  valid options: baseline_vs_belief, belief_vs_ablated"
  exit 4
fi

mkdir -p "$RUN_OUT"

if [[ -n "$TARGET_ITER" ]]; then
  echo "Using TARGET_ITER override: $selected_iter"
else
  echo "Using auto-selected iteration: $selected_iter"
fi
echo "Matchup:             $MATCHUP"
if [[ "$MATCHUP" == "baseline_vs_belief" ]]; then
  echo "Policy A:            belief"
  echo "Policy B:            baseline"
  echo "Baseline checkpoint: $BASELINE_CKPT"
  echo "Belief checkpoint:   $BELIEF_CKPT"
else
  echo "Policy A:            belief"
  echo "Policy B:            belief_ablated"
  echo "Belief checkpoint:   $BELIEF_CKPT"
  echo "Baseline checkpoint: (unused for this matchup)"
fi
echo "Output directory:    $RUN_OUT"
echo "Belief head ablated: $ABLATE_BELIEF_HEAD"
echo "Neutralize time penalty: $NEUTRALIZE_TIME_PENALTY"
echo "Time penalty per step:   $TIME_PENALTY_PER_STEP"

ABLATE_ARGS=()
if [[ "$MATCHUP" == "baseline_vs_belief" && "$ABLATE_BELIEF_HEAD" == "1" ]]; then
  ABLATE_ARGS+=(--ablate-belief-head)
fi

TIME_PENALTY_ARGS=()
if [[ "$NEUTRALIZE_TIME_PENALTY" == "1" ]]; then
  TIME_PENALTY_ARGS+=(--neutralize-time-penalty --time-penalty-per-step "$TIME_PENALTY_PER_STEP")
fi

COMPARE_ARGS=(
  --matchup "$MATCHUP"
  --belief-checkpoint "$BELIEF_CKPT" \
  --games "$GAMES" \
  --sims "$SIMS" \
  --max-moves-per-game "$MAX_MOVES_PER_GAME" \
  --seed "$SEED" \
  --device cuda \
  --env-mode "$ENV_MODE" \
  --log-every "$LOG_EVERY" \
  --output-dir "$RUN_OUT"
)

if [[ "$MATCHUP" == "baseline_vs_belief" ]]; then
  COMPARE_ARGS+=(--baseline-checkpoint "$BASELINE_CKPT")
fi

"$PYTHON_BIN" -u compare_muzero_architectures.py "${COMPARE_ARGS[@]}" "${ABLATE_ARGS[@]}" "${TIME_PENALTY_ARGS[@]}"
