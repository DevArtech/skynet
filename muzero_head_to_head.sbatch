#!/bin/bash
#SBATCH --job-name=muzero_h2h
#SBATCH --partition=dgxh100
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
GAMES=${GAMES:-1000}
SIMS=${SIMS:-200}
MAX_MOVES_PER_GAME=${MAX_MOVES_PER_GAME:-2000}
SEED=${SEED:-123}
ENV_MODE=${ENV_MODE:-decision}

mkdir -p "$OUTPUT_DIR"

latest_aligned_iter=$(
  BASELINE_DIR="$BASELINE_DIR" BELIEF_DIR="$BELIEF_DIR" python - <<'PY'
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

b = iters(os.environ["BASELINE_DIR"])
be = iters(os.environ["BELIEF_DIR"])
common = b & be
if not common:
    sys.exit(1)
print(max(common))
PY
) || {
  echo "ERROR: No aligned checkpoints found between:"
  echo "  baseline: $BASELINE_DIR"
  echo "  belief:   $BELIEF_DIR"
  exit 2
}

BASELINE_CKPT="$BASELINE_DIR/checkpoint_iter_${latest_aligned_iter}.pt"
BELIEF_CKPT="$BELIEF_DIR/checkpoint_iter_${latest_aligned_iter}.pt"
RUN_OUT="$OUTPUT_DIR/aligned_iter_${latest_aligned_iter}"

mkdir -p "$RUN_OUT"

echo "Using aligned iteration: $latest_aligned_iter"
echo "Baseline checkpoint: $BASELINE_CKPT"
echo "Belief checkpoint:   $BELIEF_CKPT"
echo "Output directory:    $RUN_OUT"

"$PYTHON_BIN" -u compare_muzero_architectures.py \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --belief-checkpoint "$BELIEF_CKPT" \
  --games "$GAMES" \
  --sims "$SIMS" \
  --max-moves-per-game "$MAX_MOVES_PER_GAME" \
  --seed "$SEED" \
  --device cuda \
  --env-mode "$ENV_MODE" \
  --output-dir "$RUN_OUT"
