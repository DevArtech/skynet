#!/bin/bash
#SBATCH --job-name=latent_viz
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=runs/latent_analysis/latent_analysis_%j.out
#SBATCH --error=runs/latent_analysis/latent_analysis_%j.err

set -euo pipefail

ENV=${ENV:-/data/csc4611/conda-csc4611}
PYTHON_BIN="$ENV/bin/python"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"$PYTHON_BIN" -u -c "import torch; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available())"

BASELINE_CKPT=${BASELINE_CKPT:-runs/muzero_baseline_superhuman_fresh_20260306_001223/checkpoints/checkpoint_iter_500.pt}
BELIEF_CKPT=${BELIEF_CKPT:-runs/muzero_belief_superhuman_fresh_20260306_001207/checkpoints/checkpoint_iter_500.pt}
GAMES=${GAMES:-150}
SIMS=${SIMS:-50}
OUTPUT=${OUTPUT:-runs/latent_analysis/latent_data.npz}
FIGURE_DIR=${FIGURE_DIR:-runs/latent_analysis/figures}

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$FIGURE_DIR"

echo ""
echo "=== Step 1: Collect latent states ==="
echo "  Baseline: $BASELINE_CKPT"
echo "  Belief:   $BELIEF_CKPT"
echo "  Games:    $GAMES"
echo "  Sims:     $SIMS"
echo "  Output:   $OUTPUT"

"$PYTHON_BIN" -u collect_latent_states.py \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --belief-checkpoint "$BELIEF_CKPT" \
  --games "$GAMES" \
  --sims "$SIMS" \
  --device cuda \
  --output "$OUTPUT"

echo ""
echo "=== Step 2: Install umap-learn + scikit-learn (if needed) ==="
"$PYTHON_BIN" -m pip install --quiet umap-learn scikit-learn 2>/dev/null || true

echo ""
echo "=== Step 3: Visualize ==="
"$PYTHON_BIN" -u visualize_latent_space.py \
  --input "$OUTPUT" \
  --output-dir "$FIGURE_DIR"

echo ""
echo "=== Done ==="
echo "Figures saved to $FIGURE_DIR"
ls -la "$FIGURE_DIR"
