# Skyjo Deterministic RL Environment

This repository contains a CLI-first Skyjo environment for reinforcement learning.

- Full 2-8 player support
- Deterministic seedable simulation
- Full hidden-information state + public observations
- Transformer-friendly token outputs
- Discrete macro-action API for training

## Quick start

```bash
python main.py inspect --players 2 --seed 7 --steps 3
python main.py simulate --players 4 --games 10 --seed 100 --setup-mode auto
python main.py play --players 3 --seed 42 --setup-mode manual
```

For full details, read:

- `SKYJO_ENVIRONMENT.md`

## MuZero Training

This repo includes two trainable architectures:

- `train_muzero.py`: classical MuZero baseline
- `train_belief_muzero.py`: belief-aware MuZero with winner/rank auxiliary heads

By default both training scripts now use the **decision-granularity** environment (`skyjo_decision_env.py`) with:

- Step A: choose source (`deck` / `discard`)
- Step B: keep-or-discard (if deck source)
- Step C: choose position

This uses a 16-action masked policy space.

### 1) Install dependencies

```bash
pip install -e .
```

### 2) Train classical MuZero baseline

```bash
python train_muzero.py --output-dir runs/muzero_baseline --device cuda
```

### 3) Train belief-aware MuZero

```bash
python train_belief_muzero.py --output-dir runs/muzero_belief --device cuda
```

Belief-aware defaults now prioritize core MuZero learning in low-data regimes:

- `--winner-loss-weight 0.1`
- `--rank-loss-weight 0.1`

### 4) Compare baseline vs belief-aware

```bash
python compare_muzero_architectures.py \
  --baseline-checkpoint runs/muzero_baseline/checkpoints/checkpoint_iter_50.pt \
  --belief-checkpoint runs/muzero_belief/checkpoints/checkpoint_iter_50.pt \
  --games 40 \
  --device cuda
```

To run the legacy macro-action setup instead, add `--env-mode macro`.

## Rosie Compute (DGX-H100) Training

This repo now includes SLURM scripts for Rosie so baseline and belief-aware jobs can run independently on DGX-H100 with **1 GPU each**:

- `run_muzero_baseline_rosie.sbatch`
- `run_muzero_belief_rosie.sbatch`

From `skynet/`:

```bash
bash run_muzero_baseline_rosie.sbatch
bash run_muzero_belief_rosie.sbatch
```

Optional overrides:

```bash
ENV=/path/to/conda-or-venv bash run_muzero_baseline_rosie.sbatch
OUTPUT_DIR=runs/custom_belief bash run_muzero_belief_rosie.sbatch
PARTITION=dgxch100 bash run_muzero_baseline_rosie.sbatch
PARTITION=teaching bash run_muzero_belief_rosie.sbatch
```

You can also pass through training flags, for example:

```bash
bash run_muzero_baseline_rosie.sbatch --iterations 100 --selfplay-sims 200
```

You may still use direct `sbatch` submission and override partition there too:

```bash
sbatch --partition=dgx run_muzero_baseline_rosie.sbatch
```

### Outputs

- Baseline outputs: `runs/muzero_baseline/`
- Belief-aware outputs: `runs/muzero_belief/`
- Head-to-head outputs: `runs/muzero_arch_compare/`

Each training run writes:

- `metrics_history.json` and `metrics_history.csv`
- `graphs/` (training, testing, diagnostics plots)
- `checkpoints/` (periodic model checkpoints)
- Belief-aware runs also write `calibration/` reliability artifacts (`*.json`, `*.csv`, `*.png`)
