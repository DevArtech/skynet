#!/usr/bin/env python3
"""
Run linear probes on latent data from multiple training iterations and produce
a small-multiples line-plot figure (one subplot per feature).

Usage:
  python plot_probe_by_iteration.py \
    --inputs runs/latent_analysis/latent_data_iter250.npz \
             runs/latent_analysis/latent_data.npz \
             runs/latent_analysis/latent_data_iter1000.npz \
    --iterations 250 500 1000 \
    --output paper/fig_linear_probe_by_iter.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ---------- Features to plot ----------
FIGURE_FEATURES = [
    "feat_hidden_count",
    "feat_deck_size",
    "feat_visible_sum",
    "feat_game_winner",
    "feat_opp_hidden_sum",
    "feat_hidden_sum",
    "feat_true_score_advantage",
]

FIGURE_LABELS = [
    "# Face-Down Cards",
    "Deck Size",
    "Visible Card Sum",
    "Game Winner",
    "Opponent Hidden Sum",
    "Own Hidden Card Sum",
    "True Score Advantage",
]

# Which features are observable vs hidden (for annotating)
OBSERVABLE_FEATURES = {
    "feat_hidden_count",
    "feat_deck_size",
    "feat_visible_sum",
    "feat_game_winner",
}

# ---------- Models ----------
MODEL_KEYS = [
    ("Baseline",          "baseline_h0"),
    ("Belief (raw)",      "belief_h0"),
    ("Belief (ego-cond)", "belief_h0_cond"),
]

MODEL_COLORS = {
    "Baseline":          "#4C72B0",
    "Belief (raw)":      "#55A868",
    "Belief (ego-cond)": "#C44E52",
}

MODEL_MARKERS = {
    "Baseline":          "o",
    "Belief (raw)":      "s",
    "Belief (ego-cond)": "D",
}


# --------------------------------------------------------------------- #
def run_probes_for_dataset(npz_path: str) -> dict[str, dict[str, float]]:
    """Return {feat_name: {model_name: mean_r2}} for one .npz file."""
    data = np.load(npz_path)
    results: dict[str, dict[str, float]] = {}

    for feat_name in FIGURE_FEATURES:
        y = data[feat_name]
        if np.std(y) < 1e-8:
            continue
        row: dict[str, float] = {}
        for model_name, array_key in MODEL_KEYS:
            X = data[array_key]
            X_scaled = StandardScaler().fit_transform(X)
            scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=5, scoring="r2")
            row[model_name] = scores.mean()
        results[feat_name] = row

    return results


# --------------------------------------------------------------------- #
def plot_small_multiples(
    all_results: dict[int, dict[str, dict[str, float]]],
    iterations: list[int],
    output_path: str,
) -> None:
    """Create a small-multiples grid: one subplot per feature, lines over iterations."""

    n_feats = len(FIGURE_FEATURES)
    n_cols = 4
    n_rows = (n_feats + n_cols - 1) // n_cols  # ceil

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 3.2 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()

    sorted_iters = sorted(iterations)

    for idx, (feat, label) in enumerate(zip(FIGURE_FEATURES, FIGURE_LABELS)):
        ax = flat_axes[idx]
        is_obs = feat in OBSERVABLE_FEATURES

        for model_name, _ in MODEL_KEYS:
            ys = [
                all_results[it].get(feat, {}).get(model_name, float("nan"))
                for it in sorted_iters
            ]
            ax.plot(
                sorted_iters,
                ys,
                marker=MODEL_MARKERS[model_name],
                color=MODEL_COLORS[model_name],
                linewidth=2,
                markersize=7,
                label=model_name,
                zorder=3,
            )

        # R² = 0 reference
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.35, zorder=1)

        # Styling
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel("R²", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", alpha=0.15)
        ax.set_xticks(sorted_iters)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Tag observable/hidden
        tag = "Observable" if is_obs else "Hidden"
        tag_color = "#2b8cbe" if is_obs else "#e34a33"
        ax.text(
            0.98, 0.96, tag,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5, fontstyle="italic",
            color=tag_color, alpha=0.7,
        )

    # Hide empty subplots
    for idx in range(n_feats, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    # X-label only on bottom row
    for ax in flat_axes:
        if ax.get_visible():
            ax.set_xlabel("")
    for c in range(n_cols):
        bottom_ax = axes[n_rows - 1, c] if axes[n_rows - 1, c].get_visible() else None
        if bottom_ax is None:
            # find the visible ax in this column
            for r in range(n_rows - 1, -1, -1):
                if axes[r, c].get_visible():
                    bottom_ax = axes[r, c]
                    break
        if bottom_ax is not None:
            bottom_ax.set_xlabel("Training Iteration", fontsize=10)

    # Shared legend in the empty subplot space (or at figure level)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.08),
        fontsize=10,
        framealpha=0.9,
        ncol=1,
        title="Representation",
        title_fontsize=10,
    )


    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved small-multiples figure to {output_path}")


# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help=".npz files in iteration order")
    parser.add_argument("--iterations", nargs="+", type=int, required=True, help="Iteration numbers matching --inputs")
    parser.add_argument("--output", type=str, default="paper/fig_linear_probe_by_iter.png")
    args = parser.parse_args()

    assert len(args.inputs) == len(args.iterations), "Must provide same number of inputs and iterations"

    all_results: dict[int, dict[str, dict[str, float]]] = {}
    for npz_path, it in zip(args.inputs, args.iterations):
        print(f"\n=== Running probes for iteration {it} ({npz_path}) ===")
        all_results[it] = run_probes_for_dataset(npz_path)

        for feat in FIGURE_FEATURES:
            r = all_results[it].get(feat, {})
            label = FIGURE_LABELS[FIGURE_FEATURES.index(feat)]
            bl = r.get("Baseline", float("nan"))
            bf = r.get("Belief (raw)", float("nan"))
            bc = r.get("Belief (ego-cond)", float("nan"))
            print(f"  {label:25s}  BL={bl:.3f}  BF={bf:.3f}  BF-cond={bc:.3f}")

    plot_small_multiples(all_results, args.iterations, args.output)


if __name__ == "__main__":
    main()
