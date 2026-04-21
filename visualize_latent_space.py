#!/usr/bin/env python3
"""
Visualize latent space structure via UMAP, colored by hidden-state features.

Reads the .npz file produced by collect_latent_states.py and generates:
  1. Side-by-side UMAP: baseline vs belief latent spaces, colored by each feature
  2. Three-way comparison: baseline h0 vs belief h0 (raw) vs belief h0 (ego-conditioned)
  3. Linear probe accuracy: how well a linear model can predict each hidden feature
     from frozen latent vectors (quantitative complement to UMAP)

Usage:
  python visualize_latent_space.py \
    --input runs/latent_analysis/latent_data.npz \
    --output-dir runs/latent_analysis/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
except ImportError:
    UMAP = None


FEATURE_LABELS = {
    "feat_hidden_sum": "Sum of Face-Down Cards",
    "feat_hidden_count": "# Face-Down Cards",
    "feat_hidden_mean": "Mean Face-Down Card Value",
    "feat_num_high_hidden": "# High Hidden Cards (≥9)",
    "feat_num_negative_hidden": "# Negative Hidden Cards",
    "feat_visible_sum": "Sum of Visible Cards",
    "feat_total_true_score": "True Total Score",
    "feat_opp_hidden_sum": "Opponent Hidden Sum",
    "feat_opp_true_score": "Opponent True Score",
    "feat_true_score_advantage": "True Score Advantage",
    "feat_deck_size": "Deck Size",
    "feat_game_progress": "Game Progress",
    "feat_game_winner": "Game Winner",
}

PRIMARY_FEATURES = [
    "feat_hidden_sum",
    "feat_num_high_hidden",
    "feat_total_true_score",
    "feat_true_score_advantage",
    "feat_game_progress",
    "feat_hidden_count",
]


def subsample(arrays: list[np.ndarray], max_n: int = 15000, seed: int = 0) -> list[np.ndarray]:
    n = arrays[0].shape[0]
    if n <= max_n:
        return arrays
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, max_n, replace=False)
    return [a[idx] for a in arrays]


def fit_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.3, seed: int = 42) -> np.ndarray:
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed, metric="cosine")
    return reducer.fit_transform(X)


def plot_side_by_side(
    bl_umap: np.ndarray,
    bf_umap: np.ndarray,
    color_values: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"wspace": 0.05})

    vmin, vmax = np.percentile(color_values, [2, 98])

    for ax, emb, label in [(axes[0], bl_umap, "Baseline MuZero"), (axes[1], bf_umap, "Belief-Aware MuZero")]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_values, cmap=cmap, s=3, alpha=0.5, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(label, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.75])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=10)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_three_way(
    bl_umap: np.ndarray,
    bf_umap: np.ndarray,
    bf_cond_umap: np.ndarray,
    color_values: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={"wspace": 0.05})

    vmin, vmax = np.percentile(color_values, [2, 98])

    labels = ["Baseline MuZero", "Belief h₀ (raw)", "Belief h₀ (ego-cond)"]
    embeddings = [bl_umap, bf_umap, bf_cond_umap]

    for ax, emb, label in zip(axes, embeddings, labels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_values, cmap=cmap, s=3, alpha=0.5, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(label, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=15, y=0.98)
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=10)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_egocond_by_player(
    bf_cond_umap: np.ndarray,
    player_ids: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#4C72B0", "#C44E52"]
    for p in sorted(np.unique(player_ids)):
        mask = player_ids == p
        ax.scatter(
            bf_cond_umap[mask, 0], bf_cond_umap[mask, 1],
            c=colors[int(p) % len(colors)], s=4, alpha=0.4,
            label=f"Player {int(p)}", rasterized=True,
        )
    ax.set_title("Ego-Conditioned Belief Latent Space\nColored by Player ID", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=12, markerscale=4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def run_linear_probes(
    baseline_h0: np.ndarray,
    belief_h0: np.ndarray,
    belief_h0_cond: np.ndarray,
    features: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Train linear probes on frozen latent vectors to predict hidden features."""
    print("\n--- Linear Probe R² (5-fold CV) ---")
    results = {}

    for feat_name in sorted(features.keys()):
        y = features[feat_name]
        if np.std(y) < 1e-8:
            continue

        label = FEATURE_LABELS.get(feat_name, feat_name)
        row = {"feature": label}

        for space_name, X in [("Baseline", baseline_h0), ("Belief (raw)", belief_h0), ("Belief (ego-cond)", belief_h0_cond)]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=1.0)
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
            mean_r2 = scores.mean()
            row[space_name] = mean_r2

        results[feat_name] = row
        print(f"  {label:35s}  BL={row['Baseline']:.3f}  BF={row['Belief (raw)']:.3f}  BF-cond={row['Belief (ego-cond)']:.3f}")

    fig, ax = plt.subplots(figsize=(12, 7))
    feat_names = [results[k]["feature"] for k in sorted(results.keys()) if k in PRIMARY_FEATURES]
    bl_scores = [results[k]["Baseline"] for k in sorted(results.keys()) if k in PRIMARY_FEATURES]
    bf_scores = [results[k]["Belief (raw)"] for k in sorted(results.keys()) if k in PRIMARY_FEATURES]
    bf_cond_scores = [results[k]["Belief (ego-cond)"] for k in sorted(results.keys()) if k in PRIMARY_FEATURES]

    x = np.arange(len(feat_names))
    width = 0.25
    ax.bar(x - width, bl_scores, width, label="Baseline", color="#C44E52")
    ax.bar(x, bf_scores, width, label="Belief (raw)", color="#55A868")
    ax.bar(x + width, bf_cond_scores, width, label="Belief (ego-cond)", color="#4C72B0")
    ax.set_ylabel("Linear Probe R²", fontsize=12)
    ax.set_title("How Much Hidden-State Info Is Linearly Decodable from Latent Vectors?", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / "linear_probe_r2.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize latent space with UMAP.")
    parser.add_argument("--input", type=str, default="runs/latent_analysis/latent_data.npz")
    parser.add_argument("--output-dir", type=str, default="runs/latent_analysis/figures")
    parser.add_argument("--max-samples", type=int, default=15000, help="Subsample for UMAP speed")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP, only run linear probes")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    baseline_h0 = data["baseline_h0"]
    belief_h0 = data["belief_h0"]
    belief_h0_cond = data["belief_h0_cond"]
    print(f"  {baseline_h0.shape[0]} samples, latent dim = {baseline_h0.shape[1]}")

    features = {k: data[k] for k in data.files if k.startswith("feat_")}
    player_ids = data["player_id"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_linear_probes(baseline_h0, belief_h0, belief_h0_cond, features, out_dir)

    if args.skip_umap:
        print("Skipping UMAP (--skip-umap set)")
        return

    if UMAP is None:
        print("WARNING: umap-learn not installed. Skipping UMAP visualizations.")
        print("  Install with: pip install umap-learn")
        return

    bl_sub, bf_sub, bf_cond_sub, pid_sub, *feat_subs = subsample(
        [baseline_h0, belief_h0, belief_h0_cond, player_ids] + [features[k] for k in sorted(features.keys())],
        max_n=args.max_samples,
    )
    feat_keys_sorted = sorted(features.keys())
    feat_sub = {k: v for k, v in zip(feat_keys_sorted, feat_subs)}

    print(f"\nFitting UMAP on baseline latent space ({bl_sub.shape[0]} samples)...")
    bl_umap = fit_umap(bl_sub)
    print(f"Fitting UMAP on belief latent space (raw)...")
    bf_umap = fit_umap(bf_sub)
    print(f"Fitting UMAP on belief latent space (ego-conditioned)...")
    bf_cond_umap = fit_umap(bf_cond_sub)

    print("\nGenerating side-by-side plots...")
    for feat_key in PRIMARY_FEATURES:
        label = FEATURE_LABELS.get(feat_key, feat_key)
        safe_name = feat_key.replace("feat_", "")

        plot_side_by_side(
            bl_umap, bf_umap, feat_sub[feat_key],
            title=f"Latent Space Colored by {label}",
            save_path=out_dir / f"umap_sidebyside_{safe_name}.png",
            cmap="viridis" if "winner" not in feat_key else "RdYlGn",
        )

    print("\nGenerating three-way plots (best features)...")
    for feat_key in ["feat_hidden_sum", "feat_true_score_advantage", "feat_num_high_hidden"]:
        label = FEATURE_LABELS.get(feat_key, feat_key)
        safe_name = feat_key.replace("feat_", "")
        plot_three_way(
            bl_umap, bf_umap, bf_cond_umap, feat_sub[feat_key],
            title=f"Three-Way Latent Comparison: {label}",
            save_path=out_dir / f"umap_threeway_{safe_name}.png",
        )

    print("\nGenerating ego-conditioned player ID plot...")
    plot_egocond_by_player(
        bf_cond_umap, pid_sub,
        save_path=out_dir / "umap_egocond_by_player.png",
    )

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
