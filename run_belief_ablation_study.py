"""Inference-time ablation study for belief-aware MuZero.

Evaluates the belief model under multiple conditioning ablation modes
against heuristic bots.  All conditions share the *same* trained weights;
only the conditioning applied before the prediction heads changes.

Ablation conditions
-------------------
full             – Full ego + current_player + num_players conditioning.
no_conditioning  – Strip all conditioning (baseline representation path).
zero_ego         – Conditioning path with ego embedding zeroed out.
wrong_ego        – Feed opponent's player ID as the ego identity.
ego_only         – Only ego embedding; zero current_player & num_players.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from belief_muzero_mcts import (
    ABLATION_EGO_ONLY,
    ABLATION_FULL,
    ABLATION_NO_CONDITIONING,
    ABLATION_WRONG_EGO,
    ABLATION_ZERO_EGO,
    run_belief_mcts,
)
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from heuristic_bots import make_heuristic_bot
from muzero_mcts import MCTSConfig
from skyjo_decision_env import SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

ALL_ABLATION_MODES = [
    ABLATION_FULL,
    ABLATION_NO_CONDITIONING,
    ABLATION_ZERO_EGO,
    ABLATION_WRONG_EGO,
    ABLATION_EGO_ONLY,
]

ABLATION_LABELS = {
    ABLATION_FULL: "Full model",
    ABLATION_NO_CONDITIONING: "No conditioning",
    ABLATION_ZERO_EGO: "Zero ego (context only)",
    ABLATION_WRONG_EGO: "Wrong ego identity",
    ABLATION_EGO_ONLY: "Ego only (no context)",
}


def _load_belief(path: str, device: torch.device) -> BeliefAwareMuZeroNet:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    action_space_size = state_dict["prediction.policy_head.net.6.weight"].shape[0]
    model = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space_size=action_space_size)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_ablation_vs_bots(
    model: BeliefAwareMuZeroNet,
    ablation_mode: str,
    ablation_label: str,
    bot_names: list[str],
    episodes_per_bot: int,
    sims: int,
    max_moves: int,
    device: str,
    seed_base: int,
    env_mode: str,
    bot_epsilon: float,
    log_every: int = 25,
) -> dict[str, Any]:
    """Play *episodes_per_bot* games per bot for a single ablation mode.

    Returns per-bot and aggregate metrics.
    """
    cfg = MCTSConfig(
        num_simulations=sims,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
    )

    per_bot: dict[str, dict[str, float]] = {}
    all_score_diffs: list[float] = []
    all_wins = 0.0
    all_games = 0
    mode_t0 = time.perf_counter()

    for bot_idx, bot_name in enumerate(bot_names):
        score_diffs: list[float] = []
        wins = 0.0
        completed = 0
        bot_t0 = time.perf_counter()

        for ep in range(episodes_per_bot):
            seed = seed_base + (bot_idx * 10_000) + ep
            bot = make_heuristic_bot(bot_name, seed=seed, epsilon=bot_epsilon)
            env: Any
            if env_mode == "decision":
                env = SkyjoDecisionEnv(num_players=2, seed=seed, setup_mode="auto")
            else:
                env = SkyjoEnv(num_players=2, seed=seed, setup_mode="auto")
            obs = env.reset()
            terminated = False
            steps = 0

            while not terminated and (max_moves <= 0 or steps < max_moves):
                actor = int(env.current_player)
                legal = env.legal_actions()
                if actor == 0:
                    stats = run_belief_mcts(
                        model=model,
                        observation=obs,
                        legal_action_ids=legal,
                        ego_player_id=actor,
                        config=cfg,
                        ablation_mode=ablation_mode,
                        device=device,
                        mcts_inference_autocast=True,
                    )
                    action = int(stats.action)
                else:
                    action = int(bot.select_action(obs, legal))
                obs, _, terminated, _ = env.step(action)
                steps += 1

            if not terminated:
                continue
            score0 = float(env.scores[0])
            score1 = float(env.scores[1])
            score_diffs.append(score0 - score1)
            min_score = min(env.scores)
            winners = [i for i, score in enumerate(env.scores) if score == min_score]
            wins += 1.0 / float(len(winners)) if 0 in winners else 0.0
            completed += 1

            done = ep + 1
            if done % log_every == 0 or done == episodes_per_bot:
                wr = wins / completed if completed > 0 else 0.0
                avg_diff = mean(score_diffs) if score_diffs else 0.0
                elapsed_bot = time.perf_counter() - bot_t0
                pace = elapsed_bot / done
                eta_bot = pace * (episodes_per_bot - done)
                remaining_bots = len(bot_names) - bot_idx - 1
                eta_mode = eta_bot + pace * episodes_per_bot * remaining_bots
                print(
                    f"  [{ablation_label}] vs {bot_name}: "
                    f"{done}/{episodes_per_bot}  "
                    f"win_rate={wr:.3f}  "
                    f"mean_diff={avg_diff:.1f}  "
                    f"({elapsed_bot:.0f}s elapsed, ~{eta_mode:.0f}s left for mode)",
                    flush=True,
                )

        if completed > 0:
            per_bot[bot_name] = {
                "games": completed,
                "mean_score_diff": float(mean(score_diffs)),
                "std_score_diff": float(stdev(score_diffs)) if len(score_diffs) > 1 else 0.0,
                "win_rate": float(wins / completed),
            }
        else:
            per_bot[bot_name] = {
                "games": 0,
                "mean_score_diff": float("nan"),
                "std_score_diff": float("nan"),
                "win_rate": float("nan"),
            }
        all_score_diffs.extend(score_diffs)
        all_wins += wins
        all_games += completed

    aggregate = {
        "games": all_games,
        "mean_score_diff": float(mean(all_score_diffs)) if all_score_diffs else float("nan"),
        "std_score_diff": float(stdev(all_score_diffs)) if len(all_score_diffs) > 1 else 0.0,
        "win_rate": float(all_wins / all_games) if all_games > 0 else float("nan"),
    }
    return {"aggregate": aggregate, "per_bot": per_bot}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-time ablation study for belief-aware MuZero.")
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument(
        "--ablation-modes",
        nargs="+",
        default=ALL_ABLATION_MODES,
        choices=ALL_ABLATION_MODES,
        help="Which ablation conditions to evaluate.",
    )
    parser.add_argument(
        "--bot-names",
        nargs="+",
        default=["greedy_value_replacement", "column_hunter", "risk_aware_unknown_replacement"],
    )
    parser.add_argument("--episodes-per-bot", type=int, default=100)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--max-moves-per-game", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/belief_ablation_study")
    parser.add_argument("--env-mode", type=str, choices=["decision", "macro"], default="decision")
    parser.add_argument("--bot-epsilon", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_belief(args.belief_checkpoint, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []

    print(f"{'=' * 72}")
    print(f"Belief-MuZero inference-time ablation study")
    print(f"  checkpoint : {args.belief_checkpoint}")
    print(f"  modes      : {args.ablation_modes}")
    print(f"  bots       : {args.bot_names}")
    print(f"  episodes   : {args.episodes_per_bot} per bot")
    print(f"  sims       : {args.sims}")
    print(f"  env_mode   : {args.env_mode}")
    print(f"{'=' * 72}")

    for mode in args.ablation_modes:
        t0 = time.perf_counter()
        label = ABLATION_LABELS.get(mode, mode)
        print(f"\n[ablation] Running condition: {label} ({mode})")

        result = evaluate_ablation_vs_bots(
            model=model,
            ablation_mode=mode,
            ablation_label=label,
            bot_names=args.bot_names,
            episodes_per_bot=args.episodes_per_bot,
            sims=args.sims,
            max_moves=args.max_moves_per_game,
            device=args.device,
            seed_base=args.seed,
            env_mode=args.env_mode,
            bot_epsilon=args.bot_epsilon,
            log_every=args.log_every,
        )
        elapsed = time.perf_counter() - t0

        agg = result["aggregate"]
        print(
            f"  => win_rate={agg['win_rate']:.3f}  "
            f"mean_score_diff={agg['mean_score_diff']:.2f} "
            f"(±{agg['std_score_diff']:.2f})  "
            f"games={agg['games']}  "
            f"time={elapsed:.1f}s"
        )
        for bot_name, bres in result["per_bot"].items():
            print(
                f"     vs {bot_name}: win_rate={bres['win_rate']:.3f}  "
                f"mean_score_diff={bres['mean_score_diff']:.2f}"
            )

        all_results[mode] = result

        table_rows.append(
            {
                "condition": mode,
                "label": label,
                "win_rate": agg["win_rate"],
                "mean_score_diff": agg["mean_score_diff"],
                "std_score_diff": agg["std_score_diff"],
                "games": agg["games"],
            }
        )

    # Summary table
    print(f"\n{'=' * 72}")
    print(f"{'Condition':<28} {'Win Rate':>10} {'Score Diff':>12} {'± Std':>10} {'Games':>7}")
    print(f"{'-' * 72}")
    for row in table_rows:
        print(
            f"{row['label']:<28} {row['win_rate']:>10.3f} "
            f"{row['mean_score_diff']:>12.2f} {row['std_score_diff']:>10.2f} "
            f"{row['games']:>7d}"
        )
    print(f"{'=' * 72}")

    # Save results
    with (out_dir / "ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"config": vars(args), "results": all_results, "summary_table": table_rows},
            f,
            indent=2,
            default=str,
        )

    with (out_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "label", "win_rate", "mean_score_diff", "std_score_diff", "games"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    # Per-bot CSV for detailed analysis
    with (out_dir / "ablation_per_bot.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "bot", "win_rate", "mean_score_diff", "std_score_diff", "games"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, result in all_results.items():
            for bot_name, bres in result["per_bot"].items():
                writer.writerow(
                    {
                        "condition": mode,
                        "bot": bot_name,
                        "win_rate": bres["win_rate"],
                        "mean_score_diff": bres["mean_score_diff"],
                        "std_score_diff": bres["std_score_diff"],
                        "games": bres["games"],
                    }
                )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
