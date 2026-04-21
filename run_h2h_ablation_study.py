"""Pairwise head-to-head ablation study for belief-aware MuZero.

For each ablation condition the *full* model (all conditioning) plays
against a copy of itself running under the ablated conditioning path.
Both sides share the same weights — only the inference-time conditioning
differs.  Seats alternate every game.

Produces a summary table suitable for a paper ablation section.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from belief_muzero_mcts import (
    ABLATION_EGO_ONLY,
    ABLATION_FULL,
    ABLATION_NO_CONDITIONING,
    ABLATION_WRONG_EGO,
    ABLATION_ZERO_EGO,
    VALID_ABLATION_MODES,
    run_belief_mcts,
)
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from muzero_mcts import MCTSConfig
from skyjo_decision_env import SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

OPPONENT_ABLATION_MODES = [
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


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _elo_diff(win_rate: float) -> float:
    if win_rate <= 0.0 or win_rate >= 1.0:
        clamped = max(0.001, min(0.999, win_rate))
    else:
        clamped = win_rate
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def play_h2h_game(
    model: BeliefAwareMuZeroNet,
    mode_a: str,
    mode_b: str,
    seat0_is_a: bool,
    sims: int,
    max_moves: int,
    seed: int,
    neutralize_time_penalty: bool,
    time_penalty_per_step: float,
    device: str,
    env_mode: str,
) -> dict[str, Any]:
    env: Any
    if env_mode == "decision":
        env = SkyjoDecisionEnv(num_players=2, seed=seed, setup_mode="auto")
    else:
        env = SkyjoEnv(num_players=2, seed=seed, setup_mode="auto")
    obs = env.reset()

    reward_offset = float(time_penalty_per_step) if neutralize_time_penalty else 0.0
    cfg = MCTSConfig(
        num_simulations=sims,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
        reward_offset=reward_offset,
    )

    terminated = False
    steps = 0
    while not terminated and steps < max_moves:
        player = int(env.current_player)
        is_side_a = (player == 0) == seat0_is_a
        mode = mode_a if is_side_a else mode_b

        stats = run_belief_mcts(
            model=model,
            observation=obs,
            legal_action_ids=env.legal_actions(),
            ego_player_id=player,
            config=cfg,
            ablation_mode=mode,
            device=device,
            mcts_inference_autocast=True,
        )
        obs, _, terminated, _ = env.step(int(stats.action))
        steps += 1

    score0, score1 = env.scores[0], env.scores[1]
    a_seat = 0 if seat0_is_a else 1
    b_seat = 1 if seat0_is_a else 0
    score_a = env.scores[a_seat]
    score_b = env.scores[b_seat]

    if score_a < score_b:
        winner = "a"
    elif score_b < score_a:
        winner = "b"
    else:
        winner = "draw"

    return {
        "winner": winner,
        "score_a": float(score_a),
        "score_b": float(score_b),
        "steps": steps,
        "terminated": terminated,
    }


def run_matchup(
    model: BeliefAwareMuZeroNet,
    opponent_mode: str,
    games: int,
    sims: int,
    max_moves: int,
    seed_base: int,
    neutralize_time_penalty: bool,
    time_penalty_per_step: float,
    device: str,
    env_mode: str,
    log_every: int,
) -> dict[str, Any]:
    full_wins = 0
    opp_wins = 0
    draws = 0
    score_diffs: list[float] = []
    opp_label = ABLATION_LABELS.get(opponent_mode, opponent_mode)
    t0 = time.perf_counter()

    for g in range(games):
        seat0_is_full = (g % 2 == 0)
        result = play_h2h_game(
            model=model,
            mode_a=ABLATION_FULL,
            mode_b=opponent_mode,
            seat0_is_a=seat0_is_full,
            sims=sims,
            max_moves=max_moves,
            seed=seed_base + g,
            neutralize_time_penalty=neutralize_time_penalty,
            time_penalty_per_step=time_penalty_per_step,
            device=device,
            env_mode=env_mode,
        )
        if result["winner"] == "a":
            full_wins += 1
        elif result["winner"] == "b":
            opp_wins += 1
        else:
            draws += 1
        score_diffs.append(result["score_a"] - result["score_b"])

        done = g + 1
        if done % log_every == 0 or done == games:
            elapsed = time.perf_counter() - t0
            pace = elapsed / done
            eta = pace * (games - done)
            wr = full_wins / done
            print(
                f"  [Full vs {opp_label}] {done}/{games}  "
                f"full_wins={full_wins} opp_wins={opp_wins} draws={draws}  "
                f"full_wr={wr:.3f}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                flush=True,
            )

    total = max(1, games)
    wr = full_wins / total
    ci_lo, ci_hi = _wilson_ci(full_wins, total)
    elo = _elo_diff(wr) if 0 < wr < 1 else (_elo_diff(0.999) if wr >= 1 else _elo_diff(0.001))

    return {
        "opponent_mode": opponent_mode,
        "opponent_label": opp_label,
        "games": games,
        "full_wins": full_wins,
        "opponent_wins": opp_wins,
        "draws": draws,
        "full_win_rate": wr,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "elo_diff": elo,
        "mean_score_diff": float(mean(score_diffs)) if score_diffs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise head-to-head ablation study for belief MuZero.")
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument(
        "--opponent-modes",
        nargs="+",
        default=OPPONENT_ABLATION_MODES,
        choices=list(VALID_ABLATION_MODES - {ABLATION_FULL}),
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--max-moves-per-game", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/h2h_ablation_study")
    parser.add_argument("--env-mode", type=str, choices=["decision", "macro"], default="decision")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--neutralize-time-penalty", action="store_true")
    parser.add_argument("--time-penalty-per-step", type=float, default=0.0002)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_belief(args.belief_checkpoint, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 76}")
    print("Belief-MuZero pairwise head-to-head ablation study")
    print(f"  checkpoint : {args.belief_checkpoint}")
    print(f"  opponents  : {args.opponent_modes}")
    print(f"  games      : {args.games} per matchup")
    print(f"  sims       : {args.sims}")
    print(f"  env_mode   : {args.env_mode}")
    print(f"  neutralize : {args.neutralize_time_penalty}")
    print(f"{'=' * 76}")

    results: list[dict[str, Any]] = []

    for opp_mode in args.opponent_modes:
        label = ABLATION_LABELS.get(opp_mode, opp_mode)
        print(f"\n[matchup] Full model vs {label} ({opp_mode})")
        r = run_matchup(
            model=model,
            opponent_mode=opp_mode,
            games=args.games,
            sims=args.sims,
            max_moves=args.max_moves_per_game,
            seed_base=args.seed,
            neutralize_time_penalty=args.neutralize_time_penalty,
            time_penalty_per_step=args.time_penalty_per_step,
            device=args.device,
            env_mode=args.env_mode,
            log_every=args.log_every,
        )
        elapsed_total = time.perf_counter()
        print(
            f"  => Full WR={r['full_win_rate']:.3f} "
            f"[{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]  "
            f"Elo={r['elo_diff']:+.0f}  "
            f"W/L/D={r['full_wins']}/{r['opponent_wins']}/{r['draws']}"
        )
        results.append(r)

    # Summary table
    print(f"\n{'=' * 76}")
    print(f"{'Opponent Condition':<28} {'Full W':>7} {'Opp W':>7} {'Draw':>5} {'Full WR (95% CI)':>22} {'Δ Elo':>7}")
    print(f"{'-' * 76}")
    for r in results:
        ci_str = f"{r['full_win_rate']:.1%} [{r['ci_lo']:.1%}–{r['ci_hi']:.1%}]"
        print(
            f"{r['opponent_label']:<28} {r['full_wins']:>7d} {r['opponent_wins']:>7d} "
            f"{r['draws']:>5d} {ci_str:>22s} {r['elo_diff']:>+7.0f}"
        )
    print(f"{'=' * 76}")

    # Save JSON
    with (out_dir / "h2h_ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, default=str)

    # Save CSV
    with (out_dir / "h2h_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "opponent_mode", "opponent_label", "games",
            "full_wins", "opponent_wins", "draws",
            "full_win_rate", "ci_lo", "ci_hi", "elo_diff", "mean_score_diff",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
