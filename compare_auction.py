"""
Head-to-head comparison of baseline vs belief-aware MuZero on the auction env.

Usage:
    python compare_auction.py \
        --baseline-checkpoint runs/auction_baseline/checkpoints/checkpoint_iter_1000.pt \
        --belief-checkpoint runs/auction_belief/checkpoints/checkpoint_iter_1000.pt \
        --games 500 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
import torch

from auction_env import (
    AuctionEnv,
    aggressive_bidder_action,
    conservative_bidder_action,
    random_action,
    value_bidder_action,
)
from auction_mcts import run_auction_belief_mcts, run_auction_mcts
from auction_model import (
    AuctionBeliefMuZeroNet,
    AuctionMuZeroNet,
    build_default_auction_config,
)
from muzero_mcts import MCTSConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_baseline(path: str, device: torch.device) -> AuctionMuZeroNet:
    ckpt = torch.load(path, map_location=device)
    cfg = build_default_auction_config()
    model = AuctionMuZeroNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_belief(path: str, device: torch.device) -> AuctionBeliefMuZeroNet:
    ckpt = torch.load(path, map_location=device)
    cfg = build_default_auction_config()
    model = AuctionBeliefMuZeroNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# Heuristic bots
AUCTION_BOTS = {
    "random": random_action,
    "value_bidder": value_bidder_action,
    "conservative": conservative_bidder_action,
    "aggressive": aggressive_bidder_action,
}


def _sample_slots_and_actions(env: AuctionEnv, action_size: int) -> tuple[list[int], list[float]]:
    actions = env.sample_action_candidates(num_raise_samples=max(1, action_size - 1))
    if len(actions) < action_size:
        pad_value = actions[-2] if len(actions) >= 2 else actions[0]
        while len(actions) < action_size:
            actions.insert(-1, pad_value)
    elif len(actions) > action_size:
        actions = actions[: action_size - 1] + [actions[-1]]
    return list(range(action_size)), actions


def _slot_for_action_amount(action_amount: float, action_amounts: list[float]) -> int:
    return min(range(len(action_amounts)), key=lambda i: abs(float(action_amounts[i]) - float(action_amount)))


def _make_bot_fn(name: str, seed: int) -> object:
    fn = AUCTION_BOTS[name]
    if name == "random":
        rng = np.random.RandomState(seed)
        return lambda obs, legal: fn(obs, legal, rng)
    return lambda obs, legal: fn(obs, legal)


def play_game(
    p0_fn, p1_fn, seed: int, action_size: int, min_raise_increment: float, max_moves: int = 500,
) -> dict[str, float]:
    env = AuctionEnv(seed=seed, min_raise_increment=min_raise_increment)
    obs = env.reset()
    terminated, steps = False, 0
    while not terminated and steps < max_moves:
        actor = env.current_player
        legal, action_amounts = _sample_slots_and_actions(env, action_size)
        slot = p0_fn(obs, legal, action_amounts) if actor == 0 else p1_fn(obs, legal, action_amounts)
        action = float(action_amounts[int(slot)])
        obs, _, terminated, _ = env.step(action)
        steps += 1
    pr0, pr1 = env.total_profit[0], env.total_profit[1]
    if pr0 > pr1:
        winner = 0
    elif pr1 > pr0:
        winner = 1
    else:
        winner = -1
    return {
        "winner": float(winner),
        "profit0": pr0, "profit1": pr1,
        "profit_diff_p0_minus_p1": pr0 - pr1,
        "steps": float(steps),
        "terminated": float(terminated),
    }


def _mcts_fn(model, cfg, device, is_belief=False):
    def select(obs, legal, action_amounts):
        actor = int(obs["current_player"])
        if is_belief:
            stats = run_auction_belief_mcts(model, obs, legal, actor, cfg, device, mcts_inference_autocast=True)
        else:
            stats = run_auction_mcts(model, obs, legal, cfg, device, mcts_inference_autocast=True)
        return int(stats.action)
    return select


def _reference_agent_for_pair(name_a: str, name_b: str) -> str:
    # Keep reported win rate anchored to belief when present.
    if name_a == "belief" or name_b == "belief":
        return "belief"
    return name_a


def _draw_credit_agent_for_pair(name_a: str, name_b: str) -> str:
    # For reporting/plots, credit draws to baseline when available.
    if name_a == "baseline" or name_b == "baseline":
        return "baseline"
    return name_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head auction comparison")
    parser.add_argument("--baseline-checkpoint", type=str, default="")
    parser.add_argument("--belief-checkpoint", type=str, default="")
    parser.add_argument(
        "--matchup",
        type=str,
        choices=["baseline_vs_belief", "all_pairs"],
        default="baseline_vs_belief",
        help="Comparison scope. Default is strict baseline-vs-belief only.",
    )
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/auction_compare")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-raise-increment", type=float, default=0.1)
    args = parser.parse_args()
    if args.games % 2 != 0:
        print("[warn] --games is odd; last game will be unpaired for seat-swapped seed matching.")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    graph_dir = out_dir / "graphs"
    _ensure_dir(out_dir)
    _ensure_dir(graph_dir)

    cfg = MCTSConfig(num_simulations=args.sims, temperature=1e-8, add_exploration_noise=False)

    agents: dict[str, object] = {}
    model_action_size: int | None = None
    if args.baseline_checkpoint:
        baseline_model = _load_baseline(args.baseline_checkpoint, device)
        agents["baseline"] = _mcts_fn(baseline_model, cfg, args.device, is_belief=False)
        model_action_size = baseline_model.config.action_space_size
    if args.belief_checkpoint:
        belief_model = _load_belief(args.belief_checkpoint, device)
        agents["belief"] = _mcts_fn(belief_model, cfg, args.device, is_belief=True)
        if model_action_size is None:
            model_action_size = belief_model.config.action_space_size

    if args.matchup == "baseline_vs_belief":
        if "baseline" not in agents or "belief" not in agents:
            raise ValueError("--matchup baseline_vs_belief requires both checkpoints.")
    else:
        for bn in AUCTION_BOTS:
            agents[bn] = lambda obs, legal, action_amounts, _bn=bn: _slot_for_action_amount(
                float(AUCTION_BOTS[_bn](obs, action_amounts) if _bn != "random" else random_action(obs, action_amounts)),
                action_amounts,
            )

    agent_names = list(agents.keys())
    if len(agent_names) < 2:
        raise ValueError("Need at least 2 agents to compare")

    all_results: list[dict] = []
    pairwise: dict[str, dict[str, list[float]]] = {}

    total_start = time.perf_counter()
    game_num = 0

    for i, name_a in enumerate(agent_names):
        for name_b in agent_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            profits_a, profits_b, diffs = [], [], []
            wins_a, wins_b, draws = 0, 0, 0

            for g in range(args.games):
                # Use paired seeds across seat swaps to reduce variance:
                # each consecutive game pair shares one environment seed.
                swap = g % 2 == 1
                game_seed = args.seed + (g // 2)
                p0_name = name_a if not swap else name_b
                p1_name = name_b if not swap else name_a
                p0_fn = agents[p0_name]
                p1_fn = agents[p1_name]
                result = play_game(
                    p0_fn,
                    p1_fn,
                    game_seed,
                    action_size=model_action_size or 17,
                    min_raise_increment=args.min_raise_increment,
                    max_moves=args.max_moves,
                )

                if swap:
                    pa = result["profit1"]
                    pb = result["profit0"]
                else:
                    pa = result["profit0"]
                    pb = result["profit1"]

                profits_a.append(pa)
                profits_b.append(pb)
                diffs.append(pa - pb)
                w = int(result["winner"])
                if w == -1:
                    draws += 1
                elif (w == 0 and not swap) or (w == 1 and swap):
                    wins_a += 1
                else:
                    wins_b += 1

                all_results.append({
                    "game": game_num + 1, "pair": pair_key,
                    "agent_a": name_a, "agent_b": name_b,
                    "seed": game_seed,
                    "seat0": p0_name, "seat1": p1_name,
                    "profit_a": pa, "profit_b": pb,
                    "profit_diff_a_minus_b": pa - pb,
                    "winner_name": name_a if (pa > pb) else (name_b if pb > pa else "draw"),
                })
                game_num += 1

                done = g + 1
                if done % max(1, args.log_every) == 0 or done == args.games:
                    elapsed = time.perf_counter() - total_start
                    ref_agent = _reference_agent_for_pair(name_a, name_b)
                    if ref_agent == name_a:
                        ref_wins = wins_a
                    else:
                        ref_wins = wins_b
                    ref_wr = (ref_wins + 0.5 * draws) / done
                    print(
                        f"[{pair_key}] {done}/{args.games} "
                        f"{name_a}_wins={wins_a} {name_b}_wins={wins_b} draws={draws} "
                        f"{ref_agent}_winrate={ref_wr:.3f} elapsed={elapsed:.1f}s"
                    )

            n = max(1, args.games)
            mn = mean(diffs) if diffs else 0
            sd = stdev(diffs) if len(diffs) > 1 else 0
            t_stat = mn / (sd / math.sqrt(n)) if sd > 0 else 0

            pairwise[pair_key] = {
                "agent_a": name_a, "agent_b": name_b,
                "games": n,
                "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
                "win_rate_a": (wins_a + 0.5 * draws) / n,
                "win_rate_b": (wins_b + 0.5 * draws) / n,
                "profit_diff_mean": mn,
                "profit_diff_std": sd,
                "t_statistic": t_stat,
                "mean_profit_a": mean(profits_a) if profits_a else 0,
                "mean_profit_b": mean(profits_b) if profits_b else 0,
            }
            ref_agent = _reference_agent_for_pair(name_a, name_b)
            if ref_agent == name_a:
                pairwise[pair_key]["reference_wins"] = wins_a
            else:
                pairwise[pair_key]["reference_wins"] = wins_b
            pairwise[pair_key]["reference_agent"] = ref_agent
            pairwise[pair_key]["reference_win_rate"] = (
                pairwise[pair_key]["reference_wins"] + 0.5 * draws
            ) / n

    summary = {
        "total_games": game_num,
        "pairwise": pairwise,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if all_results:
        with (out_dir / "per_game_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)

    for pk, stats in pairwise.items():
        na, nb = stats["agent_a"], stats["agent_b"]
        draw_credit_agent = _draw_credit_agent_for_pair(na, nb)
        games = max(1, int(stats["games"]))
        wins_a = int(stats["wins_a"])
        wins_b = int(stats["wins_b"])
        draws = int(stats["draws"])
        if draw_credit_agent == na:
            credited_wins_a = wins_a + draws
            credited_wins_b = wins_b
        else:
            credited_wins_a = wins_a
            credited_wins_b = wins_b + draws
        pct_a = 100.0 * credited_wins_a / games
        pct_b = 100.0 * credited_wins_b / games

        plt.figure(figsize=(6, 4))
        labels = [na, nb]
        values = [pct_a, pct_b]
        plt.bar(labels, values)
        plt.ylim(0, 100)
        plt.title(f"{pk}: Win rate (%)")
        plt.ylabel("Percent of games")
        plt.xlabel(f"Draws credited to: {draw_credit_agent}")
        plt.tight_layout()
        plt.savefig(graph_dir / f"{pk}_wins.png", dpi=140)
        plt.close()

    print("\n=== Summary ===")
    for pk, stats in pairwise.items():
        print(
            f"{pk}: {stats['reference_agent']}_wr={stats['reference_win_rate']:.3f} "
            f"profit_diff={stats['profit_diff_mean']:.2f}±{stats['profit_diff_std']:.2f} "
            f"t={stats['t_statistic']:.2f}"
        )
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
