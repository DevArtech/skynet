"""
Train belief-aware MuZero on the competitive auction environment.

Usage:
    python train_auction_belief_muzero.py --iterations 1000 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
import torch

from auction_env import (
    AuctionEnv,
    aggressive_bidder_action,
    conservative_bidder_action,
    random_action,
    value_bidder_action,
)
from auction_mcts import run_auction_belief_mcts
from auction_model import AuctionBeliefMuZeroNet, build_default_auction_config
from auction_train import (
    BeliefReplayBuffer,
    BeliefTrainConfig,
    create_belief_optimizer,
    generate_belief_episode,
    train_belief_step,
)
from muzero_mcts import MCTSConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Bot registry (shared with baseline)
# ---------------------------------------------------------------------------
AUCTION_BOT_REGISTRY: dict[str, Any] = {
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


def _make_bot_selector(name: str, seed: int = 0) -> Any:
    fn = AUCTION_BOT_REGISTRY[name]
    if name == "random":
        rng = __import__("numpy").random.RandomState(seed)
        return lambda obs, legal, actor: fn(obs, legal, rng)
    return lambda obs, legal, actor: fn(obs, legal)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _entropy(probs: list[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def _safe_rate(n: float, d: float) -> float:
    return float(n / d) if d > 0 else float("nan")


def _checkpoint_iter(p: Path) -> int:
    try:
        return int(p.stem.replace("checkpoint_iter_", ""))
    except ValueError:
        return -1


def _find_latest_checkpoint(d: Path) -> Path | None:
    cs = sorted(d.glob("checkpoint_iter_*.pt"), key=_checkpoint_iter)
    return cs[-1] if cs else None


def _piecewise_schedule(it: int, init: int, mid: int, fin: int, mid_it: int, fin_it: int) -> int:
    if it >= fin_it:
        return fin
    return mid if it >= mid_it else init


def _dirichlet_params(it: int, warmup: int, a0: float, f0: float, a1: float, f1: float) -> tuple[float, float]:
    return (a0, f0) if it <= warmup else (a1, f1)


def _linear_ramp(it: int, start: int, end: int, sv: float, ev: float) -> float:
    if it <= start:
        return sv
    if it >= end:
        return ev
    return sv + (it - start) / max(1, end - start) * (ev - sv)


def _rebuild_pool(ckpt_dir: Path, size: int, up_to: int, device: str) -> list[dict]:
    if size <= 0:
        return []
    cks = sorted([p for p in ckpt_dir.glob("checkpoint_iter_*.pt") if 0 < _checkpoint_iter(p) <= up_to], key=_checkpoint_iter)
    pool = []
    for p in cks[-size:]:
        sd = torch.load(p, map_location=device).get("model_state_dict")
        if sd:
            pool.append(copy.deepcopy(sd))
    return pool


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _build_history() -> dict[str, list]:
    return {
        "iteration": [], "loss_total": [], "loss_policy": [],
        "loss_value": [], "loss_reward": [], "loss_winner": [],
        "loss_rank": [], "grad_norm": [], "replay_steps": [],
        "selfplay_mean_steps": [], "selfplay_terminated_fraction": [],
        "selfplay_policy_entropy": [], "selfplay_root_value": [],
        "selfplay_num_simulations": [], "winner_loss_weight": [],
        "rank_loss_weight": [],
        "eval_profit_p0": [], "eval_profit_p1": [],
        "eval_profit_diff_p0_minus_p1": [], "eval_win_rate_p0": [],
        "eval_mean_episode_length": [], "eval_truncation_rate": [],
        "eval_bots_win_rate_p0": [],
        "eval_bots_win_rate_p1": [],
        "eval_bots_win_rate_balanced": [],
    }


def _load_history(path: Path, tmpl: dict) -> dict:
    if not path.exists():
        return {k: [] for k in tmpl}
    with path.open() as f:
        data = json.load(f)
    return {k: [float(v) for v in data.get(k, [])] for k in tmpl}


def _trim_history(h: dict, n: int) -> None:
    for k in h:
        h[k] = h[k][:n]
    rc = len(h["iteration"])
    for k in h:
        while len(h[k]) < rc:
            h[k].append(float("nan"))


def _save_metrics(h: dict, out: Path) -> None:
    with (out / "metrics_history.json").open("w") as f:
        json.dump(h, f, indent=2)
    keys = list(h.keys())
    rows = len(h["iteration"])
    with (out / "metrics_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i in range(rows):
            w.writerow({k: h[k][i] for k in keys})


def _plot(x: list, y: list, title: str, ylabel: str, path: Path) -> None:
    pts = [(xi, yi) for xi, yi in zip(x, y) if math.isfinite(yi)]
    if not pts:
        return
    plt.figure(figsize=(7, 4))
    plt.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2, marker="o", markersize=3)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _save_graphs(h: dict, g: Path) -> None:
    _ensure_dir(g)
    x = [int(v) for v in h["iteration"]]
    specs = [
        ("loss_total", "Total Loss", "Loss"),
        ("loss_policy", "Policy Loss", "Loss"),
        ("loss_value", "Value Loss", "Loss"),
        ("loss_reward", "Reward Loss", "Loss"),
        ("loss_winner", "Winner Head Loss", "Loss"),
        ("loss_rank", "Rank Head Loss", "Loss"),
        ("grad_norm", "Gradient Norm", "Norm"),
        ("eval_profit_p0", "Self-Play Eval Profit P0", "Profit"),
        ("eval_profit_p1", "Self-Play Eval Profit P1", "Profit"),
        ("eval_profit_diff_p0_minus_p1", "Self-Play Eval Profit Diff", "Diff"),
        ("eval_win_rate_p0", "Self-Play Eval Win Rate P0", "Rate"),
        ("eval_truncation_rate", "Eval Truncation Rate", "Rate"),
        ("eval_bots_win_rate_p0", "Eval vs Bots Win Rate (Model as P0)", "Rate"),
        ("eval_bots_win_rate_p1", "Eval vs Bots Win Rate (Model as P1)", "Rate"),
        ("eval_bots_win_rate_balanced", "Eval vs Bots Win Rate (Seat-Balanced)", "Rate"),
    ]
    for key, title, yl in specs:
        if key in h and len(h[key]) == len(x):
            _plot(x, h[key], title, yl, g / f"{key}.png")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: AuctionBeliefMuZeroNet, episodes: int, sims: int, max_moves: int,
    device: str, seed_base: int, env_variant: str, min_raise_increment: float,
) -> dict[str, float]:
    cfg = MCTSConfig(num_simulations=sims, temperature=1e-8, add_exploration_noise=False)
    profits_p0, profits_p1, diffs = [], [], []
    wins, lengths, truncs = 0.0, [], 0
    for e in range(episodes):
        env = AuctionEnv(seed=seed_base + e, env_variant=env_variant, min_raise_increment=min_raise_increment)
        obs = env.reset()
        terminated, steps = False, 0
        while not terminated and steps < max_moves:
            actor = env.current_player
            legal, action_amounts = _sample_slots_and_actions(env, model.config.action_space_size)
            stats = run_auction_belief_mcts(model, obs, legal, actor, cfg, device, mcts_inference_autocast=True)
            obs, _, terminated, _ = env.step(float(action_amounts[int(stats.action)]))
            steps += 1
        if not terminated:
            truncs += 1
        lengths.append(steps)
        p0, p1 = env.total_profit[0], env.total_profit[1]
        profits_p0.append(p0)
        profits_p1.append(p1)
        diffs.append(p0 - p1)
        if p0 > p1:
            wins += 1
        elif p0 == p1:
            wins += 0.5
    n = max(1, episodes)
    return {
        "eval_profit_p0": mean(profits_p0) if profits_p0 else float("nan"),
        "eval_profit_p1": mean(profits_p1) if profits_p1 else float("nan"),
        "eval_profit_diff_p0_minus_p1": mean(diffs) if diffs else float("nan"),
        "eval_win_rate_p0": wins / n,
        "eval_mean_episode_length": mean(lengths) if lengths else 0,
        "eval_truncation_rate": truncs / n,
    }


def evaluate_vs_bots(
    model: AuctionBeliefMuZeroNet, bot_names: list[str], episodes_per_bot: int,
    sims: int, max_moves: int, device: str, seed_base: int, env_variant: str, min_raise_increment: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not bot_names:
        return {
            "eval_bots_win_rate_p0": float("nan"),
            "eval_bots_win_rate_p1": float("nan"),
            "eval_bots_win_rate_balanced": float("nan"),
        }, {}
    cfg = MCTSConfig(num_simulations=sims, temperature=1e-8, add_exploration_noise=False)
    per_bot: dict[str, dict[str, float]] = {}
    all_wins_p0, all_games_p0 = 0.0, 0
    all_wins_p1, all_games_p1 = 0.0, 0
    for bi, bn in enumerate(bot_names):
        wins_p0, completed_p0 = 0.0, 0
        wins_p1, completed_p1 = 0.0, 0
        for ep in range(episodes_per_bot):
            for model_player in (0, 1):
                seed = seed_base + bi * 100_000 + ep
                bot = _make_bot_selector(bn, seed + model_player)
                env = AuctionEnv(seed=seed, env_variant=env_variant, min_raise_increment=min_raise_increment)
                obs = env.reset()
                terminated, steps = False, 0
                while not terminated and steps < max_moves:
                    actor = env.current_player
                    legal, action_amounts = _sample_slots_and_actions(env, model.config.action_space_size)
                    if actor == model_player:
                        stats = run_auction_belief_mcts(model, obs, legal, actor, cfg, device, mcts_inference_autocast=True)
                        action = float(action_amounts[int(stats.action)])
                    else:
                        bot_action = float(bot(obs, action_amounts, actor))
                        action = float(action_amounts[_slot_for_action_amount(bot_action, action_amounts)])
                    obs, _, terminated, _ = env.step(action)
                    steps += 1
                if not terminated:
                    continue
                my_profit = env.total_profit[model_player]
                opp_profit = env.total_profit[1 - model_player]
                win = 1.0 if my_profit > opp_profit else (0.5 if my_profit == opp_profit else 0.0)
                if model_player == 0:
                    wins_p0 += win
                    completed_p0 += 1
                else:
                    wins_p1 += win
                    completed_p1 += 1
        games_total = completed_p0 + completed_p1
        wins_total = wins_p0 + wins_p1
        per_bot[bn] = {
            "games_p0": completed_p0,
            "games_p1": completed_p1,
            "games_balanced": games_total,
            "win_rate_p0": wins_p0 / max(1, completed_p0),
            "win_rate_p1": wins_p1 / max(1, completed_p1),
            "win_rate_balanced": wins_total / max(1, games_total),
        }
        all_wins_p0 += wins_p0
        all_games_p0 += completed_p0
        all_wins_p1 += wins_p1
        all_games_p1 += completed_p1
    return {
        "eval_bots_win_rate_p0": all_wins_p0 / max(1, all_games_p0),
        "eval_bots_win_rate_p1": all_wins_p1 / max(1, all_games_p1),
        "eval_bots_win_rate_balanced": (all_wins_p0 + all_wins_p1) / max(1, all_games_p0 + all_games_p1),
    }, per_bot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train belief-aware MuZero on auction")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--selfplay-episodes-per-iter", type=int, default=32)
    parser.add_argument("--train-steps-per-iter", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-bot-episodes-per-bot", type=int, default=30)
    parser.add_argument("--selfplay-sims", type=int, default=100)
    parser.add_argument("--selfplay-sims-mid", type=int, default=200)
    parser.add_argument("--selfplay-sims-final", type=int, default=300)
    parser.add_argument("--selfplay-sims-mid-iter", type=int, default=300)
    parser.add_argument("--selfplay-sims-final-iter", type=int, default=700)
    parser.add_argument("--eval-sims", type=int, default=100)
    parser.add_argument("--dirichlet-alpha-initial", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac-initial", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha-late", type=float, default=0.15)
    parser.add_argument("--dirichlet-frac-late", type=float, default=0.10)
    parser.add_argument("--dirichlet-switch-iter", type=int, default=200)
    parser.add_argument("--opponent-pool-size", type=int, default=10)
    parser.add_argument("--opponent-checkpoint-fraction", type=float, default=0.3)
    parser.add_argument("--heuristic-bot-fraction", type=float, default=0.5)
    parser.add_argument("--opponent-checkpoint-fraction-final", type=float, default=0.7)
    parser.add_argument("--heuristic-bot-fraction-final", type=float, default=0.2)
    parser.add_argument("--opponent-mix-ramp-start-iter", type=int, default=250)
    parser.add_argument("--opponent-mix-ramp-end-iter", type=int, default=700)
    parser.add_argument("--heuristic-bot-names", type=str, default="value_bidder,conservative,aggressive")
    parser.add_argument("--opponent-snapshot-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-moves-per-episode", type=int, default=500)
    parser.add_argument("--eval-max-moves", type=int, default=500)
    parser.add_argument("--replay-capacity-episodes", type=int, default=4000)
    parser.add_argument("--winner-loss-weight-initial", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight-initial", type=float, default=0.05)
    parser.add_argument("--winner-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--aux-loss-ramp-start-iter", type=int, default=100)
    parser.add_argument("--aux-loss-ramp-end-iter", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--env-variant", type=str, default="v2", choices=["v2"])
    parser.add_argument("--min-raise-increment", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="runs/auction_belief")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bot_names = [n.strip() for n in args.heuristic_bot_names.split(",") if n.strip()]
    bad = [n for n in bot_names if n not in AUCTION_BOT_REGISTRY]
    if bad:
        raise ValueError(f"Unknown bots: {bad}. Available: {sorted(AUCTION_BOT_REGISTRY)}")

    out_dir = Path(args.output_dir)
    graph_dir = out_dir / "graphs"
    ckpt_dir = out_dir / "checkpoints"
    _ensure_dir(out_dir)
    _ensure_dir(graph_dir)
    _ensure_dir(ckpt_dir)

    model_cfg = build_default_auction_config()
    model = AuctionBeliefMuZeroNet(model_cfg).to(args.device)
    train_cfg = BeliefTrainConfig(
        unroll_steps=args.unroll_steps, td_steps=args.td_steps, discount=args.discount,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, max_moves_per_episode=args.max_moves_per_episode,
        replay_capacity_episodes=args.replay_capacity_episodes,
        winner_loss_weight=args.winner_loss_weight_initial,
        rank_loss_weight=args.rank_loss_weight_initial,
        device=args.device,
    )
    optimizer = create_belief_optimizer(model, train_cfg)
    grad_scaler = torch.amp.GradScaler() if args.device == "cuda" else None
    replay = BeliefReplayBuffer(train_cfg.replay_capacity_episodes)
    opp_model = AuctionBeliefMuZeroNet(model_cfg).to(args.device)
    opp_model.eval()
    pool: list[dict] = []

    history = _build_history()
    start_iter = 1
    if args.resume or args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint) if args.resume_checkpoint else _find_latest_checkpoint(ckpt_dir)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError("No checkpoint found for resume.")
        payload = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        done = int(payload.get("iteration", 0))
        start_iter = done + 1
        history = _load_history(out_dir / "metrics_history.json", _build_history())
        _trim_history(history, done)
        pool = _rebuild_pool(ckpt_dir, args.opponent_pool_size, done, args.device)
        print(f"[resume belief] checkpoint={ckpt_path} iter={done}")

    if args.device == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        opp_model = torch.compile(opp_model, mode="reduce-overhead")

    if start_iter > args.iterations:
        print("[resume belief] nothing to do")
        return

    for iteration in range(start_iter, args.iterations + 1):
        sp_sims = _piecewise_schedule(
            iteration, args.selfplay_sims, args.selfplay_sims_mid,
            args.selfplay_sims_final, args.selfplay_sims_mid_iter, args.selfplay_sims_final_iter,
        )
        da, df = _dirichlet_params(
            iteration, args.dirichlet_switch_iter,
            args.dirichlet_alpha_initial, args.dirichlet_frac_initial,
            args.dirichlet_alpha_late, args.dirichlet_frac_late,
        )
        sp_cfg = MCTSConfig(
            num_simulations=sp_sims, temperature=1.0,
            add_exploration_noise=True, root_dirichlet_alpha=da, root_exploration_fraction=df,
        )

        wlw = _linear_ramp(iteration, args.aux_loss_ramp_start_iter, args.aux_loss_ramp_end_iter,
                           args.winner_loss_weight_initial, args.winner_loss_weight)
        rlw = _linear_ramp(iteration, args.aux_loss_ramp_start_iter, args.aux_loss_ramp_end_iter,
                           args.rank_loss_weight_initial, args.rank_loss_weight)
        iter_cfg = replace(train_cfg, winner_loss_weight=wlw, rank_loss_weight=rlw)

        ep_lens, term_flags, entropies, root_vals = [], [], [], []
        sp_start = time.perf_counter()
        bot_frac = max(0.0, min(1.0, args.heuristic_bot_fraction))
        ckpt_frac = max(0.0, min(1.0, args.opponent_checkpoint_fraction))
        for ep_idx in range(args.selfplay_episodes_per_iter):
            env_seed = random.randint(0, 10_000_000)
            env_factory = lambda s=env_seed: AuctionEnv(
                seed=s,
                env_variant=args.env_variant,
                min_raise_increment=args.min_raise_increment,
            )

            opp_selector = None
            selected_opp = None
            bot_frac = _linear_ramp(
                iteration,
                args.opponent_mix_ramp_start_iter,
                args.opponent_mix_ramp_end_iter,
                args.heuristic_bot_fraction,
                args.heuristic_bot_fraction_final,
            )
            ckpt_frac = _linear_ramp(
                iteration,
                args.opponent_mix_ramp_start_iter,
                args.opponent_mix_ramp_end_iter,
                args.opponent_checkpoint_fraction,
                args.opponent_checkpoint_fraction_final,
            )
            bot_frac = max(0.0, min(1.0, bot_frac))
            ckpt_frac = max(0.0, min(1.0, ckpt_frac))
            use_bot = bool(bot_names) and random.random() < bot_frac
            if use_bot:
                bn = random.choice(bot_names)
                opp_selector = _make_bot_selector(bn, args.seed + iteration * 1_000_000 + ep_idx)
            elif pool and random.random() < ckpt_frac:
                opp_inner = opp_model._orig_mod if hasattr(opp_model, "_orig_mod") else opp_model
                opp_inner.load_state_dict(random.choice(pool))
                selected_opp = opp_model

            learner = random.randint(0, 1)
            episode = generate_belief_episode(
                model, sp_cfg, selected_opp, learner, env_factory,
                iter_cfg.max_moves_per_episode, iter_cfg.device, opp_selector,
            )
            replay.add_episode(episode)
            elapsed = time.perf_counter() - sp_start
            print(
                f"[selfplay belief] iter={iteration:04d} ep={ep_idx + 1}/{args.selfplay_episodes_per_iter} "
                f"steps={len(episode.steps)} term={int(episode.terminated)} elapsed={elapsed:.1f}s"
            )
            ep_lens.append(len(episode.steps))
            term_flags.append(1.0 if episode.terminated else 0.0)
            for s in episode.steps:
                entropies.append(_entropy(s.policy_target))
                root_vals.append(s.root_value)

        accum = {k: 0.0 for k in ["loss_total", "loss_policy", "loss_value", "loss_reward", "loss_winner", "loss_rank", "grad_norm"]}
        eff = 0
        for _ in range(args.train_steps_per_iter):
            if replay.total_steps() < iter_cfg.batch_size:
                break
            sl = train_belief_step(model, optimizer, replay, iter_cfg, grad_scaler)
            for k in accum:
                accum[k] += sl[k]
            eff += 1
        if eff > 0:
            for k in accum:
                accum[k] /= eff

        ev: dict[str, Any] = {k: float("nan") for k in [
            "eval_profit_p0", "eval_profit_p1", "eval_profit_diff_p0_minus_p1",
            "eval_win_rate_p0", "eval_mean_episode_length", "eval_truncation_rate",
            "eval_bots_win_rate_p0", "eval_bots_win_rate_p1", "eval_bots_win_rate_balanced",
        ]}
        if iteration % args.eval_every == 0:
            print(f"[eval belief] iter={iteration:04d}")
            ev.update(
                evaluate_model(
                    model,
                    args.eval_episodes,
                    args.eval_sims,
                    args.eval_max_moves,
                    args.device,
                    args.seed * 1000 + iteration * 100,
                    args.env_variant,
                    args.min_raise_increment,
                )
            )
            bot_ov, bot_det = evaluate_vs_bots(
                model, bot_names, args.eval_bot_episodes_per_bot,
                args.eval_sims, args.eval_max_moves, args.device,
                args.seed * 1_000_000 + iteration * 1000,
                args.env_variant,
                args.min_raise_increment,
            )
            ev.update(bot_ov)
            for bn in bot_names:
                bs = bot_det.get(bn, {})
                print(
                    f"[eval belief][bot {bn}] games={bs.get('games_balanced', 0)} "
                    f"win_p0={bs.get('win_rate_p0', float('nan')):.3f} "
                    f"win_p1={bs.get('win_rate_p1', float('nan')):.3f} "
                    f"win_bal={bs.get('win_rate_balanced', float('nan')):.3f}"
                )

        history["iteration"].append(float(iteration))
        for k in accum:
            history[k].append(float(accum[k]))
        history["replay_steps"].append(float(replay.total_steps()))
        history["selfplay_mean_steps"].append(mean(ep_lens) if ep_lens else 0)
        history["selfplay_terminated_fraction"].append(mean(term_flags) if term_flags else 0)
        history["selfplay_policy_entropy"].append(mean(entropies) if entropies else 0)
        history["selfplay_root_value"].append(mean(root_vals) if root_vals else 0)
        history["selfplay_num_simulations"].append(float(sp_sims))
        history["winner_loss_weight"].append(float(wlw))
        history["rank_loss_weight"].append(float(rlw))
        for k in ev:
            history[k].append(float(ev[k]))

        _save_metrics(history, out_dir)
        _save_graphs(history, graph_dir)

        if iteration % max(1, args.checkpoint_every) == 0 or iteration == args.iterations:
            torch.save({
                "iteration": iteration,
                "model_state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "train_config": vars(args),
            }, ckpt_dir / f"checkpoint_iter_{iteration}.pt")
        if args.opponent_pool_size > 0 and iteration % max(1, args.opponent_snapshot_every) == 0:
            pool.append(copy.deepcopy((model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()))
            if len(pool) > args.opponent_pool_size:
                pool = pool[-args.opponent_pool_size:]

        print(
            f"[iter {iteration:04d}] loss={accum['loss_total']:.4f} "
            f"winner_loss={accum['loss_winner']:.4f} sims={sp_sims} "
            f"selfplay_eval_win_rate={ev['eval_win_rate_p0']} "
            f"bots_eval_balanced={ev['eval_bots_win_rate_balanced']} "
            f"mix(bot={bot_frac:.2f},ckpt={ckpt_frac:.2f})"
        )

    print(f"Belief training complete. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
