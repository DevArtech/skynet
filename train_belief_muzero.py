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

from belief_muzero_mcts import run_belief_mcts
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from belief_muzero_train import (
    BeliefReplayBuffer,
    BeliefTrainConfig,
    create_belief_optimizer,
    generate_belief_self_play_episode,
    train_belief_step,
)
from heuristic_bots import BOT_REGISTRY, make_heuristic_bot
from muzero_mcts import MCTSConfig
from skyjo_decision_env import DECISION_ACTION_SPACE, SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _entropy(probs: list[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def _piecewise_constant_schedule(
    iteration: int,
    initial_value: int,
    mid_value: int,
    final_value: int,
    mid_start_iter: int,
    final_start_iter: int,
) -> int:
    if iteration >= final_start_iter:
        return int(final_value)
    if iteration >= mid_start_iter:
        return int(mid_value)
    return int(initial_value)


def _select_dirichlet_params(
    iteration: int,
    warmup_iters: int,
    alpha_initial: float,
    frac_initial: float,
    alpha_late: float,
    frac_late: float,
) -> tuple[float, float]:
    if iteration <= warmup_iters:
        return float(alpha_initial), float(frac_initial)
    return float(alpha_late), float(frac_late)


def _linear_ramp(
    iteration: int,
    start_iter: int,
    end_iter: int,
    start_value: float,
    end_value: float,
) -> float:
    if iteration <= start_iter:
        return float(start_value)
    if iteration >= end_iter:
        return float(end_value)
    span = max(1, end_iter - start_iter)
    t = (iteration - start_iter) / span
    return float(start_value + t * (end_value - start_value))


def _checkpoint_iter(path: Path) -> int:
    name = path.stem
    if not name.startswith("checkpoint_iter_"):
        return -1
    suffix = name.replace("checkpoint_iter_", "", 1)
    try:
        return int(suffix)
    except ValueError:
        return -1


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = [p for p in checkpoint_dir.glob("checkpoint_iter_*.pt") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=_checkpoint_iter)
    return candidates[-1]


def _build_history_template() -> dict[str, list[float]]:
    return {
        "iteration": [],
        "loss_total": [],
        "loss_policy": [],
        "loss_value": [],
        "loss_reward": [],
        "loss_winner": [],
        "loss_rank": [],
        "grad_norm": [],
        "replay_steps": [],
        "selfplay_policy_entropy": [],
        "selfplay_root_value": [],
        "selfplay_num_simulations": [],
        "selfplay_root_dirichlet_alpha": [],
        "selfplay_root_exploration_fraction": [],
        "selfplay_vs_bot_fraction": [],
        "selfplay_vs_checkpoint_fraction": [],
        "winner_loss_weight": [],
        "rank_loss_weight": [],
        "eval_mean_score_p0": [],
        "eval_mean_score_p1": [],
        "eval_mean_score_diff_p0_minus_p1": [],
        "eval_score_diff_p25": [],
        "eval_score_diff_p50": [],
        "eval_score_diff_p75": [],
        "eval_win_rate_p0": [],
        "eval_tie_rate": [],
        "eval_mean_episode_length": [],
        "eval_truncation_rate": [],
        "eval_bots_mean_score_diff_p0_minus_p1": [],
        "eval_bots_win_rate_p0": [],
        "eval_value_mse": [],
        "eval_value_mae": [],
        "eval_value_pred_mean": [],
        "eval_value_target_mean": [],
    }


def _load_history(history_path: Path, template: dict[str, list[float]]) -> dict[str, list[float]]:
    if not history_path.exists():
        return {k: list(v) for k, v in template.items()}
    with history_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    loaded: dict[str, list[float]] = {k: [] for k in template}
    for key in template:
        values = payload.get(key, [])
        if isinstance(values, list):
            loaded[key] = [float(v) for v in values]
    return loaded


def _trim_history(history: dict[str, list[float]], target_rows: int) -> None:
    target = max(0, int(target_rows))
    for key, values in history.items():
        if len(values) > target:
            history[key] = values[:target]
    row_count = len(history["iteration"])
    for key, values in history.items():
        if key == "iteration":
            continue
        if len(values) < row_count:
            history[key].extend([float("nan")] * (row_count - len(values)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    clamped = max(0.0, min(1.0, float(q)))
    pos = clamped * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def _parse_bot_names(raw: str) -> list[str]:
    names = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    if not names:
        return []
    invalid = [name for name in names if name not in BOT_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown heuristic bot names: {invalid}. Available: {sorted(BOT_REGISTRY)}")
    return names


def _rebuild_opponent_pool(
    checkpoint_dir: Path,
    max_pool_size: int,
    up_to_iteration: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    if max_pool_size <= 0:
        return []
    checkpoints = [p for p in checkpoint_dir.glob("checkpoint_iter_*.pt") if p.is_file()]
    checkpoints = [p for p in checkpoints if 0 < _checkpoint_iter(p) <= up_to_iteration]
    checkpoints.sort(key=_checkpoint_iter)
    checkpoints = checkpoints[-max_pool_size:]
    pool: list[dict[str, torch.Tensor]] = []
    for path in checkpoints:
        payload = torch.load(path, map_location=torch.device(device))
        model_state = payload.get("model_state_dict")
        if isinstance(model_state, dict):
            pool.append(copy.deepcopy(model_state))
    return pool


def _save_metrics(history: dict[str, list[float]], output_dir: Path) -> None:
    with (output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    keys = list(history.keys())
    rows = len(history["iteration"])
    with (output_dir / "metrics_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(rows):
            writer.writerow({k: history[k][i] for k in keys})


def _plot_series(x: list[int], y: list[float], title: str, ylabel: str, out_path: Path) -> None:
    finite_points = [(xi, yi) for xi, yi in zip(x, y, strict=False) if math.isfinite(float(yi))]
    if not finite_points:
        return
    xf = [p[0] for p in finite_points]
    yf = [float(p[1]) for p in finite_points]

    plt.figure(figsize=(7, 4))
    plt.plot(xf, yf, linewidth=2.0, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_graphs(history: dict[str, list[float]], graph_dir: Path) -> None:
    _ensure_dir(graph_dir)
    x = [int(v) for v in history["iteration"]]
    specs = [
        ("loss_total", "Total Loss", "Loss", "training_loss_total.png"),
        ("loss_policy", "Policy Loss", "Loss", "training_loss_policy.png"),
        ("loss_value", "Value Loss", "Loss", "training_loss_value.png"),
        ("loss_reward", "Reward Loss", "Loss", "training_loss_reward.png"),
        ("loss_winner", "Winner Head Loss", "Loss", "training_loss_winner.png"),
        ("loss_rank", "Rank Head Loss", "Loss", "training_loss_rank.png"),
        ("grad_norm", "Gradient Norm", "Norm", "diagnostics_grad_norm.png"),
        ("replay_steps", "Replay Steps", "Steps", "diagnostics_replay_steps.png"),
        ("selfplay_policy_entropy", "Self-Play Policy Entropy", "Entropy", "diagnostics_policy_entropy.png"),
        ("selfplay_root_value", "Self-Play Root Value", "Scalar", "diagnostics_root_value.png"),
        ("selfplay_vs_bot_fraction", "Self-Play vs Heuristic Bot Fraction", "Fraction", "diagnostics_selfplay_vs_bot_fraction.png"),
        (
            "selfplay_vs_checkpoint_fraction",
            "Self-Play vs Checkpoint Opponent Fraction",
            "Fraction",
            "diagnostics_selfplay_vs_checkpoint_fraction.png",
        ),
        ("eval_mean_score_p0", "Eval Mean Final Score (P0)", "Score", "testing_eval_mean_score_p0.png"),
        ("eval_mean_score_p1", "Eval Mean Final Score (P1)", "Score", "testing_eval_mean_score_p1.png"),
        (
            "eval_mean_score_diff_p0_minus_p1",
            "Eval Mean Score Diff (P0 - P1)",
            "Score Diff",
            "testing_eval_mean_score_diff_p0_minus_p1.png",
        ),
        ("eval_score_diff_p25", "Eval Score Diff P25", "Score Diff", "testing_eval_score_diff_p25.png"),
        ("eval_score_diff_p50", "Eval Score Diff P50 (Median)", "Score Diff", "testing_eval_score_diff_p50.png"),
        ("eval_score_diff_p75", "Eval Score Diff P75", "Score Diff", "testing_eval_score_diff_p75.png"),
        ("eval_win_rate_p0", "Eval Win Rate (P0)", "Win Rate", "testing_eval_win_rate_p0.png"),
        ("eval_tie_rate", "Eval Tie Rate", "Rate", "testing_eval_tie_rate.png"),
        ("eval_mean_episode_length", "Eval Episode Length", "Steps", "testing_eval_episode_length.png"),
        ("eval_truncation_rate", "Eval Truncation Rate", "Rate", "testing_eval_truncation_rate.png"),
        (
            "eval_bots_mean_score_diff_p0_minus_p1",
            "Eval vs Heuristic Bots: Mean Score Diff (P0 - P1)",
            "Score Diff",
            "testing_eval_bots_mean_score_diff_p0_minus_p1.png",
        ),
        ("eval_bots_win_rate_p0", "Eval vs Heuristic Bots: Win Rate (P0)", "Win Rate", "testing_eval_bots_win_rate_p0.png"),
        ("eval_value_mse", "Eval Value MSE", "MSE", "testing_eval_value_mse.png"),
        ("eval_value_mae", "Eval Value MAE", "MAE", "testing_eval_value_mae.png"),
    ]
    for key, title, ylabel, filename in specs:
        if key in history and len(history[key]) == len(x):
            _plot_series(x, history[key], title, ylabel, graph_dir / filename)


def evaluate_belief_model(
    model: BeliefAwareMuZeroNet,
    episodes: int,
    sims: int,
    max_moves: int,
    device: str,
    seed_base: int,
    env_mode: str,
) -> dict[str, float]:
    cfg = MCTSConfig(num_simulations=sims, temperature=1e-8, add_exploration_noise=False, root_exploration_fraction=0.0)
    wins = 0.0
    ties = 0
    lengths: list[int] = []
    truncations = 0
    completed_scores_p0: list[float] = []
    completed_scores_p1: list[float] = []
    completed_score_diffs: list[float] = []
    predicted_values: list[float] = []
    target_values: list[float] = []
    eval_start = time.perf_counter()
    progress_every = max(1, episodes // 4)

    for e in range(episodes):
        if env_mode == "decision":
            env = SkyjoDecisionEnv(num_players=2, seed=seed_base + e, setup_mode="auto")
        else:
            env = SkyjoEnv(num_players=2, seed=seed_base + e, setup_mode="auto")
        obs = env.reset()
        terminated = False
        steps = 0
        episode_predictions: list[tuple[int, float]] = []
        while not terminated and (max_moves <= 0 or steps < max_moves):
            actor = int(env.current_player)
            stats = run_belief_mcts(
                model=model,
                observation=obs,
                legal_action_ids=env.legal_actions(),
                ego_player_id=actor,
                config=cfg,
                device=device,
            )
            episode_predictions.append((actor, float(stats.root_value)))
            obs, _, terminated, _ = env.step(stats.action)
            steps += 1
        if not terminated:
            truncations += 1
        lengths.append(steps)
        if terminated:
            score0 = float(env.scores[0])
            score1 = float(env.scores[1])
            completed_scores_p0.append(score0)
            completed_scores_p1.append(score1)
            completed_score_diffs.append(score0 - score1)
            min_score = min(env.scores)
            winners = [i for i, s in enumerate(env.scores) if s == min_score]
            if len(winners) > 1:
                ties += 1
            wins += 1.0 / float(len(winners)) if 0 in winners else 0.0
            for actor, pred in episode_predictions:
                own = float(env.scores[actor])
                others = [float(env.scores[i]) for i in range(env.num_players) if i != actor]
                opp_mean = float(sum(others) / max(1, len(others)))
                predicted_values.append(pred)
                target_values.append(opp_mean - own)
        done = e + 1
        if done % progress_every == 0 or done == episodes:
            elapsed = time.perf_counter() - eval_start
            print(
                f"[eval belief] {done}/{episodes} episodes "
                f"elapsed={elapsed:.1f}s "
                f"win_rate_p0={wins / max(1, done):.3f} "
                f"mean_len={mean(lengths):.1f}"
            )

    value_mse = float("nan")
    value_mae = float("nan")
    if predicted_values and target_values:
        sq = [(p - y) ** 2 for p, y in zip(predicted_values, target_values, strict=False)]
        ab = [abs(p - y) for p, y in zip(predicted_values, target_values, strict=False)]
        value_mse = float(sum(sq) / len(sq))
        value_mae = float(sum(ab) / len(ab))

    return {
        "eval_mean_score_p0": float(mean(completed_scores_p0) if completed_scores_p0 else float("nan")),
        "eval_mean_score_p1": float(mean(completed_scores_p1) if completed_scores_p1 else float("nan")),
        "eval_mean_score_diff_p0_minus_p1": float(mean(completed_score_diffs) if completed_score_diffs else float("nan")),
        "eval_score_diff_p25": _percentile(completed_score_diffs, 0.25),
        "eval_score_diff_p50": _percentile(completed_score_diffs, 0.50),
        "eval_score_diff_p75": _percentile(completed_score_diffs, 0.75),
        "eval_win_rate_p0": float(wins / max(1, episodes)),
        "eval_tie_rate": float(ties / max(1, episodes)),
        "eval_mean_episode_length": float(mean(lengths) if lengths else 0.0),
        "eval_truncation_rate": float(truncations / max(1, episodes)),
        "eval_value_mse": value_mse,
        "eval_value_mae": value_mae,
        "eval_value_pred_mean": float(mean(predicted_values) if predicted_values else float("nan")),
        "eval_value_target_mean": float(mean(target_values) if target_values else float("nan")),
    }


def evaluate_belief_vs_heuristic_bots(
    model: BeliefAwareMuZeroNet,
    bot_names: list[str],
    episodes_per_bot: int,
    sims: int,
    max_moves: int,
    device: str,
    seed_base: int,
    env_mode: str,
    bot_epsilon: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not bot_names or episodes_per_bot <= 0:
        empty = {
            "eval_bots_mean_score_diff_p0_minus_p1": float("nan"),
            "eval_bots_win_rate_p0": float("nan"),
        }
        return empty, {}
    cfg = MCTSConfig(num_simulations=sims, temperature=1e-8, add_exploration_noise=False, root_exploration_fraction=0.0)
    per_bot: dict[str, dict[str, float]] = {}
    all_score_diffs: list[float] = []
    all_wins = 0.0
    all_games = 0

    for bot_idx, bot_name in enumerate(bot_names):
        score_diffs: list[float] = []
        wins = 0.0
        completed = 0
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
                        device=device,
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

        if completed > 0:
            per_bot[bot_name] = {
                "games": float(completed),
                "mean_score_diff_p0_minus_p1": float(mean(score_diffs)),
                "win_rate_p0": float(wins / completed),
            }
            all_score_diffs.extend(score_diffs)
            all_wins += wins
            all_games += completed
        else:
            per_bot[bot_name] = {
                "games": 0.0,
                "mean_score_diff_p0_minus_p1": float("nan"),
                "win_rate_p0": float("nan"),
            }

    overall = {
        "eval_bots_mean_score_diff_p0_minus_p1": float(mean(all_score_diffs) if all_score_diffs else float("nan")),
        "eval_bots_win_rate_p0": float(all_wins / all_games) if all_games > 0 else float("nan"),
    }
    return overall, per_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Train belief-aware MuZero and save diagnostics plots.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--selfplay-episodes-per-iter", type=int, default=32)
    parser.add_argument("--train-steps-per-iter", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-bot-episodes-per-bot", type=int, default=20)
    parser.add_argument("--selfplay-sims", type=int, default=50)
    parser.add_argument("--selfplay-sims-mid", type=int, default=100)
    parser.add_argument("--selfplay-sims-final", type=int, default=200)
    parser.add_argument("--selfplay-sims-mid-iter", type=int, default=200)
    parser.add_argument("--selfplay-sims-final-iter", type=int, default=500)
    parser.add_argument("--eval-sims", type=int, default=100)
    parser.add_argument("--dirichlet-alpha-initial", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac-initial", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha-late", type=float, default=0.15)
    parser.add_argument("--dirichlet-frac-late", type=float, default=0.10)
    parser.add_argument("--dirichlet-switch-iter", type=int, default=200)
    parser.add_argument("--opponent-pool-size", type=int, default=10)
    parser.add_argument("--opponent-latest-prob", type=float, default=0.7)
    parser.add_argument("--opponent-checkpoint-fraction", type=float, default=0.2)
    parser.add_argument("--heuristic-bot-fraction", type=float, default=0.3)
    parser.add_argument(
        "--heuristic-bot-names",
        type=str,
        default="greedy_value_replacement,information_first_flip,column_hunter,risk_aware_unknown_replacement,end_round_aggro,anti_discard",
    )
    parser.add_argument("--heuristic-bot-epsilon", type=float, default=0.02)
    parser.add_argument("--opponent-snapshot-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--time-penalty-per-step",
        type=float,
        default=0.0,
        help="Subtract this amount from the acting player's reward every step.",
    )
    parser.add_argument(
        "--time-penalty-max-per-episode",
        type=float,
        default=-1.0,
        help="Cap the cumulative time penalty applied over one episode. Use <0 to disable cap.",
    )
    parser.add_argument("--max-moves-per-episode", type=int, default=2000)
    parser.add_argument(
        "--eval-max-moves",
        type=int,
        default=2000,
        help="Max moves during evaluation only. Use <=0 for no cap.",
    )
    parser.add_argument("--replay-capacity-episodes", type=int, default=5000)
    parser.add_argument("--winner-loss-weight-initial", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight-initial", type=float, default=0.05)
    parser.add_argument("--winner-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--aux-loss-ramp-start-iter", type=int, default=200)
    parser.add_argument("--aux-loss-ramp-end-iter", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Resume from an explicit checkpoint file path.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/muzero_belief")
    parser.add_argument(
        "--env-mode",
        type=str,
        choices=["decision", "macro"],
        default="decision",
        help="Use decision-granularity (16 actions) or macro-action (36 actions) environment.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    bot_names = _parse_bot_names(args.heuristic_bot_names)
    heuristic_bots = [make_heuristic_bot(name, seed=args.seed + 20_000 + i, epsilon=args.heuristic_bot_epsilon) for i, name in enumerate(bot_names)]

    output_dir = Path(args.output_dir)
    graph_dir = output_dir / "graphs"
    checkpoint_dir = output_dir / "checkpoints"
    _ensure_dir(output_dir)
    _ensure_dir(graph_dir)
    _ensure_dir(checkpoint_dir)

    action_space_size = DECISION_ACTION_SPACE if args.env_mode == "decision" else 36
    model = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space_size=action_space_size)).to(
        torch.device(args.device)
    )
    train_cfg = BeliefTrainConfig(
        unroll_steps=args.unroll_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        time_penalty_per_step=args.time_penalty_per_step,
        time_penalty_max_per_episode=(
            None if args.time_penalty_max_per_episode < 0.0 else args.time_penalty_max_per_episode
        ),
        max_moves_per_episode=args.max_moves_per_episode,
        replay_capacity_episodes=args.replay_capacity_episodes,
        winner_loss_weight=args.winner_loss_weight_initial,
        rank_loss_weight=args.rank_loss_weight_initial,
        device=args.device,
    )
    optimizer = create_belief_optimizer(model, train_cfg)
    replay = BeliefReplayBuffer(capacity_episodes=train_cfg.replay_capacity_episodes)
    opponent_model = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space_size=action_space_size)).to(
        torch.device(args.device)
    )
    opponent_model.eval()
    opponent_pool: list[dict[str, torch.Tensor]] = []

    history_template = _build_history_template()
    history = {k: list(v) for k, v in history_template.items()}
    start_iteration = 1
    if args.resume or args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint) if args.resume_checkpoint else _find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("Resume requested but no checkpoint found.")
        payload = torch.load(checkpoint_path, map_location=torch.device(args.device))
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_iter = int(payload.get("iteration", 0))
        start_iteration = completed_iter + 1
        history = _load_history(output_dir / "metrics_history.json", template=history_template)
        _trim_history(history, target_rows=completed_iter)
        opponent_pool = _rebuild_opponent_pool(
            checkpoint_dir=checkpoint_dir,
            max_pool_size=args.opponent_pool_size,
            up_to_iteration=completed_iter,
            device=args.device,
        )
        print(f"[resume belief] loaded checkpoint={checkpoint_path} completed_iter={completed_iter}")

    if start_iteration > args.iterations:
        print(
            f"[resume belief] nothing to do: completed_iter={start_iteration - 1} "
            f">= target_iterations={args.iterations}"
        )
        return

    for iteration in range(start_iteration, args.iterations + 1):
        selfplay_sims = _piecewise_constant_schedule(
            iteration=iteration,
            initial_value=args.selfplay_sims,
            mid_value=args.selfplay_sims_mid,
            final_value=args.selfplay_sims_final,
            mid_start_iter=args.selfplay_sims_mid_iter,
            final_start_iter=args.selfplay_sims_final_iter,
        )
        dirichlet_alpha, dirichlet_frac = _select_dirichlet_params(
            iteration=iteration,
            warmup_iters=args.dirichlet_switch_iter,
            alpha_initial=args.dirichlet_alpha_initial,
            frac_initial=args.dirichlet_frac_initial,
            alpha_late=args.dirichlet_alpha_late,
            frac_late=args.dirichlet_frac_late,
        )
        selfplay_cfg = MCTSConfig(
            num_simulations=selfplay_sims,
            temperature=1.0,
            add_exploration_noise=True,
            root_dirichlet_alpha=dirichlet_alpha,
            root_exploration_fraction=dirichlet_frac,
        )
        winner_loss_weight = _linear_ramp(
            iteration=iteration,
            start_iter=args.aux_loss_ramp_start_iter,
            end_iter=args.aux_loss_ramp_end_iter,
            start_value=args.winner_loss_weight_initial,
            end_value=args.winner_loss_weight,
        )
        rank_loss_weight = _linear_ramp(
            iteration=iteration,
            start_iter=args.aux_loss_ramp_start_iter,
            end_iter=args.aux_loss_ramp_end_iter,
            start_value=args.rank_loss_weight_initial,
            end_value=args.rank_loss_weight,
        )
        iter_train_cfg = replace(
            train_cfg,
            winner_loss_weight=winner_loss_weight,
            rank_loss_weight=rank_loss_weight,
        )
        entropy_vals: list[float] = []
        root_vals: list[float] = []
        bot_opponent_episodes = 0
        checkpoint_opponent_episodes = 0
        selfplay_start = time.perf_counter()

        for ep_idx in range(args.selfplay_episodes_per_iter):
            ep_seed = random.randint(0, 10_000_000)
            if args.env_mode == "decision":
                env_factory = lambda s=ep_seed: SkyjoDecisionEnv(num_players=2, seed=s, setup_mode="auto")
            else:
                env_factory = lambda s=ep_seed: SkyjoEnv(num_players=2, seed=s, setup_mode="auto")

            selected_opponent: BeliefAwareMuZeroNet | None = None
            opponent_action_selector: Any = None
            use_bot = bool(heuristic_bots) and (random.random() < max(0.0, min(1.0, args.heuristic_bot_fraction)))
            if use_bot:
                bot = random.choice(heuristic_bots)
                opponent_action_selector = lambda o, legal, actor, b=bot: int(b.select_action(o, legal))
                bot_opponent_episodes += 1
            elif args.opponent_pool_size > 0 and opponent_pool:
                use_checkpoint = random.random() < max(0.0, min(1.0, args.opponent_checkpoint_fraction))
                if use_checkpoint:
                    snapshot = random.choice(opponent_pool)
                    opponent_model.load_state_dict(snapshot)
                    selected_opponent = opponent_model
                    checkpoint_opponent_episodes += 1

            learner_player_id = random.randint(0, 1)
            episode = generate_belief_self_play_episode(
                model=model,
                mcts_config=selfplay_cfg,
                opponent_model=selected_opponent,
                learner_player_id=learner_player_id,
                env_factory=env_factory,
                max_moves=train_cfg.max_moves_per_episode,
                device=train_cfg.device,
                time_penalty_per_step=train_cfg.time_penalty_per_step,
                time_penalty_max_per_episode=train_cfg.time_penalty_max_per_episode,
                opponent_action_selector=opponent_action_selector,
            )
            replay.add_episode(episode)
            elapsed = time.perf_counter() - selfplay_start
            print(
                f"[selfplay belief] iter={iteration:04d} "
                f"episode={ep_idx + 1}/{args.selfplay_episodes_per_iter} "
                f"steps={len(episode.steps)} "
                f"terminated={int(episode.terminated)} "
                f"elapsed={elapsed:.1f}s"
            )
            for step in episode.steps:
                entropy_vals.append(_entropy(step.policy_target))
                root_vals.append(step.root_value)

        accum = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_value": 0.0,
            "loss_reward": 0.0,
            "loss_winner": 0.0,
            "loss_rank": 0.0,
            "grad_norm": 0.0,
        }
        effective_steps = 0
        for _ in range(args.train_steps_per_iter):
            if replay.total_steps() < iter_train_cfg.batch_size:
                break
            losses = train_belief_step(model=model, optimizer=optimizer, replay_buffer=replay, config=iter_train_cfg)
            for key in accum:
                accum[key] += losses[key]
            effective_steps += 1
        if effective_steps > 0:
            for key in accum:
                accum[key] /= effective_steps

        eval_metrics: dict[str, Any] = {
            "eval_mean_score_p0": float("nan"),
            "eval_mean_score_p1": float("nan"),
            "eval_mean_score_diff_p0_minus_p1": float("nan"),
            "eval_score_diff_p25": float("nan"),
            "eval_score_diff_p50": float("nan"),
            "eval_score_diff_p75": float("nan"),
            "eval_win_rate_p0": float("nan"),
            "eval_tie_rate": float("nan"),
            "eval_mean_episode_length": float("nan"),
            "eval_truncation_rate": float("nan"),
            "eval_bots_mean_score_diff_p0_minus_p1": float("nan"),
            "eval_bots_win_rate_p0": float("nan"),
            "eval_value_mse": float("nan"),
            "eval_value_mae": float("nan"),
            "eval_value_pred_mean": float("nan"),
            "eval_value_target_mean": float("nan"),
        }
        if iteration % args.eval_every == 0:
            print(f"[eval belief] starting iteration {iteration:04d} with {args.eval_episodes} episodes")
            eval_metrics = evaluate_belief_model(
                model=model,
                episodes=args.eval_episodes,
                sims=args.eval_sims,
                max_moves=args.eval_max_moves,
                device=args.device,
                seed_base=args.seed * 1000 + iteration * 100,
                env_mode=args.env_mode,
            )
            bot_overall, bot_detail = evaluate_belief_vs_heuristic_bots(
                model=model,
                bot_names=bot_names,
                episodes_per_bot=args.eval_bot_episodes_per_bot,
                sims=args.eval_sims,
                max_moves=args.eval_max_moves,
                device=args.device,
                seed_base=args.seed * 1_000_000 + iteration * 1_000,
                env_mode=args.env_mode,
                bot_epsilon=args.heuristic_bot_epsilon,
            )
            eval_metrics.update(bot_overall)
            bot_eval_dir = output_dir / "bot_eval"
            _ensure_dir(bot_eval_dir)
            with (bot_eval_dir / f"iter_{iteration:04d}.json").open("w", encoding="utf-8") as f:
                json.dump({"iteration": iteration, "overall": bot_overall, "per_bot": bot_detail}, f, indent=2)

        history["iteration"].append(float(iteration))
        for key in ("loss_total", "loss_policy", "loss_value", "loss_reward", "loss_winner", "loss_rank", "grad_norm"):
            history[key].append(float(accum[key]))
        history["replay_steps"].append(float(replay.total_steps()))
        history["selfplay_policy_entropy"].append(float(mean(entropy_vals) if entropy_vals else 0.0))
        history["selfplay_root_value"].append(float(mean(root_vals) if root_vals else 0.0))
        history["selfplay_num_simulations"].append(float(selfplay_sims))
        history["selfplay_root_dirichlet_alpha"].append(float(dirichlet_alpha))
        history["selfplay_root_exploration_fraction"].append(float(dirichlet_frac))
        history["selfplay_vs_bot_fraction"].append(float(bot_opponent_episodes / max(1, args.selfplay_episodes_per_iter)))
        history["selfplay_vs_checkpoint_fraction"].append(
            float(checkpoint_opponent_episodes / max(1, args.selfplay_episodes_per_iter))
        )
        history["winner_loss_weight"].append(float(winner_loss_weight))
        history["rank_loss_weight"].append(float(rank_loss_weight))
        history["eval_mean_score_p0"].append(float(eval_metrics["eval_mean_score_p0"]))
        history["eval_mean_score_p1"].append(float(eval_metrics["eval_mean_score_p1"]))
        history["eval_mean_score_diff_p0_minus_p1"].append(float(eval_metrics["eval_mean_score_diff_p0_minus_p1"]))
        history["eval_score_diff_p25"].append(float(eval_metrics["eval_score_diff_p25"]))
        history["eval_score_diff_p50"].append(float(eval_metrics["eval_score_diff_p50"]))
        history["eval_score_diff_p75"].append(float(eval_metrics["eval_score_diff_p75"]))
        history["eval_win_rate_p0"].append(float(eval_metrics["eval_win_rate_p0"]))
        history["eval_tie_rate"].append(float(eval_metrics["eval_tie_rate"]))
        history["eval_mean_episode_length"].append(float(eval_metrics["eval_mean_episode_length"]))
        history["eval_truncation_rate"].append(float(eval_metrics["eval_truncation_rate"]))
        history["eval_bots_mean_score_diff_p0_minus_p1"].append(
            float(eval_metrics["eval_bots_mean_score_diff_p0_minus_p1"])
        )
        history["eval_bots_win_rate_p0"].append(float(eval_metrics["eval_bots_win_rate_p0"]))
        history["eval_value_mse"].append(float(eval_metrics["eval_value_mse"]))
        history["eval_value_mae"].append(float(eval_metrics["eval_value_mae"]))
        history["eval_value_pred_mean"].append(float(eval_metrics["eval_value_pred_mean"]))
        history["eval_value_target_mean"].append(float(eval_metrics["eval_value_target_mean"]))

        _save_metrics(history, output_dir)
        _save_graphs(history, graph_dir)

        if iteration % max(1, args.checkpoint_every) == 0 or iteration == args.iterations:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_config": vars(args),
                },
                checkpoint_dir / f"checkpoint_iter_{iteration}.pt",
            )
        if args.opponent_pool_size > 0 and iteration % max(1, args.opponent_snapshot_every) == 0:
            opponent_pool.append(copy.deepcopy(model.state_dict()))
            if len(opponent_pool) > args.opponent_pool_size:
                opponent_pool = opponent_pool[-args.opponent_pool_size :]

        print(
            f"[iter {iteration:04d}] "
            f"loss={accum['loss_total']:.4f} "
            f"winner_loss={accum['loss_winner']:.4f} "
            f"selfplay_sims={selfplay_sims} "
            f"pool={len(opponent_pool)} "
            f"eval_win_rate={eval_metrics['eval_win_rate_p0']}"
        )

    print(f"Belief-aware training complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
