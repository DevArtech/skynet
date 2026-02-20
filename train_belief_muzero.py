from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
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
from muzero_mcts import MCTSConfig
from skyjo_decision_env import DECISION_ACTION_SPACE, SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _entropy(probs: list[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def _compute_calibration_bins(
    predictions: list[float],
    outcomes: list[int],
    num_bins: int = 10,
) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    if not predictions or not outcomes:
        return bins

    for b in range(num_bins):
        lo = b / num_bins
        hi = (b + 1) / num_bins
        in_bin: list[tuple[float, int]] = []
        for p, y in zip(predictions, outcomes, strict=False):
            if b == num_bins - 1:
                if lo <= p <= hi:
                    in_bin.append((p, y))
            elif lo <= p < hi:
                in_bin.append((p, y))

        count = len(in_bin)
        if count == 0:
            bins.append(
                {
                    "bin_index": float(b),
                    "bin_start": float(lo),
                    "bin_end": float(hi),
                    "count": 0.0,
                    "mean_pred": float("nan"),
                    "empirical_win_rate": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue

        mean_pred = float(sum(p for p, _ in in_bin) / count)
        empirical = float(sum(y for _, y in in_bin) / count)
        bins.append(
            {
                "bin_index": float(b),
                "bin_start": float(lo),
                "bin_end": float(hi),
                "count": float(count),
                "mean_pred": mean_pred,
                "empirical_win_rate": empirical,
                "abs_gap": float(abs(empirical - mean_pred)),
            }
        )
    return bins


def _compute_ece(calibration_bins: list[dict[str, float]]) -> float:
    total = sum(int(bin_row["count"]) for bin_row in calibration_bins)
    if total <= 0:
        return float("nan")
    weighted_gap = 0.0
    for bin_row in calibration_bins:
        count = int(bin_row["count"])
        if count <= 0:
            continue
        weighted_gap += (count / total) * float(bin_row["abs_gap"])
    return float(weighted_gap)


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
        ("eval_mean_return_p0", "Eval Mean Return (P0)", "Return", "testing_eval_mean_return_p0.png"),
        ("eval_win_rate_p0", "Eval Win Rate (P0)", "Win Rate", "testing_eval_win_rate_p0.png"),
        ("eval_mean_episode_length", "Eval Episode Length", "Steps", "testing_eval_episode_length.png"),
        ("eval_truncation_rate", "Eval Truncation Rate", "Rate", "testing_eval_truncation_rate.png"),
        ("eval_value_brier", "Eval Value Brier Score", "Brier", "testing_eval_value_brier.png"),
        ("eval_value_ece", "Eval Value ECE (10 bins)", "ECE", "testing_eval_value_ece.png"),
    ]
    for key, title, ylabel, filename in specs:
        if key in history and len(history[key]) == len(x):
            _plot_series(x, history[key], title, ylabel, graph_dir / filename)


def _save_calibration(
    output_dir: Path,
    iteration: int,
    calibration_bins: list[dict[str, float]],
) -> None:
    calib_dir = output_dir / "calibration"
    _ensure_dir(calib_dir)

    payload: dict[str, Any] = {"iteration": iteration, "bins": calibration_bins}
    with (calib_dir / "latest_value_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (calib_dir / f"value_calibration_iter_{iteration}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = ["bin_index", "bin_start", "bin_end", "count", "mean_pred", "empirical_win_rate", "abs_gap"]
    with (calib_dir / "latest_value_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in calibration_bins:
            writer.writerow(row)

    with (calib_dir / f"value_calibration_iter_{iteration}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in calibration_bins:
            writer.writerow(row)

    x = [row["mean_pred"] for row in calibration_bins if math.isfinite(float(row["mean_pred"]))]
    y = [row["empirical_win_rate"] for row in calibration_bins if math.isfinite(float(row["empirical_win_rate"]))]
    if not x or not y:
        return

    plt.figure(figsize=(5, 5))
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.25, alpha=0.8, label="Perfect calibration")
    plt.plot(x, y, marker="o", linewidth=2.0, label="Belief value calibration")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Predicted win probability")
    plt.ylabel("Empirical win rate")
    plt.title("Belief Value Calibration")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(calib_dir / "latest_value_calibration.png", dpi=140)
    plt.savefig(calib_dir / f"value_calibration_iter_{iteration}.png", dpi=140)
    plt.close()


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
    wins = 0
    lengths: list[int] = []
    truncations = 0
    returns: list[float] = []
    predicted_win_probs: list[float] = []
    realized_outcomes: list[int] = []
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
        total_reward_p0 = 0.0
        episode_predictions: list[tuple[int, float]] = []
        while not terminated and steps < max_moves:
            actor = int(env.current_player)
            stats = run_belief_mcts(
                model=model,
                observation=obs,
                legal_action_ids=env.legal_actions(),
                ego_player_id=actor,
                config=cfg,
                device=device,
            )
            clipped = max(0.0, min(1.0, float(stats.root_value)))
            episode_predictions.append((actor, clipped))
            obs, rewards, terminated, _ = env.step(stats.action)
            total_reward_p0 += float(rewards.get("player_0", 0.0))
            steps += 1
        if not terminated:
            truncations += 1
        winner = min(range(env.num_players), key=lambda p: env.scores[p])
        wins += int(winner == 0)
        lengths.append(steps)
        returns.append(total_reward_p0)
        for actor, pred in episode_predictions:
            predicted_win_probs.append(pred)
            realized_outcomes.append(int(winner == actor))
        done = e + 1
        if done % progress_every == 0 or done == episodes:
            elapsed = time.perf_counter() - eval_start
            print(
                f"[eval belief] {done}/{episodes} episodes "
                f"elapsed={elapsed:.1f}s "
                f"win_rate_p0={wins / max(1, done):.3f} "
                f"mean_len={mean(lengths):.1f}"
            )

    calibration_bins = _compute_calibration_bins(predictions=predicted_win_probs, outcomes=realized_outcomes, num_bins=10)
    brier = float("nan")
    if predicted_win_probs and realized_outcomes:
        brier = float(
            sum((p - y) ** 2 for p, y in zip(predicted_win_probs, realized_outcomes, strict=False))
            / len(predicted_win_probs)
        )

    return {
        "eval_mean_return_p0": float(mean(returns) if returns else 0.0),
        "eval_win_rate_p0": float(wins / max(1, episodes)),
        "eval_mean_episode_length": float(mean(lengths) if lengths else 0.0),
        "eval_truncation_rate": float(truncations / max(1, episodes)),
        "eval_value_brier": brier,
        "eval_value_ece": _compute_ece(calibration_bins),
        "eval_value_pred_mean": float(mean(predicted_win_probs) if predicted_win_probs else float("nan")),
        "eval_value_outcome_mean": float(mean(realized_outcomes) if realized_outcomes else float("nan")),
        "eval_calibration_bins": calibration_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train belief-aware MuZero and save diagnostics plots.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--selfplay-episodes-per-iter", type=int, default=4)
    parser.add_argument("--train-steps-per-iter", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--selfplay-sims", type=int, default=64)
    parser.add_argument("--eval-sims", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-moves-per-episode", type=int, default=800)
    parser.add_argument("--replay-capacity-episodes", type=int, default=1000)
    parser.add_argument("--winner-loss-weight", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
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
        max_moves_per_episode=args.max_moves_per_episode,
        replay_capacity_episodes=args.replay_capacity_episodes,
        winner_loss_weight=args.winner_loss_weight,
        rank_loss_weight=args.rank_loss_weight,
        device=args.device,
    )
    optimizer = create_belief_optimizer(model, train_cfg)
    replay = BeliefReplayBuffer(capacity_episodes=train_cfg.replay_capacity_episodes)
    selfplay_cfg = MCTSConfig(num_simulations=args.selfplay_sims, temperature=1.0, add_exploration_noise=True)

    history: dict[str, list[float]] = {
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
        "eval_mean_return_p0": [],
        "eval_win_rate_p0": [],
        "eval_mean_episode_length": [],
        "eval_truncation_rate": [],
        "eval_value_brier": [],
        "eval_value_ece": [],
        "eval_value_pred_mean": [],
        "eval_value_outcome_mean": [],
    }

    for iteration in range(1, args.iterations + 1):
        entropy_vals: list[float] = []
        root_vals: list[float] = []
        selfplay_start = time.perf_counter()

        for ep_idx in range(args.selfplay_episodes_per_iter):
            ep_seed = random.randint(0, 10_000_000)
            if args.env_mode == "decision":
                env_factory = lambda s=ep_seed: SkyjoDecisionEnv(num_players=2, seed=s, setup_mode="auto")
            else:
                env_factory = lambda s=ep_seed: SkyjoEnv(num_players=2, seed=s, setup_mode="auto")
            episode = generate_belief_self_play_episode(
                model=model,
                mcts_config=selfplay_cfg,
                env_factory=env_factory,
                max_moves=train_cfg.max_moves_per_episode,
                device=train_cfg.device,
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
            if replay.total_steps() < train_cfg.batch_size:
                break
            losses = train_belief_step(model=model, optimizer=optimizer, replay_buffer=replay, config=train_cfg)
            for key in accum:
                accum[key] += losses[key]
            effective_steps += 1
        if effective_steps > 0:
            for key in accum:
                accum[key] /= effective_steps

        eval_metrics: dict[str, Any] = {
            "eval_mean_return_p0": float("nan"),
            "eval_win_rate_p0": float("nan"),
            "eval_mean_episode_length": float("nan"),
            "eval_truncation_rate": float("nan"),
            "eval_value_brier": float("nan"),
            "eval_value_ece": float("nan"),
            "eval_value_pred_mean": float("nan"),
            "eval_value_outcome_mean": float("nan"),
            "eval_calibration_bins": [],
        }
        if iteration % args.eval_every == 0:
            print(f"[eval belief] starting iteration {iteration:04d} with {args.eval_episodes} episodes")
            eval_metrics = evaluate_belief_model(
                model=model,
                episodes=args.eval_episodes,
                sims=args.eval_sims,
                max_moves=args.max_moves_per_episode,
                device=args.device,
                seed_base=args.seed * 1000 + iteration * 100,
                env_mode=args.env_mode,
            )
            _save_calibration(
                output_dir=output_dir,
                iteration=iteration,
                calibration_bins=list(eval_metrics["eval_calibration_bins"]),
            )

        history["iteration"].append(float(iteration))
        for key in ("loss_total", "loss_policy", "loss_value", "loss_reward", "loss_winner", "loss_rank", "grad_norm"):
            history[key].append(float(accum[key]))
        history["replay_steps"].append(float(replay.total_steps()))
        history["selfplay_policy_entropy"].append(float(mean(entropy_vals) if entropy_vals else 0.0))
        history["selfplay_root_value"].append(float(mean(root_vals) if root_vals else 0.0))
        history["eval_mean_return_p0"].append(float(eval_metrics["eval_mean_return_p0"]))
        history["eval_win_rate_p0"].append(float(eval_metrics["eval_win_rate_p0"]))
        history["eval_mean_episode_length"].append(float(eval_metrics["eval_mean_episode_length"]))
        history["eval_truncation_rate"].append(float(eval_metrics["eval_truncation_rate"]))
        history["eval_value_brier"].append(float(eval_metrics["eval_value_brier"]))
        history["eval_value_ece"].append(float(eval_metrics["eval_value_ece"]))
        history["eval_value_pred_mean"].append(float(eval_metrics["eval_value_pred_mean"]))
        history["eval_value_outcome_mean"].append(float(eval_metrics["eval_value_outcome_mean"]))

        _save_metrics(history, output_dir)
        _save_graphs(history, graph_dir)

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_config": vars(args),
                },
                checkpoint_dir / f"checkpoint_iter_{iteration}.pt",
            )

        print(
            f"[iter {iteration:04d}] "
            f"loss={accum['loss_total']:.4f} "
            f"winner_loss={accum['loss_winner']:.4f} "
            f"eval_win_rate={eval_metrics['eval_win_rate_p0']}"
        )

    print(f"Belief-aware training complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
