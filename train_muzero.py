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

from muzero_mcts import MCTSConfig, run_mcts
from muzero_model import MuZeroNet, build_default_skyjo_muzero_config
from muzero_train import (
    MuZeroTrainConfig,
    ReplayBuffer,
    create_optimizer,
    generate_self_play_episode,
    train_step,
)
from skyjo_decision_env import DECISION_ACTION_SPACE, SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _entropy(probs: list[float], eps: float = 1e-12) -> float:
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model: MuZeroNet,
    num_episodes: int,
    num_simulations: int,
    max_moves: int,
    device: str,
    seed_base: int,
    env_mode: str,
) -> dict[str, float]:
    eval_cfg = MCTSConfig(
        num_simulations=num_simulations,
        discount=1.0,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
    )

    returns: list[float] = []
    wins = 0
    lengths: list[int] = []
    truncations = 0
    eval_start = time.perf_counter()
    progress_every = max(1, num_episodes // 4)

    for ep_idx in range(num_episodes):
        env: Any
        if env_mode == "decision":
            env = SkyjoDecisionEnv(num_players=2, seed=seed_base + ep_idx, setup_mode="auto")
        else:
            env = SkyjoEnv(num_players=2, seed=seed_base + ep_idx, setup_mode="auto")
        obs = env.reset()
        terminated = False
        step_count = 0
        total_reward_p0 = 0.0

        while not terminated and step_count < max_moves:
            legal_actions = env.legal_actions()
            stats = run_mcts(
                model=model,
                observation=obs,
                legal_action_ids=legal_actions,
                config=eval_cfg,
                device=device,
            )
            obs, rewards, terminated, _ = env.step(stats.action)
            total_reward_p0 += float(rewards.get("player_0", 0.0))
            step_count += 1

        if not terminated:
            truncations += 1
        min_score = min(env.scores)
        winners = [i for i, score in enumerate(env.scores) if score == min_score]

        returns.append(total_reward_p0)
        wins += int(0 in winners)
        lengths.append(step_count)
        done = ep_idx + 1
        if done % progress_every == 0 or done == num_episodes:
            elapsed = time.perf_counter() - eval_start
            print(
                f"[eval baseline] {done}/{num_episodes} episodes "
                f"elapsed={elapsed:.1f}s "
                f"win_rate_p0={wins / max(1, done):.3f} "
                f"mean_len={mean(lengths):.1f}"
            )

    return {
        "eval_mean_return_p0": float(mean(returns) if returns else 0.0),
        "eval_win_rate_p0": float(wins / max(1, num_episodes)),
        "eval_mean_episode_length": float(mean(lengths) if lengths else 0.0),
        "eval_truncation_rate": float(truncations / max(1, num_episodes)),
    }


def _plot_single_series(x: list[int], y: list[float], title: str, y_label: str, out_path: Path) -> None:
    finite_points = [(xi, yi) for xi, yi in zip(x, y, strict=False) if math.isfinite(float(yi))]
    if not finite_points:
        return
    xf = [p[0] for p in finite_points]
    yf = [float(p[1]) for p in finite_points]

    plt.figure(figsize=(7, 4))
    plt.plot(xf, yf, linewidth=2.0, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_graphs(history: dict[str, list[float]], graph_dir: Path) -> None:
    _ensure_dir(graph_dir)
    x = [int(v) for v in history["iteration"]]

    graph_specs = [
        ("loss_total", "Total Loss", "Loss", "training_loss_total.png"),
        ("loss_policy", "Policy Loss", "Loss", "training_loss_policy.png"),
        ("loss_value", "Value Loss", "Loss", "training_loss_value.png"),
        ("loss_reward", "Reward Loss", "Loss", "training_loss_reward.png"),
        ("grad_norm", "Gradient Norm", "Norm", "diagnostics_grad_norm.png"),
        ("replay_steps", "Replay Steps", "Steps", "diagnostics_replay_steps.png"),
        ("selfplay_mean_steps", "Self-Play Episode Length", "Steps", "training_selfplay_episode_length.png"),
        ("selfplay_terminated_fraction", "Self-Play Termination Fraction", "Fraction", "training_terminated_fraction.png"),
        ("selfplay_policy_entropy", "Self-Play Policy Entropy", "Entropy", "diagnostics_policy_entropy.png"),
        ("selfplay_root_value", "Self-Play Root Value", "Scalar", "diagnostics_root_value.png"),
        ("eval_mean_return_p0", "Eval Mean Return (Player 0)", "Return", "testing_eval_mean_return_p0.png"),
        ("eval_win_rate_p0", "Eval Win Rate (Player 0)", "Win Rate", "testing_eval_win_rate_p0.png"),
        ("eval_mean_episode_length", "Eval Episode Length", "Steps", "testing_eval_episode_length.png"),
        ("eval_truncation_rate", "Eval Truncation Rate", "Rate", "testing_eval_truncation_rate.png"),
    ]

    for key, title, y_label, filename in graph_specs:
        if key in history and len(history[key]) == len(x):
            _plot_single_series(x=x, y=history[key], title=title, y_label=y_label, out_path=graph_dir / filename)


def _save_metrics(history: dict[str, list[float]], output_dir: Path) -> None:
    json_path = output_dir / "metrics_history.json"
    csv_path = output_dir / "metrics_history.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    keys = list(history.keys())
    row_count = len(history["iteration"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(row_count):
            row = {k: history[k][i] for k in keys}
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline MuZero on Skyjo and save diagnostics plots.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--selfplay-episodes-per-iter", type=int, default=4)
    parser.add_argument("--train-steps-per-iter", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--selfplay-sims", type=int, default=64)
    parser.add_argument("--eval-sims", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-moves-per-episode", type=int, default=800)
    parser.add_argument("--replay-capacity-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/muzero_baseline")
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
    model_cfg = build_default_skyjo_muzero_config(action_space_size=action_space_size)
    model = MuZeroNet(model_cfg).to(torch.device(args.device))
    train_cfg = MuZeroTrainConfig(
        unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_moves_per_episode=args.max_moves_per_episode,
        replay_capacity_episodes=args.replay_capacity_episodes,
        device=args.device,
    )
    optimizer = create_optimizer(model, train_cfg)
    replay = ReplayBuffer(capacity_episodes=train_cfg.replay_capacity_episodes)
    selfplay_cfg = MCTSConfig(num_simulations=args.selfplay_sims, temperature=1.0, add_exploration_noise=True)

    history: dict[str, list[float]] = {
        "iteration": [],
        "loss_total": [],
        "loss_policy": [],
        "loss_value": [],
        "loss_reward": [],
        "grad_norm": [],
        "replay_steps": [],
        "selfplay_mean_steps": [],
        "selfplay_terminated_fraction": [],
        "selfplay_policy_entropy": [],
        "selfplay_root_value": [],
        "eval_mean_return_p0": [],
        "eval_win_rate_p0": [],
        "eval_mean_episode_length": [],
        "eval_truncation_rate": [],
    }

    for iteration in range(1, args.iterations + 1):
        episode_lengths: list[int] = []
        terminated_flags: list[float] = []
        policy_entropies: list[float] = []
        root_values: list[float] = []
        selfplay_start = time.perf_counter()

        for ep_idx in range(args.selfplay_episodes_per_iter):
            env_seed = random.randint(0, 10_000_000)
            if args.env_mode == "decision":
                env_factory = lambda s=env_seed: SkyjoDecisionEnv(num_players=2, seed=s, setup_mode="auto")
            else:
                env_factory = lambda s=env_seed: SkyjoEnv(num_players=2, seed=s, setup_mode="auto")
            episode = generate_self_play_episode(
                model=model,
                mcts_config=selfplay_cfg,
                env_factory=env_factory,
                max_moves=train_cfg.max_moves_per_episode,
                device=train_cfg.device,
            )
            replay.add_episode(episode)
            elapsed = time.perf_counter() - selfplay_start
            print(
                f"[selfplay baseline] iter={iteration:04d} "
                f"episode={ep_idx + 1}/{args.selfplay_episodes_per_iter} "
                f"steps={len(episode.steps)} "
                f"terminated={int(episode.terminated)} "
                f"elapsed={elapsed:.1f}s"
            )
            episode_lengths.append(len(episode.steps))
            terminated_flags.append(1.0 if episode.terminated else 0.0)
            for step in episode.steps:
                policy_entropies.append(_entropy(step.policy_target))
                root_values.append(step.root_value)

        losses = {"loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0, "loss_reward": 0.0, "grad_norm": 0.0}
        effective_train_steps = 0
        for _ in range(args.train_steps_per_iter):
            if replay.total_steps() < train_cfg.batch_size:
                break
            step_losses = train_step(model=model, optimizer=optimizer, replay_buffer=replay, config=train_cfg)
            for k in losses:
                losses[k] += step_losses[k]
            effective_train_steps += 1

        if effective_train_steps > 0:
            for k in losses:
                losses[k] /= effective_train_steps

        eval_metrics = {
            "eval_mean_return_p0": float("nan"),
            "eval_win_rate_p0": float("nan"),
            "eval_mean_episode_length": float("nan"),
            "eval_truncation_rate": float("nan"),
        }
        if iteration % args.eval_every == 0:
            print(f"[eval baseline] starting iteration {iteration:04d} with {args.eval_episodes} episodes")
            eval_metrics = evaluate_model(
                model=model,
                num_episodes=args.eval_episodes,
                num_simulations=args.eval_sims,
                max_moves=args.max_moves_per_episode,
                device=args.device,
                seed_base=args.seed * 1000 + iteration * 100,
                env_mode=args.env_mode,
            )

        history["iteration"].append(float(iteration))
        history["loss_total"].append(float(losses["loss_total"]))
        history["loss_policy"].append(float(losses["loss_policy"]))
        history["loss_value"].append(float(losses["loss_value"]))
        history["loss_reward"].append(float(losses["loss_reward"]))
        history["grad_norm"].append(float(losses["grad_norm"]))
        history["replay_steps"].append(float(replay.total_steps()))
        history["selfplay_mean_steps"].append(float(mean(episode_lengths) if episode_lengths else 0.0))
        history["selfplay_terminated_fraction"].append(float(mean(terminated_flags) if terminated_flags else 0.0))
        history["selfplay_policy_entropy"].append(float(mean(policy_entropies) if policy_entropies else 0.0))
        history["selfplay_root_value"].append(float(mean(root_values) if root_values else 0.0))
        history["eval_mean_return_p0"].append(float(eval_metrics["eval_mean_return_p0"]))
        history["eval_win_rate_p0"].append(float(eval_metrics["eval_win_rate_p0"]))
        history["eval_mean_episode_length"].append(float(eval_metrics["eval_mean_episode_length"]))
        history["eval_truncation_rate"].append(float(eval_metrics["eval_truncation_rate"]))

        _save_metrics(history, output_dir=output_dir)
        _save_graphs(history, graph_dir=graph_dir)

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            checkpoint = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_config": vars(args),
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_iter_{iteration}.pt")

        print(
            f"[iter {iteration:04d}] "
            f"loss={losses['loss_total']:.4f} "
            f"replay_steps={replay.total_steps()} "
            f"eval_win_rate={eval_metrics['eval_win_rate_p0']}"
        )

    print(f"Training complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
