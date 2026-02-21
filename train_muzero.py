from __future__ import annotations

import argparse
import copy
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
        "grad_norm": [],
        "replay_steps": [],
        "selfplay_mean_steps": [],
        "selfplay_terminated_fraction": [],
        "selfplay_policy_entropy": [],
        "selfplay_root_value": [],
        "selfplay_num_simulations": [],
        "selfplay_root_dirichlet_alpha": [],
        "selfplay_root_exploration_fraction": [],
        "eval_mean_return_p0": [],
        "eval_win_rate_p0": [],
        "eval_tie_rate": [],
        "eval_nontrunc_tie_rate": [],
        "eval_mean_episode_length": [],
        "eval_truncation_rate": [],
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


def evaluate_model(
    model: MuZeroNet,
    num_episodes: int,
    num_simulations: int,
    max_moves: int,
    discount: float,
    device: str,
    seed_base: int,
    env_mode: str,
) -> dict[str, float]:
    eval_cfg = MCTSConfig(
        num_simulations=num_simulations,
        discount=discount,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
    )

    returns: list[float] = []
    wins = 0
    lengths: list[int] = []
    truncations = 0
    tie_episodes = 0
    nontrunc_tie_episodes = 0
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

        while not terminated and (max_moves <= 0 or step_count < max_moves):
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
        is_tie = len(winners) > 1
        if is_tie:
            tie_episodes += 1
            if terminated:
                nontrunc_tie_episodes += 1

        returns.append(total_reward_p0)
        if terminated:
            wins += 1.0 / float(len(winners)) if 0 in winners else 0.0
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
        "eval_win_rate_p0": float(wins / max(1.0, float(num_episodes))),
        "eval_tie_rate": float(tie_episodes / max(1, num_episodes)),
        "eval_nontrunc_tie_rate": float(nontrunc_tie_episodes / max(1, num_episodes - truncations)),
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
        ("eval_tie_rate", "Eval Tie Rate", "Rate", "testing_eval_tie_rate.png"),
        ("eval_nontrunc_tie_rate", "Eval Tie Rate (Non-Truncated)", "Rate", "testing_eval_nontrunc_tie_rate.png"),
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
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--selfplay-episodes-per-iter", type=int, default=32)
    parser.add_argument("--train-steps-per-iter", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=200)
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
    parser.add_argument("--opponent-snapshot-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument(
        "--discount",
        type=float,
        default=0.997,
        help="Discount factor for value targets and MCTS backups (<1 encourages shorter-horizon play).",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--time-penalty-per-step",
        type=float,
        default=0.0002,
        help="Subtract this amount from the acting player's reward every step to encourage shorter games.",
    )
    parser.add_argument(
        "--time-penalty-max-per-episode",
        type=float,
        default=-1.0,
        help="Cap the cumulative time penalty applied over one episode. Use <0 to disable cap.",
    )
    parser.add_argument(
        "--truncation-penalty",
        type=float,
        default=1.0,
        help="Penalty applied when self-play hits max-moves before termination.",
    )
    parser.add_argument("--max-moves-per-episode", type=int, default=2000)
    parser.add_argument(
        "--eval-max-moves",
        type=int,
        default=2000,
        help="Max moves during evaluation only. Use <=0 for no cap.",
    )
    parser.add_argument("--replay-capacity-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Resume from an explicit checkpoint file path.")
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
        discount=args.discount,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        time_penalty_per_step=args.time_penalty_per_step,
        time_penalty_max_per_episode=(
            None if args.time_penalty_max_per_episode < 0.0 else args.time_penalty_max_per_episode
        ),
        truncation_penalty=args.truncation_penalty,
        max_moves_per_episode=args.max_moves_per_episode,
        replay_capacity_episodes=args.replay_capacity_episodes,
        device=args.device,
    )
    optimizer = create_optimizer(model, train_cfg)
    replay = ReplayBuffer(capacity_episodes=train_cfg.replay_capacity_episodes)
    opponent_model = MuZeroNet(model_cfg).to(torch.device(args.device))
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
        print(f"[resume baseline] loaded checkpoint={checkpoint_path} completed_iter={completed_iter}")

    if start_iteration > args.iterations:
        print(
            f"[resume baseline] nothing to do: completed_iter={start_iteration - 1} "
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
            discount=args.discount,
            temperature=1.0,
            add_exploration_noise=True,
            root_dirichlet_alpha=dirichlet_alpha,
            root_exploration_fraction=dirichlet_frac,
        )
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

            selected_opponent: MuZeroNet | None = None
            if args.opponent_pool_size > 0 and opponent_pool:
                use_latest = random.random() < max(0.0, min(1.0, args.opponent_latest_prob))
                if not use_latest:
                    snapshot = random.choice(opponent_pool)
                    opponent_model.load_state_dict(snapshot)
                    selected_opponent = opponent_model

            learner_player_id = random.randint(0, 1)
            episode = generate_self_play_episode(
                model=model,
                mcts_config=selfplay_cfg,
                opponent_model=selected_opponent,
                learner_player_id=learner_player_id,
                env_factory=env_factory,
                max_moves=train_cfg.max_moves_per_episode,
                device=train_cfg.device,
                time_penalty_per_step=train_cfg.time_penalty_per_step,
                time_penalty_max_per_episode=train_cfg.time_penalty_max_per_episode,
                truncation_penalty=train_cfg.truncation_penalty,
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
            "eval_tie_rate": float("nan"),
            "eval_nontrunc_tie_rate": float("nan"),
            "eval_mean_episode_length": float("nan"),
            "eval_truncation_rate": float("nan"),
        }
        if iteration % args.eval_every == 0:
            print(f"[eval baseline] starting iteration {iteration:04d} with {args.eval_episodes} episodes")
            eval_metrics = evaluate_model(
                model=model,
                num_episodes=args.eval_episodes,
                num_simulations=args.eval_sims,
                max_moves=args.eval_max_moves,
                discount=args.discount,
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
        history["selfplay_num_simulations"].append(float(selfplay_sims))
        history["selfplay_root_dirichlet_alpha"].append(float(dirichlet_alpha))
        history["selfplay_root_exploration_fraction"].append(float(dirichlet_frac))
        history["eval_mean_return_p0"].append(float(eval_metrics["eval_mean_return_p0"]))
        history["eval_win_rate_p0"].append(float(eval_metrics["eval_win_rate_p0"]))
        history["eval_tie_rate"].append(float(eval_metrics["eval_tie_rate"]))
        history["eval_nontrunc_tie_rate"].append(float(eval_metrics["eval_nontrunc_tie_rate"]))
        history["eval_mean_episode_length"].append(float(eval_metrics["eval_mean_episode_length"]))
        history["eval_truncation_rate"].append(float(eval_metrics["eval_truncation_rate"]))

        _save_metrics(history, output_dir=output_dir)
        _save_graphs(history, graph_dir=graph_dir)

        if iteration % max(1, args.checkpoint_every) == 0 or iteration == args.iterations:
            checkpoint = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_config": vars(args),
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_iter_{iteration}.pt")
        if args.opponent_pool_size > 0 and iteration % max(1, args.opponent_snapshot_every) == 0:
            opponent_pool.append(copy.deepcopy(model.state_dict()))
            if len(opponent_pool) > args.opponent_pool_size:
                opponent_pool = opponent_pool[-args.opponent_pool_size :]

        print(
            f"[iter {iteration:04d}] "
            f"loss={losses['loss_total']:.4f} "
            f"replay_steps={replay.total_steps()} "
            f"selfplay_sims={selfplay_sims} "
            f"pool={len(opponent_pool)} "
            f"eval_win_rate={eval_metrics['eval_win_rate_p0']}"
        )

    print(f"Training complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
