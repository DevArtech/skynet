from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch

from belief_muzero_mcts import run_belief_mcts
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from muzero_mcts import MCTSConfig, run_mcts
from muzero_model import MuZeroNet, build_default_skyjo_muzero_config
from skyjo_decision_env import SkyjoDecisionEnv
from skyjo_env import SkyjoEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_baseline(path: str, device: torch.device) -> MuZeroNet:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    action_space_size = state_dict["prediction.policy_head.net.6.weight"].shape[0]
    model = MuZeroNet(build_default_skyjo_muzero_config(action_space_size=action_space_size)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_belief(path: str, device: torch.device) -> BeliefAwareMuZeroNet:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    action_space_size = state_dict["prediction.policy_head.net.6.weight"].shape[0]
    model = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space_size=action_space_size)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _select_action(
    policy_name: str,
    baseline: MuZeroNet | None,
    belief: BeliefAwareMuZeroNet,
    obs: dict,
    legal_actions: list[int],
    current_player: int,
    cfg: MCTSConfig,
    ablate_belief_head: bool,
    device: str,
) -> int:
    if policy_name == "baseline":
        if baseline is None:
            raise ValueError("Baseline policy requested but baseline model is not loaded.")
        stats = run_mcts(
            model=baseline,
            observation=obs,
            legal_action_ids=legal_actions,
            config=cfg,
            device=device,
        )
        return int(stats.action)
    if policy_name == "belief_ablated":
        stats = run_belief_mcts(
            model=belief,
            observation=obs,
            legal_action_ids=legal_actions,
            ego_player_id=current_player,
            config=cfg,
            ablate_belief_head=True,
            device=device,
        )
        return int(stats.action)

    stats = run_belief_mcts(
        model=belief,
        observation=obs,
        legal_action_ids=legal_actions,
        ego_player_id=current_player,
        config=cfg,
        ablate_belief_head=ablate_belief_head,
        device=device,
    )
    return int(stats.action)


def play_match(
    baseline: MuZeroNet | None,
    belief: BeliefAwareMuZeroNet,
    seat0_policy: str,
    seat1_policy: str,
    sims: int,
    max_moves: int,
    seed: int,
    ablate_belief_head: bool,
    neutralize_time_penalty: bool,
    time_penalty_per_step: float,
    device: str,
    env_mode: str,
) -> dict[str, float]:
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
        policy_name = seat0_policy if player == 0 else seat1_policy
        action = _select_action(
            policy_name=policy_name,
            baseline=baseline,
            belief=belief,
            obs=obs,
            legal_actions=env.legal_actions(),
            current_player=player,
            cfg=cfg,
            ablate_belief_head=ablate_belief_head,
            device=device,
        )
        obs, _, terminated, _ = env.step(action)
        steps += 1

    score0, score1 = env.scores[0], env.scores[1]
    if score0 < score1:
        winner = 0
    elif score1 < score0:
        winner = 1
    else:
        winner = -1
    return {
        "winner": float(winner),
        "score0": float(score0),
        "score1": float(score1),
        "score_diff_p0_minus_p1": float(score0 - score1),
        "steps": float(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline MuZero vs belief-aware MuZero in head-to-head games.")
    parser.add_argument("--baseline-checkpoint", type=str, default="")
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument(
        "--matchup",
        type=str,
        choices=["baseline_vs_belief", "belief_vs_ablated"],
        default="baseline_vs_belief",
        help="Choose which agents compete in head-to-head.",
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--max-moves-per-game", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/muzero_arch_compare")
    parser.add_argument("--env-mode", type=str, choices=["decision", "macro"], default="decision")
    parser.add_argument(
        "--ablate-belief-head",
        action="store_true",
        help="If set, disable belief-conditioning path during belief model inference for ablation.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print running head-to-head metrics every N completed games.",
    )
    parser.add_argument(
        "--neutralize-time-penalty",
        action="store_true",
        help="Add back a fixed per-step reward offset in MCTS backups to neutralize training time-penalty bias.",
    )
    parser.add_argument(
        "--time-penalty-per-step",
        type=float,
        default=0.0002,
        help="Per-step penalty magnitude to neutralize when --neutralize-time-penalty is set.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    baseline: MuZeroNet | None = None
    if args.matchup == "baseline_vs_belief":
        if not args.baseline_checkpoint:
            raise ValueError("--baseline-checkpoint is required for matchup=baseline_vs_belief")
        baseline = _load_baseline(args.baseline_checkpoint, device)
    belief = _load_belief(args.belief_checkpoint, device)

    out_dir = Path(args.output_dir)
    graph_dir = out_dir / "graphs"
    _ensure_dir(out_dir)
    _ensure_dir(graph_dir)

    rows: list[dict[str, float | str]] = []
    side_a_wins = 0
    side_b_wins = 0
    draws = 0
    log_every = max(1, int(args.log_every))
    if args.matchup == "baseline_vs_belief":
        side_a_policy = "belief"
        side_b_policy = "baseline"
    else:
        side_a_policy = "belief"
        side_b_policy = "belief_ablated"

    print(
        f"[h2h config] matchup={args.matchup} "
        f"side_a={side_a_policy} side_b={side_b_policy} "
        f"belief_head_ablated_flag={bool(args.ablate_belief_head)} "
        f"neutralize_time_penalty={bool(args.neutralize_time_penalty)} "
        f"time_penalty_per_step={float(args.time_penalty_per_step):.6f}"
    )

    for g in range(args.games):
        swap = g % 2 == 1
        seat0_policy = side_a_policy if not swap else side_b_policy
        seat1_policy = side_b_policy if not swap else side_a_policy
        result = play_match(
            baseline=baseline,
            belief=belief,
            seat0_policy=seat0_policy,
            seat1_policy=seat1_policy,
            sims=args.sims,
            max_moves=args.max_moves_per_game,
            seed=args.seed + g,
            ablate_belief_head=args.ablate_belief_head,
            neutralize_time_penalty=args.neutralize_time_penalty,
            time_penalty_per_step=args.time_penalty_per_step,
            device=args.device,
            env_mode=args.env_mode,
        )
        winner = int(result["winner"])
        if winner == -1:
            draws += 1
            winner_name = "draw"
        else:
            winner_name = seat0_policy if winner == 0 else seat1_policy
            if winner_name == side_a_policy:
                side_a_wins += 1
            else:
                side_b_wins += 1

        rows.append(
            {
                "game": str(g + 1),
                "seat0_policy": seat0_policy,
                "seat1_policy": seat1_policy,
                "winner": winner_name,
                "score0": result["score0"],
                "score1": result["score1"],
                "score_diff_p0_minus_p1": result["score_diff_p0_minus_p1"],
                "steps": result["steps"],
            }
        )

        done = g + 1
        if done % log_every == 0 or done == args.games:
            print(
                f"[h2h] {done}/{args.games} games "
                f"{side_a_policy}_wins={side_a_wins} {side_b_policy}_wins={side_b_wins} draws={draws} "
                f"{side_a_policy}_win_rate={side_a_wins / done:.3f} {side_b_policy}_win_rate={side_b_wins / done:.3f} "
                f"draw_rate={draws / done:.3f}"
            )

    total = max(1, args.games)
    summary = {
        "games": args.games,
        "matchup": args.matchup,
        "side_a_policy": side_a_policy,
        "side_b_policy": side_b_policy,
        "belief_head_ablated": bool(args.ablate_belief_head),
        "neutralize_time_penalty": bool(args.neutralize_time_penalty),
        "time_penalty_per_step": float(args.time_penalty_per_step),
        "side_a_wins": side_a_wins,
        "side_b_wins": side_b_wins,
        "draws": draws,
        "side_a_win_rate": side_a_wins / total,
        "side_b_win_rate": side_b_wins / total,
        "draw_rate": draws / total,
    }
    if args.matchup == "baseline_vs_belief":
        summary["belief_wins"] = side_a_wins
        summary["baseline_wins"] = side_b_wins
        summary["belief_win_rate"] = side_a_wins / total
        summary["baseline_win_rate"] = side_b_wins / total

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "per_game_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["game", "seat0_policy", "seat1_policy", "winner", "score0", "score1", "score_diff_p0_minus_p1", "steps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.figure(figsize=(6, 4))
    labels = [side_a_policy, side_b_policy, "draw"]
    values = [side_a_wins, side_b_wins, draws]
    plt.bar(labels, values)
    plt.title("MuZero Architecture Head-to-Head Results")
    plt.ylabel("Games")
    plt.tight_layout()
    plt.savefig(graph_dir / "head_to_head_wins.png", dpi=140)
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Comparison outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
