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
    baseline: MuZeroNet,
    belief: BeliefAwareMuZeroNet,
    obs: dict,
    legal_actions: list[int],
    current_player: int,
    cfg: MCTSConfig,
    device: str,
) -> int:
    if policy_name == "baseline":
        stats = run_mcts(
            model=baseline,
            observation=obs,
            legal_action_ids=legal_actions,
            config=cfg,
            device=device,
        )
        return int(stats.action)

    stats = run_belief_mcts(
        model=belief,
        observation=obs,
        legal_action_ids=legal_actions,
        ego_player_id=current_player,
        config=cfg,
        device=device,
    )
    return int(stats.action)


def play_match(
    baseline: MuZeroNet,
    belief: BeliefAwareMuZeroNet,
    seat0_policy: str,
    seat1_policy: str,
    sims: int,
    max_moves: int,
    seed: int,
    device: str,
    env_mode: str,
) -> dict[str, float]:
    if env_mode == "decision":
        env = SkyjoDecisionEnv(num_players=2, seed=seed, setup_mode="auto")
    else:
        env = SkyjoEnv(num_players=2, seed=seed, setup_mode="auto")
    obs = env.reset()
    cfg = MCTSConfig(num_simulations=sims, temperature=1e-8, add_exploration_noise=False, root_exploration_fraction=0.0)

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
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--max-moves-per-game", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/muzero_arch_compare")
    parser.add_argument("--env-mode", type=str, choices=["decision", "macro"], default="decision")
    args = parser.parse_args()

    device = torch.device(args.device)
    baseline = _load_baseline(args.baseline_checkpoint, device)
    belief = _load_belief(args.belief_checkpoint, device)

    out_dir = Path(args.output_dir)
    graph_dir = out_dir / "graphs"
    _ensure_dir(out_dir)
    _ensure_dir(graph_dir)

    rows: list[dict[str, float | str]] = []
    belief_wins = 0
    baseline_wins = 0
    draws = 0

    for g in range(args.games):
        swap = g % 2 == 1
        seat0_policy = "belief" if not swap else "baseline"
        seat1_policy = "baseline" if not swap else "belief"
        result = play_match(
            baseline=baseline,
            belief=belief,
            seat0_policy=seat0_policy,
            seat1_policy=seat1_policy,
            sims=args.sims,
            max_moves=args.max_moves_per_game,
            seed=args.seed + g,
            device=args.device,
            env_mode=args.env_mode,
        )
        winner = int(result["winner"])
        if winner == -1:
            draws += 1
            winner_name = "draw"
        else:
            winner_name = seat0_policy if winner == 0 else seat1_policy
            if winner_name == "belief":
                belief_wins += 1
            else:
                baseline_wins += 1

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

    total = max(1, args.games)
    summary = {
        "games": args.games,
        "belief_wins": belief_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "belief_win_rate": belief_wins / total,
        "baseline_win_rate": baseline_wins / total,
        "draw_rate": draws / total,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "per_game_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["game", "seat0_policy", "seat1_policy", "winner", "score0", "score1", "score_diff_p0_minus_p1", "steps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.figure(figsize=(6, 4))
    labels = ["belief", "baseline", "draw"]
    values = [belief_wins, baseline_wins, draws]
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
