#!/usr/bin/env python3
"""
Collect latent state vectors and ground-truth hidden state from gameplay.

For each decision step, records:
  - h0 (raw representation vector, 512-d) from both baseline and belief models
  - ground-truth hidden state features derived from env.get_full_state()

Output: an .npz file with arrays suitable for UMAP visualization.

Usage:
  python collect_latent_states.py \
    --baseline-checkpoint runs/muzero_baseline/checkpoints/checkpoint_iter_500.pt \
    --belief-checkpoint runs/muzero_belief_superhuman_fresh_.../checkpoints/checkpoint_iter_500.pt \
    --games 150 --output runs/latent_analysis/latent_data.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from belief_muzero_mcts import run_belief_mcts
from belief_muzero_model import BeliefAwareMuZeroNet, build_default_belief_muzero_config
from muzero_mcts import MCTSConfig, run_mcts
from muzero_model import (
    MuZeroNet,
    build_default_skyjo_muzero_config,
    observation_batch_to_tensors,
)
from skyjo_decision_env import SkyjoDecisionEnv


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


def extract_hidden_features(env: SkyjoDecisionEnv, observer: int) -> dict[str, float]:
    """Derive scalar hidden-state features from the full game state."""
    state = env.base.get_full_state()
    boards = state["boards"]
    num_players = state["num_players"]
    obs_board = boards[observer]

    obs_cards = np.array(obs_board["cards"])
    obs_visible = np.array(obs_board["visible"])
    obs_removed = np.array(obs_board["removed"])

    active_mask = (~obs_removed)
    hidden_mask = active_mask & (~obs_visible)
    visible_mask = active_mask & obs_visible

    hidden_cards = obs_cards[hidden_mask]
    visible_cards = obs_cards[visible_mask]

    hidden_sum = float(hidden_cards.sum()) if len(hidden_cards) > 0 else 0.0
    hidden_count = float(len(hidden_cards))
    hidden_mean = float(hidden_cards.mean()) if len(hidden_cards) > 0 else 0.0
    num_high_hidden = float((hidden_cards >= 9).sum()) if len(hidden_cards) > 0 else 0.0
    num_negative_hidden = float((hidden_cards < 0).sum()) if len(hidden_cards) > 0 else 0.0
    visible_sum = float(visible_cards.sum()) if len(visible_cards) > 0 else 0.0
    total_true_score = float(obs_cards[active_mask].sum())

    opp_hidden_sums = []
    opp_true_scores = []
    for p in range(num_players):
        if p == observer:
            continue
        opp = boards[p]
        opp_cards = np.array(opp["cards"])
        opp_vis = np.array(opp["visible"])
        opp_rem = np.array(opp["removed"])
        opp_active = ~opp_rem
        opp_hidden = opp_active & (~opp_vis)
        opp_hidden_sums.append(float(opp_cards[opp_hidden].sum()))
        opp_true_scores.append(float(opp_cards[opp_active].sum()))

    opp_hidden_sum = float(np.mean(opp_hidden_sums)) if opp_hidden_sums else 0.0
    opp_true_score = float(np.mean(opp_true_scores)) if opp_true_scores else 0.0
    true_score_advantage = total_true_score - opp_true_score

    deck_remaining = state["deck_order_remaining"]
    deck_size = float(len(deck_remaining))
    deck_mean = float(np.mean(deck_remaining)) if deck_remaining else 0.0

    game_progress = float(state["global_step"]) / 200.0

    return {
        "hidden_sum": hidden_sum,
        "hidden_count": hidden_count,
        "hidden_mean": hidden_mean,
        "num_high_hidden": num_high_hidden,
        "num_negative_hidden": num_negative_hidden,
        "visible_sum": visible_sum,
        "total_true_score": total_true_score,
        "opp_hidden_sum": opp_hidden_sum,
        "opp_true_score": opp_true_score,
        "true_score_advantage": true_score_advantage,
        "deck_size": deck_size,
        "deck_mean": deck_mean,
        "game_progress": game_progress,
    }


def collect_game(
    baseline: MuZeroNet,
    belief: BeliefAwareMuZeroNet,
    seed: int,
    sims: int,
    max_moves: int,
    device: torch.device,
) -> dict[str, list]:
    """Play one game with the belief model, collecting latent states from both models."""
    env = SkyjoDecisionEnv(num_players=2, seed=seed, setup_mode="auto")
    obs = env.reset()
    cfg = MCTSConfig(
        num_simulations=sims,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
    )

    baseline_h0s = []
    belief_h0s = []
    belief_h0_conds = []
    features_list = []
    players = []

    terminated = False
    steps = 0
    while not terminated and steps < max_moves:
        player = int(env.current_player)

        obs_tokens = observation_batch_to_tensors(
            [obs], history_length=baseline.config.history_length, device=device
        )

        with torch.no_grad():
            bl_h0 = baseline.representation(obs_tokens)[0].cpu().numpy()

            raw_h0 = belief.representation(obs_tokens)
            ego_t = torch.tensor([player], device=device)
            cur_t = torch.tensor([player], device=device)
            npl_t = torch.tensor([env.num_players], device=device)
            cond_h0 = belief._condition_hidden(raw_h0, ego_t, cur_t, npl_t)[0].cpu().numpy()
            raw_h0_np = raw_h0[0].cpu().numpy()

        feats = extract_hidden_features(env, observer=player)

        baseline_h0s.append(bl_h0)
        belief_h0s.append(raw_h0_np)
        belief_h0_conds.append(cond_h0)
        features_list.append(feats)
        players.append(player)

        stats = run_belief_mcts(
            model=belief,
            observation=obs,
            legal_action_ids=env.legal_actions(),
            ego_player_id=player,
            config=cfg,
            ablate_belief_head=False,
            device=device,
        )
        action = int(stats.action)
        obs, _, terminated, _ = env.step(action)
        steps += 1

    score0, score1 = env.scores[0], env.scores[1]
    winner = 0 if score0 < score1 else (1 if score1 < score0 else -1)

    for f in features_list:
        f["game_winner"] = float(winner)

    return {
        "baseline_h0": baseline_h0s,
        "belief_h0": belief_h0s,
        "belief_h0_cond": belief_h0_conds,
        "features": features_list,
        "players": players,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect latent states for visualization.")
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--sims", type=int, default=50, help="MCTS sims per move (lower is fine for data collection)")
    parser.add_argument("--max-moves", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="runs/latent_analysis/latent_data.npz")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading models on {device}...")
    baseline = _load_baseline(args.baseline_checkpoint, device)
    belief = _load_belief(args.belief_checkpoint, device)

    all_baseline_h0 = []
    all_belief_h0 = []
    all_belief_h0_cond = []
    all_features = {
        "hidden_sum": [], "hidden_count": [], "hidden_mean": [],
        "num_high_hidden": [], "num_negative_hidden": [],
        "visible_sum": [], "total_true_score": [],
        "opp_hidden_sum": [], "opp_true_score": [], "true_score_advantage": [],
        "deck_size": [], "deck_mean": [], "game_progress": [],
        "game_winner": [],
    }
    all_players = []
    all_game_ids = []

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for g in range(args.games):
        if (g + 1) % 10 == 0 or g == 0:
            total_samples = len(all_players)
            print(f"Game {g + 1}/{args.games} ({total_samples} samples so far)")

        data = collect_game(
            baseline=baseline,
            belief=belief,
            seed=args.seed + g,
            sims=args.sims,
            max_moves=args.max_moves,
            device=device,
        )

        n_steps = len(data["players"])
        all_baseline_h0.extend(data["baseline_h0"])
        all_belief_h0.extend(data["belief_h0"])
        all_belief_h0_cond.extend(data["belief_h0_cond"])
        for feat_dict in data["features"]:
            for k in all_features:
                all_features[k].append(feat_dict[k])
        all_players.extend(data["players"])
        all_game_ids.extend([g] * n_steps)

    print(f"\nCollection complete: {len(all_players)} samples from {args.games} games")

    save_dict = {
        "baseline_h0": np.array(all_baseline_h0, dtype=np.float32),
        "belief_h0": np.array(all_belief_h0, dtype=np.float32),
        "belief_h0_cond": np.array(all_belief_h0_cond, dtype=np.float32),
        "player_id": np.array(all_players, dtype=np.int32),
        "game_id": np.array(all_game_ids, dtype=np.int32),
    }
    for k, v in all_features.items():
        save_dict[f"feat_{k}"] = np.array(v, dtype=np.float32)

    np.savez_compressed(str(out_path), **save_dict)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    meta = {
        "games": args.games,
        "total_samples": len(all_players),
        "baseline_checkpoint": args.baseline_checkpoint,
        "belief_checkpoint": args.belief_checkpoint,
        "sims": args.sims,
        "seed": args.seed,
        "latent_dim": int(all_baseline_h0[0].shape[0]),
        "features": list(all_features.keys()),
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
