"""
Training utilities for both baseline and belief-aware MuZero on the auction env.

Provides replay buffer, episode generation, and single-step training for each
variant. Mirrors the Skyjo muzero_train / belief_muzero_train structure.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from auction_env import AuctionEnv, NUM_PLAYERS
from auction_mcts import run_auction_belief_mcts, run_auction_mcts
from auction_model import (
    AuctionBeliefMuZeroNet,
    AuctionConfig,
    AuctionMuZeroNet,
    auction_observation_batch_to_tensors,
)
from muzero_mcts import MCTSConfig

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    observation: dict[str, Any]
    action: int
    reward: float
    policy_target: list[float]
    root_value: float
    current_player_id: int


@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    terminated: bool


@dataclass
class BeliefStepRecord:
    observation: dict[str, Any]
    action: int
    reward: float
    policy_target: list[float]
    root_value: float
    current_player_id: int
    num_players: int
    winner_id: int
    final_ranks: list[int]
    final_score_utility: float


@dataclass
class BeliefEpisodeRecord:
    steps: list[BeliefStepRecord]
    terminated: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy_dict_to_vector(policy: dict[int, float], size: int) -> list[float]:
    vec = [0.0] * size
    for a, p in policy.items():
        if 0 <= a < size:
            vec[a] = p
    s = sum(vec)
    return [v / s for v in vec] if s > 0 else [1.0 / size] * size


def _one_hot_policy(action: int, size: int) -> list[float]:
    vec = [0.0] * size
    if 0 <= action < size:
        vec[action] = 1.0
    return vec


def _sample_slots_and_actions(env: AuctionEnv, action_size: int) -> tuple[list[int], list[float]]:
    num_raise_samples = max(1, action_size - 1)
    actions = env.sample_action_candidates(num_raise_samples=num_raise_samples)
    if len(actions) < action_size:
        # Pad by repeating the largest legal raise to preserve fixed model action size.
        pad_value = actions[-2] if len(actions) >= 2 else actions[0]
        while len(actions) < action_size:
            actions.insert(-1, pad_value)
    elif len(actions) > action_size:
        # Keep pass in the final slot.
        actions = actions[: action_size - 1] + [actions[-1]]
    slots = list(range(action_size))
    return slots, actions


def _slot_for_action_amount(action_amount: float, action_amounts: list[float]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for i, a in enumerate(action_amounts):
        d = abs(float(a) - float(action_amount))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _compute_ranks(profits: list[float]) -> list[int]:
    sorted_unique = sorted(set(profits), reverse=True)
    score_to_rank = {s: i + 1 for i, s in enumerate(sorted_unique)}
    return [score_to_rank[p] for p in profits]


def _compute_n_step_return(steps: list, start: int, td_steps: int, discount: float) -> float:
    accum, gamma = 0.0, 1.0
    for i in range(td_steps):
        t = start + i
        if t >= len(steps):
            break
        accum += gamma * steps[t].reward
        gamma *= discount
    bootstrap = start + td_steps
    if bootstrap < len(steps):
        accum += gamma * steps[bootstrap].root_value
    return accum


# ---------------------------------------------------------------------------
# Baseline replay & training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineTrainConfig:
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997
    batch_size: int = 64
    max_moves_per_episode: int = 500
    replay_capacity_episodes: int = 4000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.5
    reward_loss_weight: float = 1.0
    truncation_penalty: float = 0.5
    device: str = "cpu"


class BaselineReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.episodes: deque[EpisodeRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def total_steps(self) -> int:
        return sum(len(e.steps) for e in self.episodes)

    def add_episode(self, ep: EpisodeRecord) -> None:
        if ep.steps:
            self.episodes.append(ep)

    def sample_batch(
        self, batch_size: int, unroll: int, action_size: int,
        td_steps: int = 10, discount: float = 1.0,
    ) -> dict[str, Any]:
        valid = [e for e in self.episodes if e.steps]
        obs_list: list[dict[str, Any]] = []
        actions = torch.zeros((batch_size, unroll), dtype=torch.long)
        tgt_policy = torch.zeros((batch_size, unroll + 1, action_size))
        tgt_value = torch.zeros((batch_size, unroll + 1))
        tgt_reward = torch.zeros((batch_size, unroll))
        p_mask = torch.zeros((batch_size, unroll + 1))
        v_mask = torch.zeros((batch_size, unroll + 1))
        r_mask = torch.zeros((batch_size, unroll))

        for b in range(batch_size):
            ep = random.choice(valid)
            start = random.randrange(len(ep.steps))
            obs_list.append(copy.deepcopy(ep.steps[start].observation))
            for k in range(unroll + 1):
                idx = start + k
                if idx >= len(ep.steps):
                    continue
                s = ep.steps[idx]
                tgt_policy[b, k] = torch.tensor(s.policy_target)
                tgt_value[b, k] = _compute_n_step_return(ep.steps, idx, td_steps, discount)
                p_mask[b, k] = 1.0
                v_mask[b, k] = 1.0
                if k < unroll:
                    actions[b, k] = s.action
                    tgt_reward[b, k] = s.reward
                    r_mask[b, k] = 1.0

        return {
            "observations": obs_list, "actions": actions,
            "target_policy": tgt_policy, "target_value": tgt_value,
            "target_reward": tgt_reward, "policy_mask": p_mask,
            "value_mask": v_mask, "reward_mask": r_mask,
        }


def generate_baseline_episode(
    model: AuctionMuZeroNet,
    mcts_config: MCTSConfig,
    opponent_model: AuctionMuZeroNet | None = None,
    learner_player_id: int = 0,
    env_factory: Callable[[], AuctionEnv] | None = None,
    max_moves: int = 500,
    device: str = "cpu",
    truncation_penalty: float = 0.5,
    opponent_action_selector: Callable | None = None,
) -> EpisodeRecord:
    env = (env_factory or (lambda: AuctionEnv(seed=random.randint(0, 10_000_000))))()
    obs = env.reset()
    action_size = model.config.action_space_size
    pending: list[dict[str, Any]] = []
    terminated, moves = False, 0

    while not terminated and moves < max_moves:
        actor = int(env.current_player)
        legal, action_amounts = _sample_slots_and_actions(env, action_size)
        is_opp = actor != learner_player_id

        if is_opp and opponent_action_selector is not None:
            bot_action = float(opponent_action_selector(obs, action_amounts, actor))
            chosen = _slot_for_action_amount(bot_action, action_amounts)
            if chosen not in legal:
                chosen = legal[0]
            policy_target = _one_hot_policy(chosen, action_size)
            dev = torch.device(device)
            tokens = auction_observation_batch_to_tensors([obs], config=model.config, device=dev)
            with torch.no_grad():
                init = model.initial_inference(tokens)
                root_value = model.value_support.logits_to_scalar(init.value_logits[0]).item()
        else:
            acting = model
            if opponent_model is not None and is_opp:
                acting = opponent_model
            stats = run_auction_mcts(acting, obs, legal, mcts_config, device, mcts_inference_autocast=True)
            chosen = stats.action
            policy_target = _policy_dict_to_vector(stats.policy_target, action_size)
            root_value = stats.root_value

        chosen_amount = float(action_amounts[chosen])
        next_obs, rewards, terminated, _ = env.step(chosen_amount)
        r = float(rewards.get(f"player_{actor}", 0.0))
        pending.append({
            "observation": copy.deepcopy(obs), "action": chosen, "reward": r,
            "policy_target": policy_target, "root_value": root_value,
            "current_player_id": actor,
        })
        obs = next_obs
        moves += 1

    if not terminated and pending:
        pending[-1]["reward"] -= truncation_penalty

    steps = [StepRecord(**s) for s in pending]
    return EpisodeRecord(steps=steps, terminated=terminated)


def create_baseline_optimizer(model: AuctionMuZeroNet, cfg: BaselineTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def train_baseline_step(
    model: AuctionMuZeroNet, optimizer: torch.optim.Optimizer,
    replay: BaselineReplayBuffer, cfg: BaselineTrainConfig,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    dev = torch.device(cfg.device)
    use_amp = dev.type == "cuda" and grad_scaler is not None
    model.train()
    batch = replay.sample_batch(cfg.batch_size, cfg.unroll_steps, model.config.action_space_size, cfg.td_steps, cfg.discount)

    tokens = auction_observation_batch_to_tensors(batch["observations"], model.config, dev)
    actions = batch["actions"].to(dev)
    tgt_policy = batch["target_policy"].to(dev)
    tgt_value = batch["target_value"].to(dev)
    tgt_reward = batch["target_reward"].to(dev)
    p_mask = batch["policy_mask"].to(dev)
    v_mask = batch["value_mask"].to(dev)
    r_mask = batch["reward_mask"].to(dev)

    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        init = model.initial_inference(tokens)
        hidden = init.hidden_state
        pred_p = [init.policy_logits]
        pred_v = [init.value_logits]
        pred_r: list[torch.Tensor] = []
        for k in range(cfg.unroll_steps):
            rec = model.recurrent_inference(hidden, actions[:, k])
            hidden = rec.hidden_state
            pred_p.append(rec.policy_logits)
            pred_v.append(rec.value_logits)
            pred_r.append(rec.reward_logits)

        pred_p_t = torch.stack(pred_p, 1)
        pred_v_t = torch.stack(pred_v, 1)
        pred_r_t = torch.stack(pred_r, 1)

        tgt_v_dist = model.value_support.scalar_to_logits_target(tgt_value)
        tgt_r_dist = model.reward_support.scalar_to_logits_target(tgt_reward)

        logp_p = F.log_softmax(pred_p_t, -1)
        policy_loss = (-(tgt_policy * logp_p).sum(-1) * p_mask).sum() / p_mask.sum().clamp_min(1)
        logp_v = F.log_softmax(pred_v_t, -1)
        value_loss = (-(tgt_v_dist * logp_v).sum(-1) * v_mask).sum() / v_mask.sum().clamp_min(1)
        logp_r = F.log_softmax(pred_r_t, -1)
        reward_loss = (-(tgt_r_dist * logp_r).sum(-1) * r_mask).sum() / r_mask.sum().clamp_min(1)

        total = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss + cfg.reward_loss_weight * reward_loss

    optimizer.zero_grad(set_to_none=True)
    if grad_scaler is not None:
        grad_scaler.scale(total).backward()
        grad_scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

    return {
        "loss_total": total.item(), "loss_policy": policy_loss.item(),
        "loss_value": value_loss.item(), "loss_reward": reward_loss.item(),
        "grad_norm": float(gn.item() if isinstance(gn, torch.Tensor) else gn),
    }


# ---------------------------------------------------------------------------
# Belief replay & training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeliefTrainConfig:
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997
    batch_size: int = 64
    max_moves_per_episode: int = 500
    replay_capacity_episodes: int = 4000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.5
    reward_loss_weight: float = 1.0
    winner_loss_weight: float = 0.1
    rank_loss_weight: float = 0.1
    device: str = "cpu"


class BeliefReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.episodes: deque[BeliefEpisodeRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def total_steps(self) -> int:
        return sum(len(e.steps) for e in self.episodes)

    def add_episode(self, ep: BeliefEpisodeRecord) -> None:
        if ep.steps:
            self.episodes.append(ep)

    def sample_batch(
        self, batch_size: int, unroll: int, action_size: int,
        max_players: int, td_steps: int = 10, discount: float = 1.0,
    ) -> dict[str, Any]:
        valid = [e for e in self.episodes if e.steps]
        obs_list: list[dict[str, Any]] = []
        actions = torch.zeros((batch_size, unroll), dtype=torch.long)
        tgt_policy = torch.zeros((batch_size, unroll + 1, action_size))
        tgt_value = torch.zeros((batch_size, unroll + 1))
        tgt_reward = torch.zeros((batch_size, unroll))
        winner_id = torch.zeros((batch_size, unroll + 1), dtype=torch.long)
        ego_rank = torch.ones((batch_size, unroll + 1), dtype=torch.long)
        current_player = torch.zeros((batch_size, unroll + 1), dtype=torch.long)
        num_players = torch.ones((batch_size, unroll + 1), dtype=torch.long)
        p_mask = torch.zeros((batch_size, unroll + 1))
        v_mask = torch.zeros((batch_size, unroll + 1))
        r_mask = torch.zeros((batch_size, unroll))

        for b in range(batch_size):
            ep = random.choice(valid)
            start = random.randrange(len(ep.steps))
            obs_list.append(copy.deepcopy(ep.steps[start].observation))
            for k in range(unroll + 1):
                idx = start + k
                if idx >= len(ep.steps):
                    continue
                s = ep.steps[idx]
                cp = int(s.current_player_id)
                np_ = int(s.num_players)
                wr = int(s.winner_id)
                rk = int(s.final_ranks[cp] if cp < len(s.final_ranks) else max_players)
                tgt_policy[b, k] = torch.tensor(s.policy_target)
                tgt_value[b, k] = _compute_n_step_return(ep.steps, idx, td_steps, discount)
                winner_id[b, k] = max(0, min(max_players - 1, wr))
                ego_rank[b, k] = max(1, min(max_players, rk))
                current_player[b, k] = max(0, min(max_players - 1, cp))
                num_players[b, k] = max(1, min(max_players, np_))
                p_mask[b, k] = 1.0
                v_mask[b, k] = 1.0
                if k < unroll:
                    actions[b, k] = s.action
                    tgt_reward[b, k] = s.reward
                    r_mask[b, k] = 1.0

        return {
            "observations": obs_list, "actions": actions,
            "target_policy": tgt_policy, "target_value": tgt_value,
            "target_reward": tgt_reward, "winner_id": winner_id,
            "ego_rank": ego_rank, "current_player": current_player,
            "num_players": num_players, "policy_mask": p_mask,
            "value_mask": v_mask, "reward_mask": r_mask,
        }


def generate_belief_episode(
    model: AuctionBeliefMuZeroNet,
    mcts_config: MCTSConfig,
    opponent_model: AuctionBeliefMuZeroNet | None = None,
    learner_player_id: int = 0,
    env_factory: Callable[[], AuctionEnv] | None = None,
    max_moves: int = 500,
    device: str = "cpu",
    opponent_action_selector: Callable | None = None,
) -> BeliefEpisodeRecord:
    env = (env_factory or (lambda: AuctionEnv(seed=random.randint(0, 10_000_000))))()
    obs = env.reset()
    action_size = model.config.action_space_size
    pending: list[dict[str, Any]] = []
    terminated, moves = False, 0

    while not terminated and moves < max_moves:
        actor = int(env.current_player)
        legal, action_amounts = _sample_slots_and_actions(env, action_size)
        is_opp = actor != learner_player_id

        if is_opp and opponent_action_selector is not None:
            bot_action = float(opponent_action_selector(obs, action_amounts, actor))
            chosen = _slot_for_action_amount(bot_action, action_amounts)
            if chosen not in legal:
                chosen = legal[0]
            policy_target = _one_hot_policy(chosen, action_size)
            dev = torch.device(device)
            tokens = auction_observation_batch_to_tensors([obs], config=model.config, device=dev)
            ego_t = torch.tensor([actor], dtype=torch.long, device=dev)
            cur_t = torch.tensor([actor], dtype=torch.long, device=dev)
            np_t = torch.tensor([NUM_PLAYERS], dtype=torch.long, device=dev)
            with torch.no_grad():
                init = model.initial_inference(tokens, ego_t, cur_t, np_t)
                root_value = model.value_support.logits_to_scalar(init.value_logits[0]).item()
        else:
            acting = model
            if opponent_model is not None and is_opp:
                acting = opponent_model
            stats = run_auction_belief_mcts(acting, obs, legal, actor, mcts_config, device, mcts_inference_autocast=True)
            chosen = stats.action
            policy_target = _policy_dict_to_vector(stats.policy_target, action_size)
            root_value = stats.root_value

        chosen_amount = float(action_amounts[chosen])
        next_obs, rewards, terminated, _ = env.step(chosen_amount)
        r = float(rewards.get(f"player_{actor}", 0.0))
        pending.append({
            "observation": copy.deepcopy(obs), "action": chosen, "reward": r,
            "policy_target": policy_target, "root_value": root_value,
            "current_player_id": actor, "num_players": NUM_PLAYERS,
        })
        obs = next_obs
        moves += 1

    profits = list(env.total_profit)
    max_profit = max(profits) if profits else 0
    winners = [i for i, p in enumerate(profits) if p == max_profit]
    winner_id = winners[0] if winners else 0
    final_ranks = _compute_ranks(profits if profits else [0.0, 0.0])

    steps: list[BeliefStepRecord] = []
    for s in pending:
        cp = int(s["current_player_id"])
        own = profits[cp] if cp < len(profits) else 0.0
        others = [profits[i] for i in range(len(profits)) if i != cp]
        opp_mean = sum(others) / max(1, len(others))
        steps.append(BeliefStepRecord(
            observation=s["observation"], action=s["action"], reward=s["reward"],
            policy_target=s["policy_target"], root_value=s["root_value"],
            current_player_id=cp, num_players=s["num_players"],
            winner_id=winner_id, final_ranks=final_ranks,
            final_score_utility=own - opp_mean,
        ))
    return BeliefEpisodeRecord(steps=steps, terminated=terminated)


def create_belief_optimizer(model: AuctionBeliefMuZeroNet, cfg: BeliefTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def train_belief_step(
    model: AuctionBeliefMuZeroNet, optimizer: torch.optim.Optimizer,
    replay: BeliefReplayBuffer, cfg: BeliefTrainConfig,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    dev = torch.device(cfg.device)
    use_amp = dev.type == "cuda" and grad_scaler is not None
    model.train()
    batch = replay.sample_batch(
        cfg.batch_size, cfg.unroll_steps, model.config.action_space_size,
        model.config.max_players, cfg.td_steps, cfg.discount,
    )

    tokens = auction_observation_batch_to_tensors(batch["observations"], model.config, dev)
    actions = batch["actions"].to(dev)
    tgt_policy = batch["target_policy"].to(dev)
    tgt_value = batch["target_value"].to(dev)
    tgt_reward = batch["target_reward"].to(dev)
    winner = batch["winner_id"].to(dev)
    ego_rank = batch["ego_rank"].to(dev)
    cur_p = batch["current_player"].to(dev)
    n_p = batch["num_players"].to(dev)
    p_mask = batch["policy_mask"].to(dev)
    v_mask = batch["value_mask"].to(dev)
    r_mask = batch["reward_mask"].to(dev)

    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        init = model.initial_inference(tokens, cur_p[:, 0], cur_p[:, 0], n_p[:, 0])
        hidden = init.hidden_state
        pp = [init.policy_logits]
        pv = [init.value_logits]
        pw = [init.winner_logits]
        pr = [init.rank_logits]
        pred_rw: list[torch.Tensor] = []

        for k in range(cfg.unroll_steps):
            rec = model.recurrent_inference(hidden, actions[:, k], cur_p[:, k + 1], cur_p[:, k + 1], n_p[:, k + 1])
            hidden = rec.hidden_state
            pp.append(rec.policy_logits)
            pv.append(rec.value_logits)
            pw.append(rec.winner_logits)
            pr.append(rec.rank_logits)
            pred_rw.append(rec.reward_logits)

        pp_t = torch.stack(pp, 1)
        pv_t = torch.stack(pv, 1)
        pw_t = torch.stack(pw, 1)
        pr_t = torch.stack(pr, 1)
        prw_t = torch.stack(pred_rw, 1)

        tgt_v_dist = model.value_support.scalar_to_logits_target(tgt_value)
        tgt_r_dist = model.reward_support.scalar_to_logits_target(tgt_reward)

        logp_p = F.log_softmax(pp_t, -1)
        policy_loss = (-(tgt_policy * logp_p).sum(-1) * p_mask).sum() / p_mask.sum().clamp_min(1)
        logp_v = F.log_softmax(pv_t, -1)
        value_loss = (-(tgt_v_dist * logp_v).sum(-1) * v_mask).sum() / v_mask.sum().clamp_min(1)
        logp_r = F.log_softmax(prw_t, -1)
        reward_loss = (-(tgt_r_dist * logp_r).sum(-1) * r_mask).sum() / r_mask.sum().clamp_min(1)

        w_flat = winner.view(-1)
        w_logits = pw_t.view(-1, pw_t.size(-1))
        w_mask = v_mask.view(-1)
        winner_loss = (F.cross_entropy(w_logits, w_flat, reduction="none") * w_mask).sum() / w_mask.sum().clamp_min(1)

        rank_tgt = (ego_rank - 1).clamp(0, model.config.max_players - 1).view(-1)
        r_logits = pr_t.view(-1, pr_t.size(-1))
        rank_loss = (F.cross_entropy(r_logits, rank_tgt, reduction="none") * w_mask).sum() / w_mask.sum().clamp_min(1)

        total = (
            cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss
            + cfg.reward_loss_weight * reward_loss + cfg.winner_loss_weight * winner_loss
            + cfg.rank_loss_weight * rank_loss
        )

    optimizer.zero_grad(set_to_none=True)
    if grad_scaler is not None:
        grad_scaler.scale(total).backward()
        grad_scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

    return {
        "loss_total": total.item(), "loss_policy": policy_loss.item(),
        "loss_value": value_loss.item(), "loss_reward": reward_loss.item(),
        "loss_winner": winner_loss.item(), "loss_rank": rank_loss.item(),
        "grad_norm": float(gn.item() if isinstance(gn, torch.Tensor) else gn),
    }
