from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from belief_muzero_mcts import run_belief_mcts
from belief_muzero_model import BeliefAwareMuZeroNet
from muzero_mcts import MCTSConfig
from muzero_model import observation_batch_to_tensors
from skyjo_env import SkyjoEnv


@dataclass(frozen=True)
class BeliefTrainConfig:
    unroll_steps: int = 5
    discount: float = 1.0
    replay_capacity_episodes: int = 1000
    batch_size: int = 64
    max_moves_per_episode: int = 2000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    winner_loss_weight: float = 0.5
    rank_loss_weight: float = 0.25
    device: str = "cpu"


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


@dataclass
class BeliefEpisodeRecord:
    steps: list[BeliefStepRecord]
    terminated: bool


def _policy_dict_to_vector(policy_target: dict[int, float], action_space_size: int) -> list[float]:
    vec = [0.0] * action_space_size
    for action, prob in policy_target.items():
        if 0 <= action < action_space_size:
            vec[action] = float(prob)
    s = sum(vec)
    if s <= 0.0:
        return [1.0 / action_space_size] * action_space_size
    return [v / s for v in vec]


def _compute_ranks(scores: list[int]) -> list[int]:
    sorted_unique = sorted(set(scores))
    score_to_rank = {score: idx + 1 for idx, score in enumerate(sorted_unique)}
    return [score_to_rank[s] for s in scores]


class BeliefReplayBuffer:
    def __init__(self, capacity_episodes: int) -> None:
        self.episodes: deque[BeliefEpisodeRecord] = deque(maxlen=capacity_episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def total_steps(self) -> int:
        return sum(len(ep.steps) for ep in self.episodes)

    def add_episode(self, episode: BeliefEpisodeRecord) -> None:
        if episode.steps:
            self.episodes.append(episode)

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int,
        action_space_size: int,
        max_players: int,
    ) -> dict[str, Any]:
        valid_episodes = [ep for ep in self.episodes if ep.steps]
        if not valid_episodes:
            raise ValueError("Belief replay buffer is empty.")

        samples: list[tuple[BeliefEpisodeRecord, int]] = []
        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            start = random.randrange(len(ep.steps))
            samples.append((ep, start))

        observations: list[dict[str, Any]] = []
        actions = torch.zeros((batch_size, unroll_steps), dtype=torch.long)
        target_policy = torch.zeros((batch_size, unroll_steps + 1, action_space_size), dtype=torch.float32)
        target_value = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32)
        target_reward = torch.zeros((batch_size, unroll_steps), dtype=torch.float32)
        winner_id = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.long)
        ego_rank = torch.ones((batch_size, unroll_steps + 1), dtype=torch.long)
        current_player = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.long)
        num_players = torch.ones((batch_size, unroll_steps + 1), dtype=torch.long)
        policy_mask = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32)
        value_mask = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32)
        reward_mask = torch.zeros((batch_size, unroll_steps), dtype=torch.float32)

        for b, (episode, start_idx) in enumerate(samples):
            observations.append(copy.deepcopy(episode.steps[start_idx].observation))

            for k in range(unroll_steps + 1):
                idx = start_idx + k
                if idx >= len(episode.steps):
                    continue
                step = episode.steps[idx]
                cp = int(step.current_player_id)
                np = int(step.num_players)
                wr = int(step.winner_id)
                rk = int(step.final_ranks[cp] if cp < len(step.final_ranks) else max_players)

                target_policy[b, k] = torch.tensor(step.policy_target, dtype=torch.float32)
                target_value[b, k] = 1.0 if cp == wr else 0.0
                winner_id[b, k] = max(0, min(max_players - 1, wr))
                ego_rank[b, k] = max(1, min(max_players, rk))
                current_player[b, k] = max(0, min(max_players - 1, cp))
                num_players[b, k] = max(1, min(max_players, np))
                policy_mask[b, k] = 1.0
                value_mask[b, k] = 1.0

                if k < unroll_steps:
                    actions[b, k] = step.action
                    target_reward[b, k] = step.reward
                    reward_mask[b, k] = 1.0

        return {
            "observations": observations,
            "actions": actions,
            "target_policy": target_policy,
            "target_value": target_value,
            "target_reward": target_reward,
            "winner_id": winner_id,
            "ego_rank": ego_rank,
            "current_player": current_player,
            "num_players": num_players,
            "policy_mask": policy_mask,
            "value_mask": value_mask,
            "reward_mask": reward_mask,
        }


def generate_belief_self_play_episode(
    model: BeliefAwareMuZeroNet,
    mcts_config: MCTSConfig,
    env_factory: Callable[[], Any] | None = None,
    max_moves: int = 2000,
    device: str = "cpu",
) -> BeliefEpisodeRecord:
    if env_factory is None:
        env_factory = lambda: SkyjoEnv(num_players=2, seed=random.randint(0, 10_000_000), setup_mode="auto")

    env = env_factory()
    obs = env.reset()
    action_space_size = model.config.action_space_size
    pending_steps: list[dict[str, Any]] = []
    terminated = False
    moves = 0

    while not terminated and moves < max_moves:
        actor = int(env.current_player)
        legal = env.legal_actions()
        stats = run_belief_mcts(
            model=model,
            observation=obs,
            legal_action_ids=legal,
            ego_player_id=actor,
            config=mcts_config,
            device=device,
        )
        next_obs, rewards, terminated, _ = env.step(stats.action)
        pending_steps.append(
            {
                "observation": copy.deepcopy(obs),
                "action": int(stats.action),
                "reward": float(rewards.get(f"player_{actor}", 0.0)),
                "policy_target": _policy_dict_to_vector(stats.policy_target, action_space_size),
                "root_value": float(stats.root_value),
                "current_player_id": actor,
                "num_players": env.num_players,
            }
        )
        obs = next_obs
        moves += 1

    scores = list(env.scores)
    min_score = min(scores) if scores else 0
    winners = [i for i, s in enumerate(scores) if s == min_score]
    winner_id = int(winners[0] if winners else 0)
    final_ranks = _compute_ranks(scores if scores else [0, 1])

    steps: list[BeliefStepRecord] = []
    for s in pending_steps:
        steps.append(
            BeliefStepRecord(
                observation=s["observation"],
                action=s["action"],
                reward=s["reward"],
                policy_target=s["policy_target"],
                root_value=s["root_value"],
                current_player_id=s["current_player_id"],
                num_players=s["num_players"],
                winner_id=winner_id,
                final_ranks=final_ranks,
            )
        )
    return BeliefEpisodeRecord(steps=steps, terminated=terminated)


def create_belief_optimizer(model: BeliefAwareMuZeroNet, config: BeliefTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def train_belief_step(
    model: BeliefAwareMuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: BeliefReplayBuffer,
    config: BeliefTrainConfig,
) -> dict[str, float]:
    if replay_buffer.total_steps() < config.batch_size:
        raise ValueError("Not enough replay samples for belief training step.")

    device = torch.device(config.device)
    model.train()
    batch = replay_buffer.sample_batch(
        batch_size=config.batch_size,
        unroll_steps=config.unroll_steps,
        action_space_size=model.config.action_space_size,
        max_players=model.config.max_players,
    )

    obs_tokens = observation_batch_to_tensors(
        batch["observations"],
        history_length=model.config.history_length,
        device=device,
    )
    actions = batch["actions"].to(device)
    target_policy = batch["target_policy"].to(device)
    target_value = batch["target_value"].to(device)
    target_reward = batch["target_reward"].to(device)
    winner_id = batch["winner_id"].to(device)
    ego_rank = batch["ego_rank"].to(device)
    current_player = batch["current_player"].to(device)
    num_players = batch["num_players"].to(device)
    policy_mask = batch["policy_mask"].to(device)
    value_mask = batch["value_mask"].to(device)
    reward_mask = batch["reward_mask"].to(device)

    init = model.initial_inference(
        obs_tokens,
        ego_player_id=current_player[:, 0],
        current_player_id=current_player[:, 0],
        num_players=num_players[:, 0],
    )
    hidden = init.hidden_state

    pred_policy = [init.policy_logits]
    pred_value = [init.value_logits]
    pred_winner = [init.winner_logits]
    pred_rank = [init.rank_logits]
    pred_reward: list[torch.Tensor] = []

    for k in range(config.unroll_steps):
        rec = model.recurrent_inference(
            hidden_state=hidden,
            action=actions[:, k],
            ego_player_id=current_player[:, k + 1],
            current_player_id=current_player[:, k + 1],
            num_players=num_players[:, k + 1],
        )
        hidden = rec.hidden_state
        pred_policy.append(rec.policy_logits)
        pred_value.append(rec.value_logits)
        pred_winner.append(rec.winner_logits)
        pred_rank.append(rec.rank_logits)
        pred_reward.append(rec.reward_logits)

    pred_policy_logits = torch.stack(pred_policy, dim=1)
    pred_value_logits = torch.stack(pred_value, dim=1)
    pred_winner_logits = torch.stack(pred_winner, dim=1)
    pred_rank_logits = torch.stack(pred_rank, dim=1)
    pred_reward_logits = torch.stack(pred_reward, dim=1)

    target_value_dist = model.value_support.scalar_to_logits_target(target_value)
    target_reward_dist = model.reward_support.scalar_to_logits_target(target_reward)

    logp_policy = F.log_softmax(pred_policy_logits, dim=-1)
    policy_loss_step = -(target_policy * logp_policy).sum(dim=-1)
    policy_loss = (policy_loss_step * policy_mask).sum() / policy_mask.sum().clamp_min(1.0)

    logp_value = F.log_softmax(pred_value_logits, dim=-1)
    value_loss_step = -(target_value_dist * logp_value).sum(dim=-1)
    value_loss = (value_loss_step * value_mask).sum() / value_mask.sum().clamp_min(1.0)

    logp_reward = F.log_softmax(pred_reward_logits, dim=-1)
    reward_loss_step = -(target_reward_dist * logp_reward).sum(dim=-1)
    reward_loss = (reward_loss_step * reward_mask).sum() / reward_mask.sum().clamp_min(1.0)

    winner_flat = winner_id.view(-1)
    winner_logits_flat = pred_winner_logits.view(-1, pred_winner_logits.size(-1))
    winner_mask = value_mask.view(-1)
    winner_loss_raw = F.cross_entropy(winner_logits_flat, winner_flat, reduction="none")
    winner_loss = (winner_loss_raw * winner_mask).sum() / winner_mask.sum().clamp_min(1.0)

    rank_targets = (ego_rank - 1).clamp_min(0).clamp_max(model.config.max_players - 1).view(-1)
    rank_logits_flat = pred_rank_logits.view(-1, pred_rank_logits.size(-1))
    rank_loss_raw = F.cross_entropy(rank_logits_flat, rank_targets, reduction="none")
    rank_loss = (rank_loss_raw * winner_mask).sum() / winner_mask.sum().clamp_min(1.0)

    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.reward_loss_weight * reward_loss
        + config.winner_loss_weight * winner_loss
        + config.rank_loss_weight * rank_loss
    )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()

    return {
        "loss_total": float(total_loss.detach().cpu().item()),
        "loss_policy": float(policy_loss.detach().cpu().item()),
        "loss_value": float(value_loss.detach().cpu().item()),
        "loss_reward": float(reward_loss.detach().cpu().item()),
        "loss_winner": float(winner_loss.detach().cpu().item()),
        "loss_rank": float(rank_loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
    }
