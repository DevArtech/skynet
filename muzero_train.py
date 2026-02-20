from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from muzero_mcts import MCTSConfig, run_mcts
from muzero_model import MuZeroNet, observation_batch_to_tensors
from skyjo_env import SkyjoEnv


@dataclass(frozen=True)
class MuZeroTrainConfig:
    unroll_steps: int = 5
    td_steps: int = 5
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
    device: str = "cpu"


@dataclass
class StepRecord:
    observation: dict[str, Any]
    action: int
    reward: float
    policy_target: list[float]
    root_value: float


@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity_episodes: int) -> None:
        self.capacity_episodes = capacity_episodes
        self.episodes: deque[EpisodeRecord] = deque(maxlen=capacity_episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: EpisodeRecord) -> None:
        if episode.steps:
            self.episodes.append(episode)

    def total_steps(self) -> int:
        return sum(len(ep.steps) for ep in self.episodes)

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
        action_space_size: int,
    ) -> dict[str, Any]:
        if not self.episodes:
            raise ValueError("Replay buffer is empty.")

        valid_episodes = [ep for ep in self.episodes if ep.steps]
        if not valid_episodes:
            raise ValueError("Replay buffer has no non-empty episodes.")

        samples: list[tuple[EpisodeRecord, int]] = []
        for _ in range(batch_size):
            episode = random.choice(valid_episodes)
            start_idx = random.randrange(len(episode.steps))
            samples.append((episode, start_idx))

        observations: list[dict[str, Any]] = []
        actions = torch.zeros((batch_size, unroll_steps), dtype=torch.long)
        target_policy = torch.zeros((batch_size, unroll_steps + 1, action_space_size), dtype=torch.float32)
        target_value = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32)
        target_reward = torch.zeros((batch_size, unroll_steps), dtype=torch.float32)
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
                target_policy[b, k] = torch.tensor(step.policy_target, dtype=torch.float32)
                target_value[b, k] = self._compute_value_target(
                    episode=episode,
                    start_index=idx,
                    td_steps=td_steps,
                    discount=discount,
                )
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
            "policy_mask": policy_mask,
            "value_mask": value_mask,
            "reward_mask": reward_mask,
        }

    @staticmethod
    def _compute_value_target(
        episode: EpisodeRecord,
        start_index: int,
        td_steps: int,
        discount: float,
    ) -> float:
        value = 0.0
        discount_acc = 1.0
        end_index = min(start_index + td_steps, len(episode.steps))
        for i in range(start_index, end_index):
            value += discount_acc * episode.steps[i].reward
            discount_acc *= discount

        bootstrap_index = start_index + td_steps
        if bootstrap_index < len(episode.steps):
            value += discount_acc * episode.steps[bootstrap_index].root_value
        return float(value)


def _policy_dict_to_vector(policy_target: dict[int, float], action_space_size: int) -> list[float]:
    policy = [0.0] * action_space_size
    for action, prob in policy_target.items():
        if 0 <= action < action_space_size:
            policy[action] = float(prob)
    total = sum(policy)
    if total <= 0.0:
        uniform = 1.0 / action_space_size
        return [uniform] * action_space_size
    return [p / total for p in policy]


def generate_self_play_episode(
    model: MuZeroNet,
    mcts_config: MCTSConfig,
    env_factory: Callable[[], Any] | None = None,
    max_moves: int = 2000,
    device: str = "cpu",
) -> EpisodeRecord:
    if env_factory is None:
        env_factory = lambda: SkyjoEnv(num_players=2, seed=random.randint(0, 10_000_000), setup_mode="auto")

    env = env_factory()
    observation = env.reset()
    steps: list[StepRecord] = []
    terminated = False
    move_count = 0
    action_space_size = model.config.action_space_size

    while not terminated and move_count < max_moves:
        actor = env.current_player
        legal_actions = env.legal_actions()
        stats = run_mcts(
            model=model,
            observation=observation,
            legal_action_ids=legal_actions,
            config=mcts_config,
            device=device,
        )
        next_observation, rewards, terminated, _ = env.step(stats.action)
        reward = float(rewards.get(f"player_{actor}", 0.0))

        steps.append(
            StepRecord(
                observation=copy.deepcopy(observation),
                action=int(stats.action),
                reward=reward,
                policy_target=_policy_dict_to_vector(stats.policy_target, action_space_size=action_space_size),
                root_value=float(stats.root_value),
            )
        )

        observation = next_observation
        move_count += 1

    return EpisodeRecord(steps=steps, terminated=terminated)


def create_optimizer(model: MuZeroNet, config: MuZeroTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_step(
    model: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: MuZeroTrainConfig,
) -> dict[str, float]:
    if replay_buffer.total_steps() < config.batch_size:
        raise ValueError("Not enough replay samples for a training step.")

    model.train()
    batch = replay_buffer.sample_batch(
        batch_size=config.batch_size,
        unroll_steps=config.unroll_steps,
        td_steps=config.td_steps,
        discount=config.discount,
        action_space_size=model.config.action_space_size,
    )
    device = torch.device(config.device)
    obs_tokens = observation_batch_to_tensors(
        batch["observations"],
        history_length=model.config.history_length,
        device=device,
    )
    actions = batch["actions"].to(device)
    target_policy = batch["target_policy"].to(device)
    target_value = batch["target_value"].to(device)
    target_reward = batch["target_reward"].to(device)
    policy_mask = batch["policy_mask"].to(device)
    value_mask = batch["value_mask"].to(device)
    reward_mask = batch["reward_mask"].to(device)

    initial = model.initial_inference(obs_tokens)
    initial_policy_logits = initial.policy_logits.unsqueeze(1)
    initial_value_logits = initial.value_logits.unsqueeze(1)

    recurrent_policy_logits, recurrent_value_logits, recurrent_reward_logits = model.unroll(
        hidden_state=initial.hidden_state,
        actions=actions,
    )

    pred_policy_logits = torch.cat([initial_policy_logits, recurrent_policy_logits], dim=1)
    pred_value_logits = torch.cat([initial_value_logits, recurrent_value_logits], dim=1)

    target_value_dist = model.value_support.scalar_to_logits_target(target_value)
    target_reward_dist = model.reward_support.scalar_to_logits_target(target_reward)

    logp_policy = F.log_softmax(pred_policy_logits, dim=-1)
    policy_loss_per_step = -(target_policy * logp_policy).sum(dim=-1)
    policy_loss = (policy_loss_per_step * policy_mask).sum() / policy_mask.sum().clamp_min(1.0)

    logp_value = F.log_softmax(pred_value_logits, dim=-1)
    value_loss_per_step = -(target_value_dist * logp_value).sum(dim=-1)
    value_loss = (value_loss_per_step * value_mask).sum() / value_mask.sum().clamp_min(1.0)

    logp_reward = F.log_softmax(recurrent_reward_logits, dim=-1)
    reward_loss_per_step = -(target_reward_dist * logp_reward).sum(dim=-1)
    reward_loss = (reward_loss_per_step * reward_mask).sum() / reward_mask.sum().clamp_min(1.0)

    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.reward_loss_weight * reward_loss
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
        "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
    }


def collect_self_play_data(
    model: MuZeroNet,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    mcts_config: MCTSConfig,
    train_config: MuZeroTrainConfig,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, float]:
    total_steps = 0
    terminated_count = 0

    for _ in range(num_episodes):
        episode = generate_self_play_episode(
            model=model,
            mcts_config=mcts_config,
            env_factory=env_factory,
            max_moves=train_config.max_moves_per_episode,
            device=train_config.device,
        )
        replay_buffer.add_episode(episode)
        total_steps += len(episode.steps)
        if episode.terminated:
            terminated_count += 1

    mean_steps = total_steps / max(1, num_episodes)
    return {
        "episodes": float(num_episodes),
        "total_steps": float(total_steps),
        "mean_steps": float(mean_steps),
        "terminated_fraction": float(terminated_count / max(1, num_episodes)),
    }
