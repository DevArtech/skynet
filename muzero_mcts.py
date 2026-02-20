from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from muzero_model import MuZeroNet, observation_batch_to_tensors


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 100
    discount: float = 1.0
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature: float = 1.0
    add_exploration_noise: bool = True


@dataclass
class SearchStats:
    action: int
    visit_counts: dict[int, int]
    q_values: dict[int, float]
    policy_target: dict[int, float]
    root_value: float


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    hidden_state: torch.Tensor | None = None
    children: dict[int, "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MinMaxStats:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def _model_action_space_size(model: MuZeroNet) -> int:
    return model.config.action_space_size


def _legal_model_actions(legal_action_ids: list[int], action_space_size: int) -> list[int]:
    return [a for a in legal_action_ids if 0 <= a < action_space_size]


def _expand_root(
    root: Node,
    model: MuZeroNet,
    observation: dict[str, Any],
    legal_actions: list[int],
    device: torch.device,
) -> float:
    tokens = observation_batch_to_tensors([observation], history_length=model.config.history_length, device=device)
    with torch.no_grad():
        initial = model.initial_inference(tokens)
        policy_logits = initial.policy_logits[0]
        value = model.value_support.logits_to_scalar(initial.value_logits[0]).item()

    root.hidden_state = initial.hidden_state[0]
    root.reward = 0.0

    legal_mask = torch.zeros_like(policy_logits)
    legal_mask[legal_actions] = 1.0
    masked_logits = policy_logits.clone()
    masked_logits[legal_mask <= 0] = -1e9
    priors = torch.softmax(masked_logits, dim=-1)

    for action in legal_actions:
        root.children[action] = Node(prior=float(priors[action].item()))

    return value


def _add_root_dirichlet_noise(root: Node, legal_actions: list[int], cfg: MCTSConfig) -> None:
    if not cfg.add_exploration_noise or not legal_actions:
        return
    noise = torch.distributions.Dirichlet(torch.full((len(legal_actions),), cfg.root_dirichlet_alpha)).sample()
    for i, action in enumerate(legal_actions):
        prior = root.children[action].prior
        root.children[action].prior = (1.0 - cfg.root_exploration_fraction) * prior + (
            cfg.root_exploration_fraction * float(noise[i].item())
        )


def _ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, cfg: MCTSConfig) -> float:
    pb_c = (
        math.log((parent.visit_count + cfg.pb_c_base + 1.0) / cfg.pb_c_base) + cfg.pb_c_init
    ) * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def _select_child(node: Node, min_max_stats: MinMaxStats, cfg: MCTSConfig) -> tuple[int, Node]:
    best_action = -1
    best_child: Node | None = None
    best_score = float("-inf")
    for action, child in node.children.items():
        score = _ucb_score(node, child, min_max_stats, cfg)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    if best_child is None:
        raise RuntimeError("Cannot select child from an unexpanded node.")
    return best_action, best_child


def _expand_with_model(parent: Node, node: Node, action: int, model: MuZeroNet) -> float:
    if parent.hidden_state is None:
        raise RuntimeError("Parent node hidden_state is missing.")

    action_tensor = torch.tensor([action], device=parent.hidden_state.device, dtype=torch.long)
    with torch.no_grad():
        recurrent = model.recurrent_inference(parent.hidden_state.unsqueeze(0), action_tensor)
        next_state = recurrent.hidden_state[0]
        reward = model.reward_support.logits_to_scalar(recurrent.reward_logits[0]).item()
        policy_logits = recurrent.policy_logits[0]
        value = model.value_support.logits_to_scalar(recurrent.value_logits[0]).item()
        priors = torch.softmax(policy_logits, dim=-1)

    node.hidden_state = next_state
    node.reward = reward
    for a in range(_model_action_space_size(model)):
        node.children[a] = Node(prior=float(priors[a].item()))
    return value


def _backpropagate(search_path: list[Node], value: float, discount: float, min_max_stats: MinMaxStats) -> None:
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


def _policy_from_visits(visit_counts: dict[int, int], temperature: float) -> dict[int, float]:
    if not visit_counts:
        return {}
    if temperature <= 1e-8:
        best_action = max(visit_counts, key=visit_counts.get)
        return {a: 1.0 if a == best_action else 0.0 for a in visit_counts}

    values = torch.tensor([max(1e-8, float(v)) for v in visit_counts.values()], dtype=torch.float32)
    values = values.pow(1.0 / temperature)
    values = values / values.sum()
    return {a: float(p.item()) for a, p in zip(visit_counts.keys(), values, strict=False)}


def run_mcts(
    model: MuZeroNet,
    observation: dict[str, Any],
    legal_action_ids: list[int],
    config: MCTSConfig | None = None,
    device: torch.device | str = "cpu",
) -> SearchStats:
    cfg = config or MCTSConfig()
    dev = torch.device(device)
    model = model.to(dev)

    action_space_size = _model_action_space_size(model)
    legal_actions = _legal_model_actions(legal_action_ids, action_space_size)
    if not legal_actions:
        if not legal_action_ids:
            raise ValueError("No legal actions available.")
        fallback = legal_action_ids[0]
        return SearchStats(
            action=fallback,
            visit_counts={fallback: 1},
            q_values={fallback: 0.0},
            policy_target={fallback: 1.0},
            root_value=0.0,
        )

    root = Node(prior=1.0)
    root_value = _expand_root(root, model, observation, legal_actions, dev)
    _add_root_dirichlet_noise(root, legal_actions, cfg)

    min_max_stats = MinMaxStats()
    for _ in range(cfg.num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = _select_child(node, min_max_stats, cfg)
            search_path.append(node)
            if node.visit_count == 0:
                parent = search_path[-2]
                leaf_value = _expand_with_model(parent=parent, node=node, action=action, model=model)
                break
        else:
            # Not expected for root in this setup, but keep safe fallback.
            leaf_value = 0.0

        _backpropagate(search_path, leaf_value, cfg.discount, min_max_stats)

    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    q_values = {action: child.value() for action, child in root.children.items()}
    policy_target = _policy_from_visits(visit_counts, cfg.temperature)
    selected_action = max(policy_target, key=policy_target.get)

    return SearchStats(
        action=selected_action,
        visit_counts=visit_counts,
        q_values=q_values,
        policy_target=policy_target,
        root_value=root_value,
    )


def select_action_with_mcts(
    env: Any,
    model: MuZeroNet,
    config: MCTSConfig | None = None,
    device: torch.device | str = "cpu",
) -> SearchStats:
    """
    Convenience helper that runs MCTS from the environment's current player.
    """
    cfg = config or MCTSConfig()
    observation = env.observe(env.current_player)
    legal_actions = env.legal_actions()
    return run_mcts(
        model=model,
        observation=observation,
        legal_action_ids=legal_actions,
        config=cfg,
        device=device,
    )


def play_one_mcts_step(
    env: Any,
    model: MuZeroNet,
    config: MCTSConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[SearchStats, dict[str, Any], dict[str, float], bool, dict[str, Any]]:
    """
    Runs search, applies the selected action to env, and returns env transition.
    """
    stats = select_action_with_mcts(env=env, model=model, config=config, device=device)
    next_obs, rewards, terminated, info = env.step(stats.action)
    return stats, next_obs, rewards, terminated, info


def clone_env(env: Any) -> Any:
    """
    Utility kept here for future environment-based MCTS variants.
    """
    return copy.deepcopy(env)
