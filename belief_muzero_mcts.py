from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from belief_muzero_model import BeliefAwareMuZeroNet
from muzero_mcts import MCTSConfig, SearchStats
from muzero_model import observation_batch_to_tensors


@dataclass
class _Node:
    prior: float
    current_player_id: int
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    hidden_state: torch.Tensor | None = None
    children: dict[int, "_Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class _MinMaxStats:
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


def _legal_actions_for_model(legal_action_ids: list[int], action_space_size: int) -> list[int]:
    return [a for a in legal_action_ids if 0 <= a < action_space_size]


def _policy_from_visits(visit_counts: dict[int, int], temperature: float) -> dict[int, float]:
    if not visit_counts:
        return {}
    if temperature <= 1e-8:
        best_action = max(visit_counts, key=visit_counts.get)
        return {a: 1.0 if a == best_action else 0.0 for a in visit_counts}
    counts = torch.tensor([max(1e-8, float(v)) for v in visit_counts.values()], dtype=torch.float32)
    probs = counts.pow(1.0 / temperature)
    probs = probs / probs.sum()
    return {a: float(p.item()) for a, p in zip(visit_counts.keys(), probs, strict=False)}


def _ucb_score(parent: _Node, child: _Node, min_max_stats: _MinMaxStats, cfg: MCTSConfig) -> float:
    pb_c = (
        math.log((parent.visit_count + cfg.pb_c_base + 1.0) / cfg.pb_c_base) + cfg.pb_c_init
    ) * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    return pb_c * child.prior + min_max_stats.normalize(child.value())


def _select_child(node: _Node, min_max_stats: _MinMaxStats, cfg: MCTSConfig) -> tuple[int, _Node]:
    best_action = -1
    best_child: _Node | None = None
    best_score = float("-inf")
    for action, child in node.children.items():
        score = _ucb_score(node, child, min_max_stats, cfg)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    if best_child is None:
        raise RuntimeError("Cannot select from unexpanded node.")
    return best_action, best_child


def _add_root_noise(root: _Node, legal_actions: list[int], cfg: MCTSConfig) -> None:
    if not cfg.add_exploration_noise or not legal_actions:
        return
    noise = torch.distributions.Dirichlet(torch.full((len(legal_actions),), cfg.root_dirichlet_alpha)).sample()
    for i, action in enumerate(legal_actions):
        root.children[action].prior = (1.0 - cfg.root_exploration_fraction) * root.children[action].prior + (
            cfg.root_exploration_fraction * float(noise[i].item())
        )


def _backpropagate(path: list[_Node], leaf_value: float, discount: float, min_max_stats: _MinMaxStats) -> None:
    value = leaf_value
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


def run_belief_mcts(
    model: BeliefAwareMuZeroNet,
    observation: dict[str, Any],
    legal_action_ids: list[int],
    ego_player_id: int,
    config: MCTSConfig | None = None,
    device: str | torch.device = "cpu",
) -> SearchStats:
    cfg = config or MCTSConfig()
    dev = torch.device(device)
    model = model.to(dev)

    action_space_size = model.config.action_space_size
    legal_actions = _legal_actions_for_model(legal_action_ids, action_space_size)
    if not legal_actions:
        fallback = legal_action_ids[0] if legal_action_ids else 0
        return SearchStats(
            action=fallback,
            visit_counts={fallback: 1},
            q_values={fallback: 0.0},
            policy_target={fallback: 1.0},
            root_value=0.0,
        )

    current_player_id = int(observation["current_player"])
    num_players = int(len(observation.get("scores", [])) or 2)

    root = _Node(prior=1.0, current_player_id=current_player_id)
    obs_tokens = observation_batch_to_tensors([observation], history_length=model.config.history_length, device=dev)
    ego_tensor = torch.tensor([ego_player_id], dtype=torch.long, device=dev)
    cur_tensor = torch.tensor([current_player_id], dtype=torch.long, device=dev)
    nplayers_tensor = torch.tensor([num_players], dtype=torch.long, device=dev)

    with torch.no_grad():
        initial = model.initial_inference(obs_tokens, ego_tensor, cur_tensor, nplayers_tensor)
        root.hidden_state = initial.hidden_state[0]
        root_value = model.value_support.logits_to_scalar(initial.value_logits[0]).item()
        root_logits = initial.policy_logits[0]

    mask = torch.zeros_like(root_logits)
    mask[legal_actions] = 1.0
    masked = root_logits.clone()
    masked[mask <= 0] = -1e9
    priors = torch.softmax(masked, dim=-1)
    for action in legal_actions:
        root.children[action] = _Node(
            prior=float(priors[action].item()),
            current_player_id=current_player_id,
        )

    _add_root_noise(root, legal_actions, cfg)
    min_max_stats = _MinMaxStats()

    for _ in range(cfg.num_simulations):
        node = root
        path = [node]
        chosen_action = -1

        while node.expanded():
            chosen_action, node = _select_child(node, min_max_stats, cfg)
            path.append(node)
            if node.visit_count == 0:
                parent = path[-2]
                if parent.hidden_state is None:
                    raise RuntimeError("Missing parent hidden state during expansion.")
                next_player_id = (parent.current_player_id + 1) % max(1, num_players)
                a = torch.tensor([chosen_action], dtype=torch.long, device=dev)
                cur = torch.tensor([next_player_id], dtype=torch.long, device=dev)
                with torch.no_grad():
                    rec = model.recurrent_inference(parent.hidden_state.unsqueeze(0), a, ego_tensor, cur, nplayers_tensor)
                    node.hidden_state = rec.hidden_state[0]
                    node.current_player_id = next_player_id
                    node.reward = model.reward_support.logits_to_scalar(rec.reward_logits[0]).item()
                    value = model.value_support.logits_to_scalar(rec.value_logits[0]).item()
                    probs = torch.softmax(rec.policy_logits[0], dim=-1)
                for action_id in range(action_space_size):
                    node.children[action_id] = _Node(prior=float(probs[action_id].item()), current_player_id=next_player_id)
                _backpropagate(path, value, cfg.discount, min_max_stats)
                break
        else:
            _backpropagate(path, 0.0, cfg.discount, min_max_stats)

    visits = {action: child.visit_count for action, child in root.children.items()}
    q_values = {action: child.value() for action, child in root.children.items()}
    policy_target = _policy_from_visits(visits, cfg.temperature)
    selected_action = max(policy_target, key=policy_target.get)
    return SearchStats(
        action=selected_action,
        visit_counts=visits,
        q_values=q_values,
        policy_target=policy_target,
        root_value=float(root_value),
    )
