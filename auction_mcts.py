"""
MCTS for the auction environment — baseline and belief-aware variants.

Both share the same tree-search skeleton. The belief variant conditions
hidden states on ego/current-player embeddings like Skyjo belief MCTS.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from auction_env import NUM_PLAYERS
from auction_model import (
    AuctionBeliefMuZeroNet,
    AuctionConfig,
    AuctionMuZeroNet,
    auction_observation_batch_to_tensors,
)
from muzero_mcts import MCTSConfig, SearchStats


def _autocast_ctx(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class _MinMaxStats:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, v: float) -> None:
        self.minimum = min(self.minimum, v)
        self.maximum = max(self.maximum, v)

    def normalize(self, v: float) -> float:
        if self.maximum > self.minimum:
            return (v - self.minimum) / (self.maximum - self.minimum)
        return v


def _legal_model_actions(legal: list[int], size: int) -> list[int]:
    return [a for a in legal if 0 <= a < size]


def _ucb(parent: _Node, child: _Node, mm: _MinMaxStats, cfg: MCTSConfig) -> float:
    pb_c = (
        math.log((parent.visit_count + cfg.pb_c_base + 1.0) / cfg.pb_c_base) + cfg.pb_c_init
    ) * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    return pb_c * child.prior + mm.normalize(child.value())


def _select_child(node: _Node, mm: _MinMaxStats, cfg: MCTSConfig) -> tuple[int, _Node]:
    best_a, best_c, best_s = -1, None, float("-inf")
    for a, c in node.children.items():
        s = _ucb(node, c, mm, cfg)
        if s > best_s:
            best_a, best_c, best_s = a, c, s
    assert best_c is not None
    return best_a, best_c


def _add_root_noise(root: _Node, legal: list[int], cfg: MCTSConfig) -> None:
    if not cfg.add_exploration_noise or not legal:
        return
    noise = torch.distributions.Dirichlet(
        torch.full((len(legal),), cfg.root_dirichlet_alpha)
    ).sample()
    for i, a in enumerate(legal):
        root.children[a].prior = (
            (1 - cfg.root_exploration_fraction) * root.children[a].prior
            + cfg.root_exploration_fraction * float(noise[i].item())
        )


def _backpropagate(
    path: list[_Node], leaf_value: float, discount: float, offset: float, mm: _MinMaxStats,
) -> None:
    v = leaf_value
    for node in reversed(path):
        node.value_sum += v
        node.visit_count += 1
        mm.update(node.value())
        v = (node.reward + offset) + discount * v


def _policy_from_visits(visits: dict[int, int], temperature: float) -> dict[int, float]:
    if not visits:
        return {}
    if temperature <= 1e-8:
        best = max(visits, key=visits.get)
        return {a: 1.0 if a == best else 0.0 for a in visits}
    counts = torch.tensor([max(1e-8, float(v)) for v in visits.values()], dtype=torch.float32)
    probs = counts.pow(1.0 / temperature)
    probs = probs / probs.sum()
    return {a: float(p.item()) for a, p in zip(visits.keys(), probs, strict=False)}


def _infer_next_player(current: int, action: int, pass_slot: int) -> int:
    if action == pass_slot:
        return current  # round ends, but next node won't matter
    return 1 - current


def _all_legal_in_tree(size: int) -> list[int]:
    """In the imagined tree, all policy slots are legal."""
    return list(range(size))


# ---------------------------------------------------------------------------
# Baseline MCTS
# ---------------------------------------------------------------------------

def run_auction_mcts(
    model: AuctionMuZeroNet,
    observation: dict[str, Any],
    legal_action_ids: list[int],
    config: MCTSConfig | None = None,
    device: str | torch.device = "cpu",
    *,
    mcts_inference_autocast: bool = False,
) -> SearchStats:
    cfg = config or MCTSConfig()
    dev = torch.device(device)
    model = model.to(dev)

    legal = _legal_model_actions(legal_action_ids, model.config.action_space_size)
    if not legal:
        fb = legal_action_ids[0] if legal_action_ids else 0
        return SearchStats(action=fb, visit_counts={fb: 1}, q_values={fb: 0.0}, policy_target={fb: 1.0}, root_value=0.0)

    current_player_id = int(observation["current_player"])
    pass_slot = model.config.action_space_size - 1
    root = _Node(prior=1.0, current_player_id=current_player_id)
    tokens = auction_observation_batch_to_tensors([observation], config=model.config, device=dev)

    ac_ctx = lambda: _autocast_ctx(dev) if mcts_inference_autocast else contextlib.nullcontext()
    with torch.no_grad(), ac_ctx():
        init = model.initial_inference(tokens)
        root.hidden_state = init.hidden_state[0]
        root_value = model.value_support.logits_to_scalar(init.value_logits[0]).item()
        logits = init.policy_logits[0]

    mask = torch.zeros_like(logits)
    mask[legal] = 1.0
    masked = logits.clone()
    masked[mask <= 0] = -1e9
    priors = torch.softmax(masked, dim=-1)
    for a in legal:
        root.children[a] = _Node(prior=float(priors[a].item()), current_player_id=current_player_id)

    _add_root_noise(root, legal, cfg)
    mm = _MinMaxStats()

    for _ in range(cfg.num_simulations):
        node = root
        path = [node]
        while node.expanded():
            action, node = _select_child(node, mm, cfg)
            path.append(node)
            if node.visit_count == 0:
                parent = path[-2]
                next_player = _infer_next_player(parent.current_player_id, action, pass_slot)
                a_t = torch.tensor([action], dtype=torch.long, device=dev)
                with torch.no_grad(), ac_ctx():
                    rec = model.recurrent_inference(parent.hidden_state.unsqueeze(0), a_t)
                    node.hidden_state = rec.hidden_state[0]
                    node.current_player_id = next_player
                    node.reward = model.reward_support.logits_to_scalar(rec.reward_logits[0]).item()
                    value = model.value_support.logits_to_scalar(rec.value_logits[0]).item()
                    probs = torch.softmax(rec.policy_logits[0], dim=-1)
                for la in _all_legal_in_tree(model.config.action_space_size):
                    node.children[la] = _Node(prior=float(probs[la].item()), current_player_id=next_player)
                _backpropagate(path, value, cfg.discount, float(cfg.reward_offset), mm)
                break
        else:
            _backpropagate(path, 0.0, cfg.discount, float(cfg.reward_offset), mm)

    visits = {a: c.visit_count for a, c in root.children.items()}
    q_vals = {a: c.value() for a, c in root.children.items()}
    policy = _policy_from_visits(visits, cfg.temperature)
    chosen = max(policy, key=policy.get)
    return SearchStats(action=chosen, visit_counts=visits, q_values=q_vals, policy_target=policy, root_value=root_value)


# ---------------------------------------------------------------------------
# Belief MCTS
# ---------------------------------------------------------------------------

def run_auction_belief_mcts(
    model: AuctionBeliefMuZeroNet,
    observation: dict[str, Any],
    legal_action_ids: list[int],
    ego_player_id: int,
    config: MCTSConfig | None = None,
    device: str | torch.device = "cpu",
    *,
    mcts_inference_autocast: bool = False,
) -> SearchStats:
    cfg = config or MCTSConfig()
    dev = torch.device(device)
    model = model.to(dev)

    legal = _legal_model_actions(legal_action_ids, model.config.action_space_size)
    if not legal:
        fb = legal_action_ids[0] if legal_action_ids else 0
        return SearchStats(action=fb, visit_counts={fb: 1}, q_values={fb: 0.0}, policy_target={fb: 1.0}, root_value=0.0)

    current_player_id = int(observation["current_player"])
    num_players = NUM_PLAYERS
    pass_slot = model.config.action_space_size - 1
    root = _Node(prior=1.0, current_player_id=current_player_id)
    tokens = auction_observation_batch_to_tensors([observation], config=model.config, device=dev)
    ego_t = torch.tensor([ego_player_id], dtype=torch.long, device=dev)
    cur_t = torch.tensor([current_player_id], dtype=torch.long, device=dev)
    np_t = torch.tensor([num_players], dtype=torch.long, device=dev)

    ac_ctx = lambda: _autocast_ctx(dev) if mcts_inference_autocast else contextlib.nullcontext()
    with torch.no_grad(), ac_ctx():
        init = model.initial_inference(tokens, ego_t, cur_t, np_t)
        root.hidden_state = init.hidden_state[0]
        root_value = model.value_support.logits_to_scalar(init.value_logits[0]).item()
        logits = init.policy_logits[0]

    mask = torch.zeros_like(logits)
    mask[legal] = 1.0
    masked = logits.clone()
    masked[mask <= 0] = -1e9
    priors = torch.softmax(masked, dim=-1)
    for a in legal:
        root.children[a] = _Node(prior=float(priors[a].item()), current_player_id=current_player_id)

    _add_root_noise(root, legal, cfg)
    mm = _MinMaxStats()

    for _ in range(cfg.num_simulations):
        node = root
        path = [node]
        while node.expanded():
            action, node = _select_child(node, mm, cfg)
            path.append(node)
            if node.visit_count == 0:
                parent = path[-2]
                next_player = _infer_next_player(parent.current_player_id, action, pass_slot)
                a_t = torch.tensor([action], dtype=torch.long, device=dev)
                cur = torch.tensor([next_player], dtype=torch.long, device=dev)
                with torch.no_grad(), ac_ctx():
                    next_h, reward_logits = model.dynamics(parent.hidden_state.unsqueeze(0), a_t)
                    conditioned = model._condition_hidden(next_h, ego_t, cur, np_t)
                    p_logits, v_logits, _, _ = model.prediction(conditioned)
                    node.hidden_state = next_h[0]
                    node.current_player_id = next_player
                    node.reward = model.reward_support.logits_to_scalar(reward_logits[0]).item()
                    value = model.value_support.logits_to_scalar(v_logits[0]).item()
                    probs = torch.softmax(p_logits[0], dim=-1)
                for la in _all_legal_in_tree(model.config.action_space_size):
                    node.children[la] = _Node(prior=float(probs[la].item()), current_player_id=next_player)
                _backpropagate(path, value, cfg.discount, float(cfg.reward_offset), mm)
                break
        else:
            _backpropagate(path, 0.0, cfg.discount, float(cfg.reward_offset), mm)

    visits = {a: c.visit_count for a, c in root.children.items()}
    q_vals = {a: c.value() for a, c in root.children.items()}
    policy = _policy_from_visits(visits, cfg.temperature)
    chosen = max(policy, key=policy.get)
    return SearchStats(action=chosen, visit_counts=visits, q_values=q_vals, policy_target=policy, root_value=root_value)
