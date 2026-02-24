from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from skyjo_env import CARD_DISTRIBUTION


BOARD_SIZE = 12
COLS = 4
CHOOSE_DECK = 0
CHOOSE_DISCARD = 1
KEEP_DRAWN = 2
DISCARD_DRAWN = 3
CHOOSE_POS_BASE = 4
UNKNOWN_VALUE = -99


def _column_positions(column: int) -> tuple[int, int, int]:
    return (column, column + COLS, column + (2 * COLS))


def _safe_view(obs: dict) -> tuple[list[int], list[int]]:
    current_player = int(obs.get("current_player", 0))
    board_views = obs.get("board_views", {})
    view = board_views.get(current_player)
    if view is None and isinstance(board_views, dict):
        view = board_views.get(str(current_player))
    if not isinstance(view, dict):
        return [UNKNOWN_VALUE] * BOARD_SIZE, [0] * BOARD_SIZE
    values = list(view.get("values", [UNKNOWN_VALUE] * BOARD_SIZE))
    visible_mask = list(view.get("visible_mask", [0] * BOARD_SIZE))
    if len(values) != BOARD_SIZE:
        values = (values + [UNKNOWN_VALUE] * BOARD_SIZE)[:BOARD_SIZE]
    if len(visible_mask) != BOARD_SIZE:
        visible_mask = (visible_mask + [0] * BOARD_SIZE)[:BOARD_SIZE]
    return [int(v) for v in values], [int(v) for v in visible_mask]


def _legal_positions(legal_actions: list[int]) -> list[int]:
    return [int(a) - CHOOSE_POS_BASE for a in legal_actions if int(a) >= CHOOSE_POS_BASE]


def _known_unknown_legal_positions(obs: dict, legal_actions: list[int]) -> tuple[list[int], list[int]]:
    values, visible = _safe_view(obs)
    known: list[int] = []
    unknown: list[int] = []
    for pos in _legal_positions(legal_actions):
        if not (0 <= pos < BOARD_SIZE):
            continue
        if visible[pos] == 1 and values[pos] != UNKNOWN_VALUE:
            known.append(pos)
        else:
            unknown.append(pos)
    return known, unknown


def _expected_unknown_value() -> float:
    total = float(sum(CARD_DISTRIBUTION.values()))
    weighted = float(sum(value * count for value, count in CARD_DISTRIBUTION.items()))
    return weighted / max(1.0, total)


def _estimate_score_from_view(view: dict, unknown_expectation: float) -> float:
    values = list(view.get("values", []))
    visible_mask = list(view.get("visible_mask", []))
    n = min(len(values), len(visible_mask))
    est = 0.0
    for i in range(n):
        if int(visible_mask[i]) == 1 and int(values[i]) != UNKNOWN_VALUE:
            est += float(values[i])
        else:
            est += float(unknown_expectation)
    return est


def _column_needed_values(obs: dict) -> set[int]:
    values, visible = _safe_view(obs)
    needed: set[int] = set()
    for col in range(COLS):
        cpos = _column_positions(col)
        known_vals = [values[p] for p in cpos if visible[p] == 1 and values[p] != UNKNOWN_VALUE]
        if not known_vals:
            continue
        # Value most represented in this partially observed column is the "target".
        freq: dict[int, int] = {}
        for v in known_vals:
            freq[int(v)] = freq.get(int(v), 0) + 1
        target = max(freq, key=freq.get)
        needed.add(target)
    return needed


@dataclass
class HeuristicDecisionBot:
    name: str
    epsilon: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, obs: dict, legal_actions: list[int]) -> int:
        legal = [int(a) for a in legal_actions]
        if not legal:
            raise ValueError(f"{self.name}: no legal actions available.")
        if self.rng.random() < max(0.0, min(1.0, float(self.epsilon))):
            return self.rng.choice(legal)
        phase = str(obs.get("decision_phase", ""))
        if phase == "CHOOSE_SOURCE":
            return self._choose_source(obs, legal)
        if phase == "KEEP_OR_DISCARD":
            return self._keep_or_discard(obs, legal)
        if phase == "CHOOSE_POSITION":
            return self._choose_position(obs, legal)
        return self.rng.choice(legal)

    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        return legal[0]

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        return legal[0]

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        return self.rng.choice(legal)


class GreedyValueReplacementBot(HeuristicDecisionBot):
    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        discard_top = int(obs.get("discard_top", UNKNOWN_VALUE))
        _, visible = _safe_view(obs)
        known_count = int(sum(1 for x in visible if x == 1))
        threshold_take = 2 if known_count < 6 else 4
        if CHOOSE_DISCARD in legal and discard_top <= threshold_take:
            return CHOOSE_DISCARD
        return CHOOSE_DECK if CHOOSE_DECK in legal else legal[0]

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        values, visible = _safe_view(obs)
        known_vals = [values[p] for p in range(BOARD_SIZE) if visible[p] == 1 and values[p] != UNKNOWN_VALUE]
        min_known = min(known_vals) if known_vals else None
        should_keep = drawn <= 2 or (min_known is not None and drawn <= (int(min_known) - 1))
        if should_keep and KEEP_DRAWN in legal:
            return KEEP_DRAWN
        return DISCARD_DRAWN if DISCARD_DRAWN in legal else legal[0]

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        values, visible = _safe_view(obs)
        known, unknown = _known_unknown_legal_positions(obs, legal)
        if known:
            worst_pos = max(known, key=lambda p: values[p])
            return CHOOSE_POS_BASE + worst_pos
        if unknown:
            return CHOOSE_POS_BASE + self.rng.choice(unknown)
        return self.rng.choice(legal)


class InformationFirstFlipBot(HeuristicDecisionBot):
    reveal_target: int = 7

    def __post_init__(self) -> None:
        super().__post_init__()
        self.greedy = GreedyValueReplacementBot(name=f"{self.name}_greedy", epsilon=0.0, seed=self.seed + 17)

    def _known_count(self, obs: dict) -> int:
        _, visible = _safe_view(obs)
        return int(sum(1 for x in visible if x == 1))

    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        if self._known_count(obs) < self.reveal_target and CHOOSE_DECK in legal:
            return CHOOSE_DECK
        return self.greedy._choose_source(obs, legal)

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        if self._known_count(obs) < self.reveal_target and DISCARD_DRAWN in legal:
            return DISCARD_DRAWN
        return self.greedy._keep_or_discard(obs, legal)

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        if self._known_count(obs) < self.reveal_target:
            _, unknown = _known_unknown_legal_positions(obs, legal)
            if unknown:
                return CHOOSE_POS_BASE + self.rng.choice(unknown)
        return self.greedy._choose_position(obs, legal)


class ColumnHunterBot(HeuristicDecisionBot):
    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        discard_top = int(obs.get("discard_top", UNKNOWN_VALUE))
        needed = _column_needed_values(obs)
        if CHOOSE_DISCARD in legal and (discard_top in needed or discard_top <= 1):
            return CHOOSE_DISCARD
        return CHOOSE_DECK if CHOOSE_DECK in legal else legal[0]

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        needed = _column_needed_values(obs)
        if KEEP_DRAWN in legal and (drawn in needed or drawn <= 2):
            return KEEP_DRAWN
        return DISCARD_DRAWN if DISCARD_DRAWN in legal else legal[0]

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        values, visible = _safe_view(obs)
        legal_pos = _legal_positions(legal)
        best_pos: int | None = None
        best_score = float("-inf")
        for pos in legal_pos:
            score = 0.0
            col = pos % COLS
            cpos = _column_positions(col)
            other = [p for p in cpos if p != pos]
            matching = sum(1 for p in other if visible[p] == 1 and values[p] == drawn)
            if matching == 2:
                score += 50.0
            elif matching == 1:
                score += 20.0
            if visible[pos] == 1 and values[pos] != UNKNOWN_VALUE:
                score += float(values[pos]) - float(drawn)
            else:
                score += 2.0
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos is not None:
            return CHOOSE_POS_BASE + best_pos
        return self.rng.choice(legal)


class RiskAwareUnknownReplacementBot(HeuristicDecisionBot):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.unknown_expectation = _expected_unknown_value()

    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        discard_top = int(obs.get("discard_top", UNKNOWN_VALUE))
        if CHOOSE_DISCARD in legal and discard_top <= int(self.unknown_expectation - 1.0):
            return CHOOSE_DISCARD
        return CHOOSE_DECK if CHOOSE_DECK in legal else legal[0]

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        keep = drawn <= int(self.unknown_expectation - 1.0)
        if keep and KEEP_DRAWN in legal:
            return KEEP_DRAWN
        return DISCARD_DRAWN if DISCARD_DRAWN in legal else legal[0]

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        values, _ = _safe_view(obs)
        known, unknown = _known_unknown_legal_positions(obs, legal)
        if known:
            worst_pos = max(known, key=lambda p: values[p])
            if float(values[worst_pos]) > self.unknown_expectation + 1.0:
                return CHOOSE_POS_BASE + worst_pos
        if unknown and drawn <= 2:
            return CHOOSE_POS_BASE + self.rng.choice(unknown)
        if known:
            return CHOOSE_POS_BASE + max(known, key=lambda p: values[p])
        if unknown:
            return CHOOSE_POS_BASE + self.rng.choice(unknown)
        return self.rng.choice(legal)


class EndRoundAggroBot(HeuristicDecisionBot):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.unknown_expectation = _expected_unknown_value()
        self.greedy = GreedyValueReplacementBot(name=f"{self.name}_greedy", epsilon=0.0, seed=self.seed + 29)

    def _ahead_enough(self, obs: dict, lead_threshold: float = 10.0) -> bool:
        board_views = obs.get("board_views", {})
        current_player = int(obs.get("current_player", 0))
        cur_view = board_views.get(current_player) or board_views.get(str(current_player))
        opp_id = 1 - current_player
        opp_view = board_views.get(opp_id) or board_views.get(str(opp_id))
        if not isinstance(cur_view, dict) or not isinstance(opp_view, dict):
            return False
        self_est = _estimate_score_from_view(cur_view, self.unknown_expectation)
        opp_est = _estimate_score_from_view(opp_view, self.unknown_expectation)
        return (opp_est - self_est) > float(lead_threshold)

    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        if self._ahead_enough(obs) and CHOOSE_DECK in legal:
            return CHOOSE_DECK
        return self.greedy._choose_source(obs, legal)

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        if self._ahead_enough(obs) and DISCARD_DRAWN in legal:
            return DISCARD_DRAWN
        return self.greedy._keep_or_discard(obs, legal)

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        if self._ahead_enough(obs):
            _, unknown = _known_unknown_legal_positions(obs, legal)
            if unknown:
                return CHOOSE_POS_BASE + self.rng.choice(unknown)
        return self.greedy._choose_position(obs, legal)


class AntiDiscardBot(HeuristicDecisionBot):
    def _choose_source(self, obs: dict, legal: list[int]) -> int:
        discard_top = int(obs.get("discard_top", UNKNOWN_VALUE))
        if CHOOSE_DISCARD in legal and discard_top <= -1:
            return CHOOSE_DISCARD
        return CHOOSE_DECK if CHOOSE_DECK in legal else legal[0]

    def _keep_or_discard(self, obs: dict, legal: list[int]) -> int:
        drawn = int(obs.get("current_drawn_value", UNKNOWN_VALUE))
        if drawn <= 1 and KEEP_DRAWN in legal:
            return KEEP_DRAWN
        return DISCARD_DRAWN if DISCARD_DRAWN in legal else legal[0]

    def _choose_position(self, obs: dict, legal: list[int]) -> int:
        values, _ = _safe_view(obs)
        known, unknown = _known_unknown_legal_positions(obs, legal)
        if known:
            return CHOOSE_POS_BASE + max(known, key=lambda p: values[p])
        if unknown:
            return CHOOSE_POS_BASE + self.rng.choice(unknown)
        return self.rng.choice(legal)


BOT_REGISTRY: dict[str, Callable[..., HeuristicDecisionBot]] = {
    "greedy_value_replacement": GreedyValueReplacementBot,
    "information_first_flip": InformationFirstFlipBot,
    "column_hunter": ColumnHunterBot,
    "risk_aware_unknown_replacement": RiskAwareUnknownReplacementBot,
    "end_round_aggro": EndRoundAggroBot,
    "anti_discard": AntiDiscardBot,
}


def make_heuristic_bot(name: str, seed: int = 0, epsilon: float = 0.02) -> HeuristicDecisionBot:
    key = str(name).strip().lower()
    if key not in BOT_REGISTRY:
        raise ValueError(f"Unknown heuristic bot '{name}'. Available: {sorted(BOT_REGISTRY.keys())}")
    return BOT_REGISTRY[key](name=key, seed=seed, epsilon=epsilon)

