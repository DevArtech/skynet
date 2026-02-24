from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


CARD_DISTRIBUTION: dict[int, int] = {
    -2: 5,
    -1: 10,
    0: 15,
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10,
    11: 10,
    12: 10,
}

BOARD_SIZE = 12
ROWS = 3
COLS = 4
UNKNOWN_VALUE = -99
ACTION_HISTORY_UNKNOWN = -127

TAKE_DISCARD_BASE = 0
DRAW_DECK_KEEP_BASE = 12
DRAW_DECK_DISCARD_FLIP_BASE = 24
SETUP_FLIP_BASE = 36
TOTAL_ACTIONS = 48


class TurnPhase(IntEnum):
    SETUP = 0
    MAIN = 1
    ROUND_END = 2
    GAME_OVER = 3


class PublicActionType(IntEnum):
    SETUP_FLIP = 0
    TAKE_DISCARD_AND_REPLACE = 1
    DRAW_DECK_KEEP_AND_REPLACE = 2
    DRAW_DECK_DISCARD_AND_FLIP = 3
    COLUMN_CLEARED = 4
    ROUND_END = 5


@dataclass(frozen=True)
class PublicAction:
    step_index: int
    round_index: int
    actor: int
    action_type: PublicActionType
    source: int
    target_pos: int
    a_value: int
    b_value: int
    note: str = ""

    def to_token(self) -> list[int]:
        return [
            self.actor,
            int(self.action_type),
            self.source,
            self.target_pos,
            self.a_value,
            self.b_value,
        ]


@dataclass
class PlayerBoard:
    cards: list[int | None]
    visible: list[bool]
    removed: list[bool]

    @classmethod
    def empty(cls) -> "PlayerBoard":
        return cls(cards=[None] * BOARD_SIZE, visible=[False] * BOARD_SIZE, removed=[False] * BOARD_SIZE)

    def score(self) -> int:
        total = 0
        for i in range(BOARD_SIZE):
            if self.removed[i]:
                continue
            value = self.cards[i]
            if value is None:
                continue
            total += value
        return total

    def all_active_cards_visible(self) -> bool:
        for i in range(BOARD_SIZE):
            if self.removed[i]:
                continue
            if not self.visible[i]:
                return False
        return True

    def to_public_slots(self, owner_id: int) -> list[list[int]]:
        slots: list[list[int]] = []
        for pos in range(BOARD_SIZE):
            is_visible = self.visible[pos]
            if self.removed[pos]:
                visible_flag = 1
                value = 0
            else:
                visible_flag = 1 if is_visible else 0
                value = self.cards[pos] if is_visible else UNKNOWN_VALUE
                if value is None:
                    value = UNKNOWN_VALUE
            slots.append([owner_id, pos, visible_flag, int(value)])
        return slots


def _column_positions(column: int) -> tuple[int, int, int]:
    return (column, column + COLS, column + (2 * COLS))


def action_to_macro(action_id: int) -> tuple[str, int]:
    if TAKE_DISCARD_BASE <= action_id < TAKE_DISCARD_BASE + BOARD_SIZE:
        return ("TAKE_DISCARD_AND_REPLACE", action_id - TAKE_DISCARD_BASE)
    if DRAW_DECK_KEEP_BASE <= action_id < DRAW_DECK_KEEP_BASE + BOARD_SIZE:
        return ("DRAW_DECK_KEEP_AND_REPLACE", action_id - DRAW_DECK_KEEP_BASE)
    if DRAW_DECK_DISCARD_FLIP_BASE <= action_id < DRAW_DECK_DISCARD_FLIP_BASE + BOARD_SIZE:
        return ("DRAW_DECK_DISCARD_AND_FLIP", action_id - DRAW_DECK_DISCARD_FLIP_BASE)
    if SETUP_FLIP_BASE <= action_id < SETUP_FLIP_BASE + BOARD_SIZE:
        return ("SETUP_FLIP", action_id - SETUP_FLIP_BASE)
    raise ValueError(f"Invalid action id: {action_id}")


def macro_to_action(macro: str, position: int) -> int:
    if position < 0 or position >= BOARD_SIZE:
        raise ValueError(f"Invalid board position: {position}")
    if macro == "TAKE_DISCARD_AND_REPLACE":
        return TAKE_DISCARD_BASE + position
    if macro == "DRAW_DECK_KEEP_AND_REPLACE":
        return DRAW_DECK_KEEP_BASE + position
    if macro == "DRAW_DECK_DISCARD_AND_FLIP":
        return DRAW_DECK_DISCARD_FLIP_BASE + position
    if macro == "SETUP_FLIP":
        return SETUP_FLIP_BASE + position
    raise ValueError(f"Invalid macro action: {macro}")


class SkyjoEnv:
    """
    Deterministic, seedable, fully-observable Skyjo environment with public observation views.

    Notes:
    - Supports 2-8 players.
    - Implements all turn mechanics and round/game scoring.
    - Action ids 0..47 include setup and turn macros.
    """

    def __init__(
        self,
        num_players: int = 2,
        seed: int = 0,
        history_window_k: int = 16,
        score_limit: int = 100,
        setup_mode: str = "auto",
        manual_initial_reveals: bool = False,
    ) -> None:
        if num_players < 2 or num_players > 8:
            raise ValueError("Skyjo officially supports 2-8 players.")
        self.num_players = num_players
        self.initial_seed = seed
        self.rng = random.Random(seed)
        self.history_window_k = history_window_k
        self.score_limit = score_limit
        normalized_setup_mode = setup_mode.strip().lower()
        if normalized_setup_mode not in {"auto", "manual"}:
            raise ValueError("setup_mode must be either 'auto' or 'manual'.")
        self.setup_mode = normalized_setup_mode
        # Backward-compatible override: if manually requested, force manual setup mode.
        if manual_initial_reveals:
            self.setup_mode = "manual"
        self.manual_initial_reveals = self.setup_mode == "manual"

        self.round_index = 0
        self.global_step = 0
        self.turns_in_round = 0
        self.phase = TurnPhase.SETUP
        self.current_player = 0
        self.boards: list[PlayerBoard] = []
        self.scores: list[int] = [0] * self.num_players
        self.round_scores: list[int] = [0] * self.num_players
        self.deck: list[int] = []
        self.discard_pile: list[int] = []
        self.public_history: list[PublicAction] = []
        self.round_history_start_index = 0
        self.setup_reveals_remaining: list[int] = [2] * self.num_players
        self.pending_final_turn_players: set[int] = set()
        self.round_ender: int | None = None
        self.column_clear_used_this_round: list[bool] = [False] * self.num_players
        self.game_over = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.initial_seed = seed
        self.rng = random.Random(self.initial_seed)
        self.scores = [0] * self.num_players
        self.round_index = 0
        self.global_step = 0
        self.public_history = []
        self.round_history_start_index = 0
        self.game_over = False
        self._start_new_round(starting_player=0)
        return self.observe(self.current_player)

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        board = self.boards[self.current_player]
        if self.phase == TurnPhase.SETUP:
            actions = []
            for pos in range(BOARD_SIZE):
                if not board.removed[pos] and not board.visible[pos]:
                    actions.append(SETUP_FLIP_BASE + pos)
            return actions

        if self.phase in (TurnPhase.MAIN, TurnPhase.ROUND_END):
            actions: list[int] = []
            if self.discard_pile:
                for pos in range(BOARD_SIZE):
                    if not board.removed[pos]:
                        actions.append(TAKE_DISCARD_BASE + pos)
            for pos in range(BOARD_SIZE):
                if not board.removed[pos]:
                    actions.append(DRAW_DECK_KEEP_BASE + pos)
            for pos in range(BOARD_SIZE):
                if not board.removed[pos] and not board.visible[pos]:
                    actions.append(DRAW_DECK_DISCARD_FLIP_BASE + pos)
            return actions
        return []

    def observe(self, observer_player: int, history_k: int | None = None) -> dict[str, Any]:
        if history_k is None:
            history_k = self.history_window_k
        owner_order = [observer_player] + [p for p in range(self.num_players) if p != observer_player]
        board_tokens: list[list[int]] = []
        board_views: dict[int, dict[str, list[int]]] = {}
        for owner in owner_order:
            board = self.boards[owner]
            values: list[int] = []
            visible_mask: list[int] = []
            for pos in range(BOARD_SIZE):
                if board.removed[pos]:
                    values.append(0)
                    visible_mask.append(1)
                    board_tokens.append([owner, pos, 1, 0])
                    continue
                is_visible = board.visible[pos]
                visible_mask.append(1 if is_visible else 0)
                value = board.cards[pos] if is_visible else UNKNOWN_VALUE
                if value is None:
                    value = UNKNOWN_VALUE
                values.append(int(value))
                board_tokens.append([owner, pos, 1 if is_visible else 0, int(value)])
            board_views[owner] = {"values": values, "visible_mask": visible_mask}

        discard_top = self.discard_pile[-1] if self.discard_pile else UNKNOWN_VALUE
        discard_token = [discard_top, len(self.discard_pile)]

        phase_ratio = self._round_progress_ratio()
        if phase_ratio < 0.34:
            phase_bucket = 0
        elif phase_ratio < 0.67:
            phase_bucket = 1
        else:
            phase_bucket = 2
        global_token = [
            len(self.deck),
            self.global_step,
            self.current_player,
            int(self.phase),
            phase_bucket,
            self.round_index,
        ]

        last_k = self.public_history[-history_k:]
        action_tokens = [a.to_token() for a in last_k]

        since_last_self = self._public_actions_since_player_last_turn(observer_player)
        since_last_self_tokens = [a.to_token() for a in since_last_self[-history_k:]]

        return {
            "observer": observer_player,
            "phase": self.phase.name,
            "current_player": self.current_player,
            "scores": list(self.scores),
            "round_scores": list(self.round_scores),
            "deck_size": len(self.deck),
            "discard_top": discard_top,
            "discard_pile": list(self.discard_pile),
            "board_views": board_views,
            "history": [self._action_to_public_dict(a) for a in last_k],
            "history_since_last_turn": [self._action_to_public_dict(a) for a in since_last_self[-history_k:]],
            "tokens": {
                "board_tokens": board_tokens,
                "discard_token": discard_token,
                "global_token": global_token,
                "action_tokens": action_tokens,
                "action_tokens_since_last_turn": since_last_self_tokens,
            },
        }

    def get_full_state(self) -> dict[str, Any]:
        return {
            "seed": self.initial_seed,
            "num_players": self.num_players,
            "round_index": self.round_index,
            "global_step": self.global_step,
            "turns_in_round": self.turns_in_round,
            "phase": self.phase.name,
            "current_player": self.current_player,
            "scores": list(self.scores),
            "round_scores": list(self.round_scores),
            "deck_order_remaining": list(self.deck),
            "discard_pile": list(self.discard_pile),
            "boards": [
                {
                    "cards": list(board.cards),
                    "visible": list(board.visible),
                    "removed": list(board.removed),
                }
                for board in self.boards
            ],
            "setup_reveals_remaining": list(self.setup_reveals_remaining),
            "round_ender": self.round_ender,
            "pending_final_turn_players": sorted(self.pending_final_turn_players),
            "column_clear_used_this_round": list(self.column_clear_used_this_round),
            "game_over": self.game_over,
            "public_history_size": len(self.public_history),
        }

    def step(self, action_id: int) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        if self.game_over:
            raise RuntimeError("Game is already over. Call reset() to start a new game.")
        if action_id not in self.legal_actions():
            raise ValueError(f"Illegal action {action_id} for player {self.current_player} in phase {self.phase.name}")

        actor = self.current_player
        macro, pos = action_to_macro(action_id)
        board = self.boards[actor]
        self.global_step += 1

        if macro == "SETUP_FLIP":
            board.visible[pos] = True
            self.setup_reveals_remaining[actor] -= 1
            self._record_action(
                actor=actor,
                action_type=PublicActionType.SETUP_FLIP,
                source=ACTION_HISTORY_UNKNOWN,
                target_pos=pos,
                a_value=board.cards[pos] if board.cards[pos] is not None else ACTION_HISTORY_UNKNOWN,
                b_value=ACTION_HISTORY_UNKNOWN,
                note="setup reveal",
            )
            self._advance_setup_turn()

        elif macro == "TAKE_DISCARD_AND_REPLACE":
            drawn = self.discard_pile.pop()
            replaced = board.cards[pos]
            board.cards[pos] = drawn
            board.visible[pos] = True
            if replaced is not None:
                self.discard_pile.append(replaced)
            self._record_action(
                actor=actor,
                action_type=PublicActionType.TAKE_DISCARD_AND_REPLACE,
                source=1,
                target_pos=pos,
                a_value=drawn,
                b_value=replaced if replaced is not None else ACTION_HISTORY_UNKNOWN,
                note="take discard, replace",
            )
            self._resolve_columns(actor)
            self._advance_main_turn(actor)

        elif macro == "DRAW_DECK_KEEP_AND_REPLACE":
            drawn = self._draw_from_deck()
            replaced = board.cards[pos]
            board.cards[pos] = drawn
            board.visible[pos] = True
            if replaced is not None:
                self.discard_pile.append(replaced)
            self._record_action(
                actor=actor,
                action_type=PublicActionType.DRAW_DECK_KEEP_AND_REPLACE,
                source=0,
                target_pos=pos,
                a_value=drawn,
                b_value=replaced if replaced is not None else ACTION_HISTORY_UNKNOWN,
                note="draw keep replace",
            )
            self._resolve_columns(actor)
            self._advance_main_turn(actor)

        elif macro == "DRAW_DECK_DISCARD_AND_FLIP":
            drawn = self._draw_from_deck()
            self.discard_pile.append(drawn)
            board.visible[pos] = True
            revealed = board.cards[pos]
            self._record_action(
                actor=actor,
                action_type=PublicActionType.DRAW_DECK_DISCARD_AND_FLIP,
                source=0,
                target_pos=pos,
                a_value=drawn,
                b_value=revealed if revealed is not None else ACTION_HISTORY_UNKNOWN,
                note="draw discard flip",
            )
            self._resolve_columns(actor)
            self._advance_main_turn(actor)

        else:
            raise RuntimeError(f"Unhandled macro action {macro}")

        rewards = {f"player_{p}": 0.0 for p in range(self.num_players)}
        if self.phase == TurnPhase.GAME_OVER:
            winner = min(range(self.num_players), key=lambda p: self.scores[p])
            for p in range(self.num_players):
                rewards[f"player_{p}"] = 1.0 if p == winner else -1.0
        elif self._last_round_ended():
            for p in range(self.num_players):
                rewards[f"player_{p}"] = -float(self.round_scores[p])

        info = {
            "actor": actor,
            "action_id": action_id,
            "macro_action": macro,
            "position": pos,
            "phase": self.phase.name,
            "legal_actions_next": self.legal_actions(),
            "round_index": self.round_index,
            "scores": list(self.scores),
            "round_scores": list(self.round_scores),
        }
        next_obs = self.observe(self.current_player)
        terminated = self.phase == TurnPhase.GAME_OVER
        return next_obs, rewards, terminated, info

    def _start_new_round(self, starting_player: int) -> None:
        self.round_index += 1
        self.turns_in_round = 0
        self.round_ender = None
        self.pending_final_turn_players = set()
        self.setup_reveals_remaining = [2] * self.num_players
        self.column_clear_used_this_round = [False] * self.num_players
        self.phase = TurnPhase.SETUP
        self.current_player = starting_player
        self.round_history_start_index = len(self.public_history)

        self.deck = self._build_and_shuffle_deck()
        self.discard_pile = []
        self.boards = [PlayerBoard.empty() for _ in range(self.num_players)]
        for player in range(self.num_players):
            for pos in range(BOARD_SIZE):
                self.boards[player].cards[pos] = self.deck.pop()

        starter_discard = self._draw_from_deck()
        self.discard_pile.append(starter_discard)

        if not self.manual_initial_reveals:
            for player in range(self.num_players):
                options = list(range(BOARD_SIZE))
                self.rng.shuffle(options)
                reveal_a, reveal_b = options[0], options[1]
                self.boards[player].visible[reveal_a] = True
                self.boards[player].visible[reveal_b] = True
                self.setup_reveals_remaining[player] = 0
                self._record_action(
                    actor=player,
                    action_type=PublicActionType.SETUP_FLIP,
                    source=ACTION_HISTORY_UNKNOWN,
                    target_pos=reveal_a,
                    a_value=(
                        self.boards[player].cards[reveal_a]
                        if self.boards[player].cards[reveal_a] is not None
                        else ACTION_HISTORY_UNKNOWN
                    ),
                    b_value=ACTION_HISTORY_UNKNOWN,
                    note="auto setup reveal",
                )
                self._record_action(
                    actor=player,
                    action_type=PublicActionType.SETUP_FLIP,
                    source=ACTION_HISTORY_UNKNOWN,
                    target_pos=reveal_b,
                    a_value=(
                        self.boards[player].cards[reveal_b]
                        if self.boards[player].cards[reveal_b] is not None
                        else ACTION_HISTORY_UNKNOWN
                    ),
                    b_value=ACTION_HISTORY_UNKNOWN,
                    note="auto setup reveal",
                )
            self.phase = TurnPhase.MAIN
            self.current_player = self._starting_player_from_revealed_sums()

    def _starting_player_from_revealed_sums(self) -> int:
        sums: list[int] = []
        for player in range(self.num_players):
            total = 0
            board = self.boards[player]
            for i in range(BOARD_SIZE):
                if board.visible[i] and board.cards[i] is not None and not board.removed[i]:
                    total += board.cards[i]  # type: ignore[operator]
            sums.append(total)
        highest = max(sums)
        candidates = [p for p, s in enumerate(sums) if s == highest]
        return min(candidates)

    def _advance_setup_turn(self) -> None:
        if self.setup_reveals_remaining[self.current_player] > 0:
            return
        next_player = (self.current_player + 1) % self.num_players
        while self.setup_reveals_remaining[next_player] == 0 and next_player != self.current_player:
            next_player = (next_player + 1) % self.num_players
        if all(v == 0 for v in self.setup_reveals_remaining):
            self.phase = TurnPhase.MAIN
            self.current_player = self._starting_player_from_revealed_sums()
            return
        self.current_player = next_player

    def _advance_main_turn(self, actor: int) -> None:
        self.turns_in_round += 1
        actor_board = self.boards[actor]
        if self.round_ender is None and actor_board.all_active_cards_visible():
            self.round_ender = actor
            self.pending_final_turn_players = {p for p in range(self.num_players) if p != actor}
            self.phase = TurnPhase.ROUND_END

        if self.round_ender is not None and actor in self.pending_final_turn_players:
            self.pending_final_turn_players.remove(actor)

        if self.round_ender is not None and not self.pending_final_turn_players:
            self._finish_round()
            return

        self.current_player = (self.current_player + 1) % self.num_players

    def _finish_round(self) -> None:
        per_round = [board.score() for board in self.boards]
        if self.round_ender is not None:
            ender = self.round_ender
            ender_score = per_round[ender]
            others = [per_round[p] for p in range(self.num_players) if p != ender]
            if others and ender_score > min(others):
                per_round[ender] = ender_score * 2

        self.round_scores = per_round
        for p in range(self.num_players):
            self.scores[p] += per_round[p]

        self._record_action(
            actor=self.round_ender if self.round_ender is not None else ACTION_HISTORY_UNKNOWN,
            action_type=PublicActionType.ROUND_END,
            source=ACTION_HISTORY_UNKNOWN,
            target_pos=ACTION_HISTORY_UNKNOWN,
            a_value=min(self.scores),
            b_value=max(self.scores),
            note="round finished",
        )

        if any(score >= self.score_limit for score in self.scores):
            self.phase = TurnPhase.GAME_OVER
            self.game_over = True
            return

        next_starting_player = self.round_ender if self.round_ender is not None else self.current_player
        self._start_new_round(starting_player=next_starting_player)

    def _resolve_columns(self, player: int) -> None:
        if self.column_clear_used_this_round[player]:
            return
        board = self.boards[player]
        for col in range(COLS):
            p0, p1, p2 = _column_positions(col)
            if board.removed[p0] or board.removed[p1] or board.removed[p2]:
                continue
            if not (board.visible[p0] and board.visible[p1] and board.visible[p2]):
                continue
            v0, v1, v2 = board.cards[p0], board.cards[p1], board.cards[p2]
            if v0 is None or v1 is None or v2 is None:
                continue
            if v0 == v1 == v2:
                board.removed[p0] = True
                board.removed[p1] = True
                board.removed[p2] = True
                board.cards[p0] = None
                board.cards[p1] = None
                board.cards[p2] = None
                board.visible[p0] = True
                board.visible[p1] = True
                board.visible[p2] = True
                self._record_action(
                    actor=player,
                    action_type=PublicActionType.COLUMN_CLEARED,
                    source=ACTION_HISTORY_UNKNOWN,
                    target_pos=col,
                    a_value=v0,
                    b_value=0,
                    note="column cleared",
                )
                self.column_clear_used_this_round[player] = True
                break

    def _draw_from_deck(self) -> int:
        if not self.deck:
            self._refill_deck_from_discard()
        if not self.deck:
            raise RuntimeError("Deck is empty and cannot be refilled.")
        return self.deck.pop()

    def _refill_deck_from_discard(self) -> None:
        if len(self.discard_pile) <= 1:
            return
        top = self.discard_pile.pop()
        pool = self.discard_pile
        self.rng.shuffle(pool)
        self.deck.extend(pool)
        self.discard_pile = [top]

    def _build_and_shuffle_deck(self) -> list[int]:
        deck: list[int] = []
        for value, count in CARD_DISTRIBUTION.items():
            deck.extend([value] * count)
        self.rng.shuffle(deck)
        return deck

    def _record_action(
        self,
        actor: int,
        action_type: PublicActionType,
        source: int,
        target_pos: int,
        a_value: int,
        b_value: int,
        note: str = "",
    ) -> None:
        self.public_history.append(
            PublicAction(
                step_index=self.global_step,
                round_index=self.round_index,
                actor=actor,
                action_type=action_type,
                source=source,
                target_pos=target_pos,
                a_value=a_value,
                b_value=b_value,
                note=note,
            )
        )

    def _public_actions_since_player_last_turn(self, player: int) -> list[PublicAction]:
        idx = len(self.public_history) - 1
        found_count = 0
        while idx >= self.round_history_start_index:
            if self.public_history[idx].actor == player:
                found_count += 1
                if found_count == 2:
                    break
            idx -= 1
        return self.public_history[idx + 1 :]

    def _action_to_public_dict(self, action: PublicAction) -> dict[str, Any]:
        return {
            "step_index": action.step_index,
            "round_index": action.round_index,
            "actor": action.actor,
            "action_type": action.action_type.name,
            "source": action.source,
            "target_pos": action.target_pos,
            "a_value": action.a_value,
            "b_value": action.b_value,
            "note": action.note,
        }

    def _round_progress_ratio(self) -> float:
        max_round_steps_guess = self.num_players * 30
        return min(1.0, self.turns_in_round / max_round_steps_guess)

    def _last_round_ended(self) -> bool:
        if not self.public_history:
            return False
        return self.public_history[-1].action_type == PublicActionType.ROUND_END


def run_random_game(
    num_players: int,
    seed: int,
    history_k: int = 16,
    score_limit: int = 100,
    setup_mode: str = "auto",
    max_steps: int = 10000,
) -> dict[str, Any]:
    env = SkyjoEnv(
        num_players=num_players,
        seed=seed,
        history_window_k=history_k,
        score_limit=score_limit,
        setup_mode=setup_mode,
    )
    env.reset()
    local_rng = random.Random(seed + 17)
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        legal = env.legal_actions()
        if not legal:
            break
        action = local_rng.choice(legal)
        _, _, terminated, _ = env.step(action)
        steps += 1
    winner = min(range(env.num_players), key=lambda p: env.scores[p])
    return {
        "seed": seed,
        "num_players": num_players,
        "steps": steps,
        "winner": winner,
        "scores": env.scores,
        "round_index": env.round_index,
        "terminated": terminated,
    }


def _format_actions(actions: list[int], limit: int = 15) -> str:
    labels = []
    for action in actions[:limit]:
        macro, pos = action_to_macro(action)
        labels.append(f"{action}:{macro}@{pos}")
    if len(actions) > limit:
        labels.append("...")
    return ", ".join(labels)


def cli_play(args: argparse.Namespace) -> None:
    env = SkyjoEnv(
        num_players=args.players,
        seed=args.seed,
        history_window_k=args.history_k,
        score_limit=args.score_limit,
        setup_mode=args.setup_mode,
    )
    env.reset()
    print(f"Skyjo CLI game started (seed={args.seed}, players={args.players})")
    while env.phase != TurnPhase.GAME_OVER:
        p = env.current_player
        obs = env.observe(p)
        print(f"\nRound {env.round_index} | Phase {env.phase.name} | Player {p} turn")
        print(f"Scores: {env.scores}")
        print(f"Discard top: {obs['discard_top']} | Deck size: {obs['deck_size']}")
        print(f"Your visible mask: {obs['board_views'][p]['visible_mask']}")
        print(f"Your values view:  {obs['board_views'][p]['values']}")
        legal = env.legal_actions()
        print(f"Legal actions ({len(legal)}): {_format_actions(legal)}")

        while True:
            raw = input("Choose action id: ").strip()
            if not raw.isdigit():
                print("Enter a numeric action id.")
                continue
            chosen = int(raw)
            if chosen not in legal:
                print("Illegal action. Try again.")
                continue
            break
        _, _, _, info = env.step(chosen)
        print(f"Applied: {info['macro_action']} @ {info['position']}")

    winner = min(range(env.num_players), key=lambda i: env.scores[i])
    print("\nGame over")
    print(f"Final scores: {env.scores}")
    print(f"Winner: player_{winner}")


def cli_simulate(args: argparse.Namespace) -> None:
    results = []
    for i in range(args.games):
        seed = args.seed + i
        result = run_random_game(
            num_players=args.players,
            seed=seed,
            history_k=args.history_k,
            score_limit=args.score_limit,
            setup_mode=args.setup_mode,
            max_steps=args.max_steps,
        )
        results.append(result)
    wins = [0] * args.players
    for result in results:
        wins[result["winner"]] += 1
    print(json.dumps({"games": results, "wins": wins}, indent=2))


def cli_inspect(args: argparse.Namespace) -> None:
    env = SkyjoEnv(
        num_players=args.players,
        seed=args.seed,
        history_window_k=args.history_k,
        score_limit=args.score_limit,
        setup_mode=args.setup_mode,
    )
    obs = env.reset()
    print(json.dumps({"full_state": env.get_full_state(), "observation_player_0": obs}, indent=2))
    for _ in range(args.steps):
        legal = env.legal_actions()
        if not legal:
            break
        action = legal[0]
        obs, _, terminated, _ = env.step(action)
        if terminated:
            break
    print(json.dumps({"after_steps_state": env.get_full_state(), "observation_current": obs}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic CLI Skyjo environment")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--players", type=int, default=2, help="Number of players (2-8)")
        p.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed")
        p.add_argument("--history-k", type=int, default=16, help="Public history token window size")
        p.add_argument("--score-limit", type=int, default=100, help="Game ends when any score reaches this")
        p.add_argument(
            "--setup-mode",
            type=str,
            choices=["auto", "manual"],
            default="auto",
            help="Setup reveal mode: auto (random by seed) or manual (player flips).",
        )

    play = sub.add_parser("play", help="Play game manually in CLI")
    add_common_flags(play)
    play.set_defaults(setup_mode="manual")
    play.set_defaults(func=cli_play)

    simulate = sub.add_parser("simulate", help="Run random-policy self-play simulations")
    add_common_flags(simulate)
    simulate.add_argument("--games", type=int, default=5, help="Number of games")
    simulate.add_argument("--max-steps", type=int, default=10000, help="Step limit per game")
    simulate.set_defaults(func=cli_simulate)

    inspect = sub.add_parser("inspect", help="Inspect full state and observation schema")
    add_common_flags(inspect)
    inspect.add_argument("--steps", type=int, default=3, help="Auto-played steps before second dump")
    inspect.set_defaults(func=cli_inspect)

    return parser


def cli_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    cli_main()
