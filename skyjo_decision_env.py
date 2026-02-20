from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from skyjo_env import (
    ACTION_HISTORY_UNKNOWN,
    BOARD_SIZE,
    DRAW_DECK_DISCARD_FLIP_BASE,
    DRAW_DECK_KEEP_BASE,
    PublicActionType,
    SETUP_FLIP_BASE,
    SkyjoEnv,
    TAKE_DISCARD_BASE,
    TurnPhase,
    UNKNOWN_VALUE,
)


class DecisionAction(IntEnum):
    CHOOSE_DECK = 0
    CHOOSE_DISCARD = 1
    KEEP_DRAWN = 2
    DISCARD_DRAWN = 3
    CHOOSE_POS_BASE = 4


DECISION_ACTION_SPACE = 16


class DecisionPhase(IntEnum):
    SETUP_REVEAL = 0
    CHOOSE_SOURCE = 1
    KEEP_OR_DISCARD = 2
    CHOOSE_POSITION = 3


@dataclass
class PendingTurnState:
    source: str | None = None
    drawn_value: int | None = None
    keep_drawn: bool | None = None


class SkyjoDecisionEnv:
    """
    Decision-granularity Skyjo environment.

    Main-phase turns are decomposed into:
    - choose source (deck/discard)
    - keep vs discard (if deck)
    - choose position
    """

    def __init__(
        self,
        num_players: int = 2,
        seed: int = 0,
        history_window_k: int = 16,
        score_limit: int = 100,
        setup_mode: str = "auto",
    ) -> None:
        self.base = SkyjoEnv(
            num_players=num_players,
            seed=seed,
            history_window_k=history_window_k,
            score_limit=score_limit,
            setup_mode=setup_mode,
        )
        self.pending = PendingTurnState()
        self.decision_phase = DecisionPhase.CHOOSE_SOURCE

    @property
    def num_players(self) -> int:
        return self.base.num_players

    @property
    def current_player(self) -> int:
        return self.base.current_player

    @property
    def scores(self) -> list[int]:
        return self.base.scores

    @property
    def phase(self) -> TurnPhase:
        return self.base.phase

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        obs = self.base.reset(seed=seed)
        self.pending = PendingTurnState()
        self._sync_decision_phase()
        return self.observe(self.current_player)

    def _sync_decision_phase(self) -> None:
        if self.base.phase == TurnPhase.SETUP:
            self.decision_phase = DecisionPhase.SETUP_REVEAL
        elif self.base.phase in (TurnPhase.MAIN, TurnPhase.ROUND_END):
            if self.pending.source is None:
                self.decision_phase = DecisionPhase.CHOOSE_SOURCE
            elif self.pending.source == "DECK" and self.pending.keep_drawn is None:
                self.decision_phase = DecisionPhase.KEEP_OR_DISCARD
            else:
                self.decision_phase = DecisionPhase.CHOOSE_POSITION

    def legal_actions(self) -> list[int]:
        if self.base.game_over:
            return []

        board = self.base.boards[self.current_player]
        if self.decision_phase == DecisionPhase.SETUP_REVEAL:
            legal = []
            for pos in range(BOARD_SIZE):
                if not board.removed[pos] and not board.visible[pos]:
                    legal.append(DecisionAction.CHOOSE_POS_BASE + pos)
            return legal

        if self.decision_phase == DecisionPhase.CHOOSE_SOURCE:
            legal = [int(DecisionAction.CHOOSE_DECK)]
            if self.base.discard_pile:
                legal.append(int(DecisionAction.CHOOSE_DISCARD))
            return legal

        if self.decision_phase == DecisionPhase.KEEP_OR_DISCARD:
            return [int(DecisionAction.KEEP_DRAWN), int(DecisionAction.DISCARD_DRAWN)]

        if self.decision_phase == DecisionPhase.CHOOSE_POSITION:
            legal: list[int] = []
            if self.pending.source == "DECK" and self.pending.keep_drawn is False:
                for pos in range(BOARD_SIZE):
                    if not board.removed[pos] and not board.visible[pos]:
                        legal.append(DecisionAction.CHOOSE_POS_BASE + pos)
            else:
                for pos in range(BOARD_SIZE):
                    if not board.removed[pos]:
                        legal.append(DecisionAction.CHOOSE_POS_BASE + pos)
            return legal

        return []

    def observe(self, observer_player: int, history_k: int | None = None) -> dict[str, Any]:
        obs = self.base.observe(observer_player, history_k=history_k)
        drawn_value = self.pending.drawn_value if self.pending.drawn_value is not None else UNKNOWN_VALUE
        obs["decision_phase"] = self.decision_phase.name
        obs["decision_phase_id"] = int(self.decision_phase)
        obs["current_drawn_value"] = int(drawn_value)
        obs["tokens"]["decision_token"] = [int(self.decision_phase), int(drawn_value)]
        # Reuse global phase slot for decision phase id in this mode.
        obs["tokens"]["global_token"][3] = int(self.decision_phase)
        return obs

    def step(self, action_id: int) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        legal = self.legal_actions()
        if action_id not in legal:
            raise ValueError(
                f"Illegal decision action {action_id} for player {self.current_player} in phase {self.decision_phase.name}"
            )
        if self.base.game_over:
            raise RuntimeError("Game is already over. Call reset() to start a new game.")

        actor = self.current_player
        self.base.global_step += 1
        macro_action = "INTERMEDIATE_DECISION"
        position = -1

        if self.decision_phase == DecisionPhase.SETUP_REVEAL:
            pos = action_id - int(DecisionAction.CHOOSE_POS_BASE)
            _, rewards, terminated, info = self.base.step(SETUP_FLIP_BASE + pos)
            self.pending = PendingTurnState()
            self._sync_decision_phase()
            info["decision_phase"] = self.decision_phase.name
            return self.observe(self.current_player), rewards, terminated, info

        if self.decision_phase == DecisionPhase.CHOOSE_SOURCE:
            if action_id == int(DecisionAction.CHOOSE_DISCARD):
                self.pending.source = "DISCARD"
                self.pending.drawn_value = self.base.discard_pile.pop()
                self.pending.keep_drawn = True
                self.decision_phase = DecisionPhase.CHOOSE_POSITION
                macro_action = "CHOOSE_SOURCE_DISCARD"
            else:
                self.pending.source = "DECK"
                self.pending.drawn_value = self.base._draw_from_deck()
                self.pending.keep_drawn = None
                self.decision_phase = DecisionPhase.KEEP_OR_DISCARD
                macro_action = "CHOOSE_SOURCE_DECK"

        elif self.decision_phase == DecisionPhase.KEEP_OR_DISCARD:
            if action_id == int(DecisionAction.KEEP_DRAWN):
                self.pending.keep_drawn = True
                macro_action = "KEEP_DRAWN"
            else:
                self.pending.keep_drawn = False
                if self.pending.drawn_value is None:
                    raise RuntimeError("Missing drawn card value during DISCARD decision.")
                self.base.discard_pile.append(self.pending.drawn_value)
                macro_action = "DISCARD_DRAWN"
            self.decision_phase = DecisionPhase.CHOOSE_POSITION

        elif self.decision_phase == DecisionPhase.CHOOSE_POSITION:
            pos = action_id - int(DecisionAction.CHOOSE_POS_BASE)
            position = pos
            board = self.base.boards[actor]
            drawn = self.pending.drawn_value
            if drawn is None:
                raise RuntimeError("Missing drawn card value before CHOOSE_POSITION.")

            if self.pending.source == "DISCARD" or self.pending.keep_drawn:
                replaced = board.cards[pos]
                board.cards[pos] = drawn
                board.visible[pos] = True
                if replaced is not None:
                    self.base.discard_pile.append(replaced)
                if self.pending.source == "DISCARD":
                    macro_action = "TAKE_DISCARD_AND_REPLACE"
                    action_type = PublicActionType.TAKE_DISCARD_AND_REPLACE
                    source = 1
                else:
                    macro_action = "DRAW_DECK_KEEP_AND_REPLACE"
                    action_type = PublicActionType.DRAW_DECK_KEEP_AND_REPLACE
                    source = 0
                self.base._record_action(
                    actor=actor,
                    action_type=action_type,
                    source=source,
                    target_pos=pos,
                    a_value=drawn,
                    b_value=replaced if replaced is not None else ACTION_HISTORY_UNKNOWN,
                    note="decision-mode replace",
                )
            else:
                board.visible[pos] = True
                revealed = board.cards[pos]
                macro_action = "DRAW_DECK_DISCARD_AND_FLIP"
                self.base._record_action(
                    actor=actor,
                    action_type=PublicActionType.DRAW_DECK_DISCARD_AND_FLIP,
                    source=0,
                    target_pos=pos,
                    a_value=drawn,
                    b_value=revealed if revealed is not None else ACTION_HISTORY_UNKNOWN,
                    note="decision-mode flip",
                )

            self.base._resolve_columns(actor)
            self.base._advance_main_turn(actor)
            self.pending = PendingTurnState()
            self._sync_decision_phase()

        rewards = {f"player_{p}": 0.0 for p in range(self.num_players)}
        if self.base.phase == TurnPhase.GAME_OVER:
            winner = min(range(self.num_players), key=lambda p: self.scores[p])
            for p in range(self.num_players):
                rewards[f"player_{p}"] = 1.0 if p == winner else -1.0
        elif self.base._last_round_ended():
            for p in range(self.num_players):
                rewards[f"player_{p}"] = -float(self.base.round_scores[p])

        info = {
            "actor": actor,
            "action_id": action_id,
            "macro_action": macro_action,
            "position": position,
            "decision_phase": self.decision_phase.name,
            "phase": self.base.phase.name,
            "legal_actions_next": self.legal_actions(),
            "round_index": self.base.round_index,
            "scores": list(self.scores),
            "round_scores": list(self.base.round_scores),
            "drawn_value": self.pending.drawn_value if self.pending.drawn_value is not None else UNKNOWN_VALUE,
        }
        terminated = self.base.phase == TurnPhase.GAME_OVER
        return self.observe(self.current_player), rewards, terminated, info
