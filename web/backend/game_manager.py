from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from skyjo_env import SkyjoEnv, action_to_macro

from .schemas import AIActionResult, AgentConfig, AgentType, CreateGameRequest, GameMode, GameStateResponse, LegalActionView


class AgentSelector(Protocol):
    def select_action(self, session: "GameSession", observation: dict[str, Any], legal_actions: list[int]) -> AIActionResult:
        pass


@dataclass
class GameSession:
    game_id: str
    mode: GameMode
    env: SkyjoEnv
    agents: list[AgentConfig]
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_info: dict[str, Any] = field(default_factory=dict)


class GameManager:
    def __init__(self, selector: AgentSelector) -> None:
        self._selector = selector
        self._sessions: dict[str, GameSession] = {}
        self._sessions_lock = threading.Lock()

    def create_game(self, request: CreateGameRequest) -> GameSession:
        if len(request.agents) != request.num_players:
            raise ValueError("agents length must match num_players.")
        if request.mode == GameMode.HUMAN_VS_AI and sum(1 for a in request.agents if a.type == AgentType.HUMAN) != 1:
            raise ValueError("human_vs_ai mode requires exactly one human seat.")
        if request.mode == GameMode.AI_VS_AI and any(a.type == AgentType.HUMAN for a in request.agents):
            raise ValueError("ai_vs_ai mode does not allow human seats.")

        env = SkyjoEnv(
            num_players=request.num_players,
            seed=request.seed,
            history_window_k=request.history_window_k,
            score_limit=request.score_limit,
            setup_mode=request.setup_mode,
        )
        env.reset(seed=request.seed)

        session = GameSession(
            game_id=uuid.uuid4().hex,
            mode=request.mode,
            env=env,
            agents=request.agents,
        )
        with self._sessions_lock:
            self._sessions[session.game_id] = session
        return session

    def get_session(self, game_id: str) -> GameSession:
        with self._sessions_lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise KeyError(f"Unknown game_id '{game_id}'.")
        return session

    def get_state(self, game_id: str, observer_player: int | None = None) -> GameStateResponse:
        session = self.get_session(game_id)
        with session.lock:
            observer = session.env.current_player if observer_player is None else observer_player
            return self._build_state(session, observer_player=observer)

    def apply_action(self, game_id: str, action_id: int, observer_player: int | None = None) -> GameStateResponse:
        session = self.get_session(game_id)
        with session.lock:
            actor = session.env.current_player
            if session.agents[actor].type != AgentType.HUMAN:
                raise ValueError("Current seat is controlled by AI. Use /step-ai for AI turns.")
            _, _, _, info = session.env.step(action_id)
            session.last_info = info
            observer = session.env.current_player if observer_player is None else observer_player
            return self._build_state(session, observer_player=observer)

    def step_ai(self, game_id: str, observer_player: int | None = None) -> tuple[GameStateResponse, AIActionResult]:
        session = self.get_session(game_id)
        with session.lock:
            actor = session.env.current_player
            if session.agents[actor].type == AgentType.HUMAN:
                raise ValueError("Current seat is human-controlled. Use /actions for human turns.")
            legal_actions = session.env.legal_actions()
            observation = session.env.observe(actor)
            ai_result = self._selector.select_action(session, observation, legal_actions)
            _, _, _, info = session.env.step(ai_result.action_id)
            session.last_info = info
            observer = session.env.current_player if observer_player is None else observer_player
            return self._build_state(session, observer_player=observer), ai_result

    def autoplay(
        self,
        game_id: str,
        max_steps: int,
        observer_player: int | None = None,
        stop_on_human_turn: bool = True,
    ) -> tuple[GameStateResponse, list[AIActionResult]]:
        session = self.get_session(game_id)
        actions: list[AIActionResult] = []
        with session.lock:
            for _ in range(max_steps):
                if session.env.game_over:
                    break
                actor = session.env.current_player
                if stop_on_human_turn and session.agents[actor].type == AgentType.HUMAN:
                    break
                if session.agents[actor].type == AgentType.HUMAN:
                    raise ValueError("autoplay hit a human seat while stop_on_human_turn is false.")
                legal_actions = session.env.legal_actions()
                observation = session.env.observe(actor)
                ai_result = self._selector.select_action(session, observation, legal_actions)
                _, _, _, info = session.env.step(ai_result.action_id)
                session.last_info = info
                actions.append(ai_result)

            observer = session.env.current_player if observer_player is None else observer_player
            return self._build_state(session, observer_player=observer), actions

    def _build_state(self, session: GameSession, observer_player: int) -> GameStateResponse:
        obs = session.env.observe(observer_player)
        legal_actions = [self._legal_action_view(action_id) for action_id in session.env.legal_actions()]
        return GameStateResponse(
            game_id=session.game_id,
            mode=session.mode,
            terminated=session.env.game_over,
            phase=obs["phase"],
            current_player=obs["current_player"],
            observer_player=observer_player,
            scores=obs["scores"],
            round_scores=obs["round_scores"],
            deck_size=obs["deck_size"],
            discard_top=obs["discard_top"],
            discard_size=len(obs["discard_pile"]),
            legal_actions=legal_actions,
            board_views={str(k): v for k, v in obs["board_views"].items()},
            public_history=obs["history"],
            info=session.last_info,
        )

    def _legal_action_view(self, action_id: int) -> LegalActionView:
        macro, position = action_to_macro(action_id)
        label = f"{macro} @ {position}"
        return LegalActionView(action_id=action_id, macro=macro, position=position, label=label)
