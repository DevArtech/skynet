from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GameMode(str, Enum):
    HUMAN_VS_AI = "human_vs_ai"
    AI_VS_AI = "ai_vs_ai"


class AgentType(str, Enum):
    HUMAN = "human"
    BASELINE = "baseline"
    BELIEF = "belief"


class AgentConfig(BaseModel):
    type: AgentType
    checkpoint_path: str | None = None
    simulations: int = Field(default=32, ge=1, le=400)
    device: str = "cpu"
    ablate_belief_head: bool = False


class CreateGameRequest(BaseModel):
    num_players: int = Field(default=2, ge=2, le=8)
    seed: int = 0
    setup_mode: str = "auto"
    history_window_k: int = Field(default=16, ge=1, le=128)
    score_limit: int = Field(default=100, ge=10, le=1000)
    observer_player: int = 0


class StateRequest(BaseModel):
    state: dict[str, Any]
    observer_player: int | None = None


class ActionRequest(StateRequest):
    action_id: int


class AIActionResult(BaseModel):
    actor: int
    action_id: int
    macro_action: str
    position: int
    root_value: float | None = None
    visit_counts: dict[int, int] | None = None
    q_values: dict[int, float] | None = None
    policy_target: dict[int, float] | None = None


class AutoPlayRequest(BaseModel):
    state: dict[str, Any]
    agent_by_player: list[AgentConfig]
    max_steps: int = Field(default=50, ge=1, le=1000)
    observer_player: int | None = None
    stop_on_human_turn: bool = True


class InferActionRequest(StateRequest):
    agent: AgentConfig


class AgentDecisionContext(BaseModel):
    decision_phase_id: int
    pending_source: str | None = None
    pending_drawn_value: int | None = None
    pending_keep_drawn: bool | None = None


class InferAgentStepRequest(StateRequest):
    agent: AgentConfig
    decision_context: AgentDecisionContext | None = None


class LegalActionView(BaseModel):
    action_id: int
    macro: str
    position: int
    label: str


class GameStateResponse(BaseModel):
    terminated: bool
    phase: str
    current_player: int
    observer_player: int
    scores: list[int]
    round_scores: list[int]
    deck_size: int
    discard_top: int
    discard_size: int
    legal_actions: list[LegalActionView]
    board_views: dict[str, dict[str, list[int]]]
    public_history: list[dict[str, Any]]
    info: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    state: dict[str, Any]
    view: GameStateResponse


class ActionResponse(SessionResponse):
    ai_action: AIActionResult | None = None


class AutoPlayResponse(SessionResponse):
    steps_executed: int
    actions: list[AIActionResult]


class AgentStepResponse(SessionResponse):
    decision_context: AgentDecisionContext | None = None
    step_log: str
    turn_completed: bool
