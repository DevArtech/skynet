from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent_service import AgentService
from .schemas import (
    AgentDecisionContext,
    AgentStepResponse,
    ActionRequest,
    ActionResponse,
    AgentType,
    AutoPlayRequest,
    AutoPlayResponse,
    CreateGameRequest,
    GameStateResponse,
    InferActionRequest,
    InferAgentStepRequest,
    LegalActionView,
    SessionResponse,
    StateRequest,
)
from .state_codec import dump_env_state, restore_env_state
from skyjo_decision_env import DecisionAction, DecisionPhase, PendingTurnState, SkyjoDecisionEnv
from skyjo_env import SkyjoEnv, action_to_macro


app = FastAPI(title="Skyjo Web API", version="0.1.0")
agent_service = AgentService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _decision_step_log(phase: DecisionPhase, action_id: int) -> str:
    if phase == DecisionPhase.SETUP_REVEAL:
        pos = action_id - int(DecisionAction.CHOOSE_POS_BASE)
        return f"setup: choose position {pos}"
    if phase == DecisionPhase.CHOOSE_SOURCE:
        return "choose source: discard" if action_id == int(DecisionAction.CHOOSE_DISCARD) else "choose source: deck"
    if phase == DecisionPhase.KEEP_OR_DISCARD:
        return "decision: keep drawn card" if action_id == int(DecisionAction.KEEP_DRAWN) else "decision: discard drawn card"
    if phase == DecisionPhase.CHOOSE_POSITION:
        pos = action_id - int(DecisionAction.CHOOSE_POS_BASE)
        return f"choose position: {pos}"
    return f"decision action {action_id}"


def _build_view(env: SkyjoEnv, observer_player: int, info: dict[str, Any] | None = None) -> GameStateResponse:
    obs = env.observe(observer_player)
    legal_actions: list[LegalActionView] = []
    for action_id in env.legal_actions():
        macro, position = action_to_macro(action_id)
        legal_actions.append(
            LegalActionView(
                action_id=action_id,
                macro=macro,
                position=position,
                label=f"{macro} @ {position}",
            )
        )
    return GameStateResponse(
        terminated=env.game_over,
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
        public_history=[
            {
                "step_index": int(action.step_index),
                "round_index": int(action.round_index),
                "actor": int(action.actor),
                "action_type": action.action_type.name,
                "source": int(action.source),
                "target_pos": int(action.target_pos),
                "a_value": int(action.a_value),
                "b_value": int(action.b_value),
                "note": str(action.note),
            }
            for action in env.public_history
        ],
        info=info or {},
    )


@app.post("/api/session/new", response_model=SessionResponse)
def create_game(request: CreateGameRequest) -> SessionResponse:
    try:
        env = SkyjoEnv(
            num_players=request.num_players,
            seed=request.seed,
            history_window_k=request.history_window_k,
            score_limit=request.score_limit,
            setup_mode=request.setup_mode,
        )
        env.reset(seed=request.seed)
        observer = max(0, min(request.num_players - 1, request.observer_player))
        return SessionResponse(state=dump_env_state(env), view=_build_view(env, observer_player=observer))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/session/view", response_model=SessionResponse)
def get_game_state(request: StateRequest) -> SessionResponse:
    try:
        env = restore_env_state(request.state)
        observer = env.current_player if request.observer_player is None else request.observer_player
        return SessionResponse(state=dump_env_state(env), view=_build_view(env, observer_player=observer))
    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error


@app.post("/api/session/legal-actions", response_model=list[int])
def get_legal_actions(request: StateRequest) -> list[int]:
    try:
        env = restore_env_state(request.state)
        return env.legal_actions()
    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error


@app.post("/api/session/apply-action", response_model=ActionResponse)
def apply_action(request: ActionRequest) -> ActionResponse:
    try:
        env = restore_env_state(request.state)
        _, _, _, info = env.step(request.action_id)
        observer = env.current_player if request.observer_player is None else request.observer_player
        return ActionResponse(
            state=dump_env_state(env),
            view=_build_view(env, observer_player=observer, info=info),
            ai_action=None,
        )
    except (KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/session/infer-action", response_model=ActionResponse)
def step_ai(request: InferActionRequest) -> ActionResponse:
    try:
        env = restore_env_state(request.state)
        legal_actions = env.legal_actions()
        observation = env.observe(env.current_player)
        ai_action = agent_service.select_action(request.agent, observation, legal_actions, env=env)
        _, _, _, info = env.step(ai_action.action_id)
        observer = env.current_player if request.observer_player is None else request.observer_player
        return ActionResponse(
            state=dump_env_state(env),
            view=_build_view(env, observer_player=observer, info=info),
            ai_action=ai_action,
        )
    except (KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/session/infer-agent-step", response_model=AgentStepResponse)
def infer_agent_step(request: InferAgentStepRequest) -> AgentStepResponse:
    try:
        env = restore_env_state(request.state)
        decision_env = SkyjoDecisionEnv(
            num_players=env.num_players,
            seed=env.initial_seed,
            history_window_k=env.history_window_k,
            score_limit=env.score_limit,
            setup_mode=env.setup_mode,
        )
        decision_env.base = env
        if request.decision_context is not None:
            decision_env.pending = PendingTurnState(
                source=request.decision_context.pending_source,
                drawn_value=request.decision_context.pending_drawn_value,
                keep_drawn=request.decision_context.pending_keep_drawn,
            )
            decision_env.decision_phase = DecisionPhase(request.decision_context.decision_phase_id)
        else:
            decision_env.pending = PendingTurnState()
            decision_env._sync_decision_phase()

        actor_before = decision_env.current_player
        phase_before = decision_env.decision_phase
        observation = decision_env.observe(actor_before)
        legal_actions = decision_env.legal_actions()
        ai_step = agent_service.select_decision_action(request.agent, observation, legal_actions)
        decision_action = int(ai_step.action_id)
        step_log = _decision_step_log(phase_before, decision_action)

        _, _, _, info = decision_env.step(decision_action)
        turn_completed = decision_env.current_player != actor_before
        observer = decision_env.current_player if request.observer_player is None else request.observer_player

        if turn_completed:
            next_context = None
        else:
            next_context = AgentDecisionContext(
                decision_phase_id=int(decision_env.decision_phase),
                pending_source=decision_env.pending.source,
                pending_drawn_value=decision_env.pending.drawn_value,
                pending_keep_drawn=decision_env.pending.keep_drawn,
            )

        return AgentStepResponse(
            state=dump_env_state(decision_env.base),
            view=_build_view(decision_env.base, observer_player=observer, info=info),
            decision_context=next_context,
            step_log=step_log,
            turn_completed=turn_completed,
        )
    except (KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/session/autoplay", response_model=AutoPlayResponse)
def autoplay(request: AutoPlayRequest) -> AutoPlayResponse:
    try:
        env = restore_env_state(request.state)
        actions = []
        for _ in range(request.max_steps):
            if env.game_over:
                break
            actor = env.current_player
            if actor < 0 or actor >= len(request.agent_by_player):
                raise ValueError("agent_by_player does not cover current player.")
            agent = request.agent_by_player[actor]
            if request.stop_on_human_turn and agent.type == AgentType.HUMAN:
                break
            if agent.type == AgentType.HUMAN:
                raise ValueError("Autoplay reached a human seat while stop_on_human_turn is false.")
            legal_actions = env.legal_actions()
            observation = env.observe(actor)
            ai_action = agent_service.select_action(agent, observation, legal_actions, env=env)
            _, _, _, _ = env.step(ai_action.action_id)
            actions.append(ai_action)
        observer = env.current_player if request.observer_player is None else request.observer_player
        return AutoPlayResponse(
            state=dump_env_state(env),
            view=_build_view(env, observer_player=observer),
            steps_executed=len(actions),
            actions=actions,
        )
    except (KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
