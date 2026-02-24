from __future__ import annotations

import base64
import pickle
from typing import Any

from skyjo_env import PlayerBoard, PublicAction, PublicActionType, SkyjoEnv, TurnPhase


def dump_env_state(env: SkyjoEnv) -> dict[str, Any]:
    rng_state_b64 = base64.b64encode(pickle.dumps(env.rng.getstate())).decode("ascii")
    return {
        "initial_seed": int(env.initial_seed),
        "rng_state_b64": rng_state_b64,
        "num_players": int(env.num_players),
        "history_window_k": int(env.history_window_k),
        "score_limit": int(env.score_limit),
        "setup_mode": str(env.setup_mode),
        "manual_initial_reveals": bool(env.manual_initial_reveals),
        "round_index": int(env.round_index),
        "global_step": int(env.global_step),
        "turns_in_round": int(env.turns_in_round),
        "phase": env.phase.name,
        "current_player": int(env.current_player),
        "scores": [int(v) for v in env.scores],
        "round_scores": [int(v) for v in env.round_scores],
        "deck": [int(v) for v in env.deck],
        "discard_pile": [int(v) for v in env.discard_pile],
        "boards": [
            {
                "cards": [None if value is None else int(value) for value in board.cards],
                "visible": [bool(v) for v in board.visible],
                "removed": [bool(v) for v in board.removed],
            }
            for board in env.boards
        ],
        "setup_reveals_remaining": [int(v) for v in env.setup_reveals_remaining],
        "pending_final_turn_players": sorted(int(v) for v in env.pending_final_turn_players),
        "round_ender": None if env.round_ender is None else int(env.round_ender),
        "game_over": bool(env.game_over),
        "round_history_start_index": int(env.round_history_start_index),
        "public_history": [
            {
                "step_index": int(a.step_index),
                "round_index": int(a.round_index),
                "actor": int(a.actor),
                "action_type": int(a.action_type),
                "source": int(a.source),
                "target_pos": int(a.target_pos),
                "a_value": int(a.a_value),
                "b_value": int(a.b_value),
                "note": str(a.note),
            }
            for a in env.public_history
        ],
    }


def restore_env_state(state: dict[str, Any]) -> SkyjoEnv:
    env = SkyjoEnv(
        num_players=int(state["num_players"]),
        seed=int(state["initial_seed"]),
        history_window_k=int(state["history_window_k"]),
        score_limit=int(state["score_limit"]),
        setup_mode=str(state["setup_mode"]),
        manual_initial_reveals=bool(state["manual_initial_reveals"]),
    )
    env.round_index = int(state["round_index"])
    env.global_step = int(state["global_step"])
    env.turns_in_round = int(state["turns_in_round"])
    env.phase = TurnPhase[str(state["phase"])]
    env.current_player = int(state["current_player"])
    env.scores = [int(v) for v in state["scores"]]
    env.round_scores = [int(v) for v in state["round_scores"]]
    env.deck = [int(v) for v in state["deck"]]
    env.discard_pile = [int(v) for v in state["discard_pile"]]
    env.boards = [
        PlayerBoard(
            cards=[None if value is None else int(value) for value in board["cards"]],
            visible=[bool(v) for v in board["visible"]],
            removed=[bool(v) for v in board["removed"]],
        )
        for board in state["boards"]
    ]
    env.setup_reveals_remaining = [int(v) for v in state["setup_reveals_remaining"]]
    env.pending_final_turn_players = {int(v) for v in state["pending_final_turn_players"]}
    env.round_ender = None if state["round_ender"] is None else int(state["round_ender"])
    env.game_over = bool(state["game_over"])
    env.round_history_start_index = int(state["round_history_start_index"])
    env.public_history = [
        PublicAction(
            step_index=int(a["step_index"]),
            round_index=int(a["round_index"]),
            actor=int(a["actor"]),
            action_type=PublicActionType(int(a["action_type"])),
            source=int(a["source"]),
            target_pos=int(a["target_pos"]),
            a_value=int(a["a_value"]),
            b_value=int(a["b_value"]),
            note=str(a.get("note", "")),
        )
        for a in state["public_history"]
    ]
    rng_state_b64 = state.get("rng_state_b64")
    if isinstance(rng_state_b64, str) and rng_state_b64:
        env.rng.setstate(pickle.loads(base64.b64decode(rng_state_b64.encode("ascii"))))
    return env
