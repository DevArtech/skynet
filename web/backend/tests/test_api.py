from pathlib import Path
import sys

from fastapi.testclient import TestClient
from skyjo_env import SkyjoEnv, TurnPhase

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from web.backend.main import app
from web.backend.state_codec import dump_env_state, restore_env_state


client = TestClient(app)


def test_create_game_and_apply_human_action() -> None:
    create = client.post(
        "/api/session/new",
        json={
            "num_players": 2,
            "seed": 7,
            "setup_mode": "manual",
            "observer_player": 0,
        },
    )
    assert create.status_code == 200
    session = create.json()
    assert "state" in session
    assert "view" in session
    assert session["view"]["current_player"] == 0
    assert session["view"]["phase"] == "SETUP"
    assert session["view"]["legal_actions"]

    legal = client.post("/api/session/legal-actions", json={"state": session["state"]})
    assert legal.status_code == 200
    legal_actions = legal.json()
    assert isinstance(legal_actions, list)
    assert legal_actions

    apply_action = client.post(
        "/api/session/apply-action",
        json={"state": session["state"], "action_id": legal_actions[0], "observer_player": 0},
    )
    assert apply_action.status_code == 200
    updated = apply_action.json()
    assert "state" in updated
    assert "view" in updated
    assert "legal_actions" in updated["view"]


def test_state_codec_preserves_rng_state() -> None:
    env = SkyjoEnv(num_players=2, seed=13, setup_mode="auto")
    env.reset()
    state = dump_env_state(env)
    restored = restore_env_state(state)

    # RNG continuity must match across serialization boundaries.
    assert env.rng.random() == restored.rng.random()


def test_infer_agent_step_with_heuristic_agent() -> None:
    create = client.post(
        "/api/session/new",
        json={
            "num_players": 2,
            "seed": 11,
            "setup_mode": "manual",
            "observer_player": 0,
        },
    )
    assert create.status_code == 200
    session = create.json()

    step = client.post(
        "/api/session/infer-agent-step",
        json={
            "state": session["state"],
            "observer_player": 0,
            "agent": {
                "type": "heuristic",
                "heuristic_bot_name": "greedy_value_replacement",
                "heuristic_bot_epsilon": 0.0,
            },
            "decision_context": None,
        },
    )
    assert step.status_code == 200
    payload = step.json()
    assert "state" in payload
    assert "view" in payload
    assert "step_log" in payload


def test_column_clear_only_applies_once_per_player_per_round() -> None:
    env = SkyjoEnv(num_players=2, seed=3, setup_mode="manual")
    env.reset()
    env.phase = TurnPhase.MAIN
    env.current_player = 0
    board = env.boards[0]

    # Prepare two matching visible columns for the same player.
    for pos in range(12):
        board.cards[pos] = 9
        board.visible[pos] = False
        board.removed[pos] = False

    for pos in (0, 4, 8):
        board.cards[pos] = 2
        board.visible[pos] = True
    for pos in (1, 5, 9):
        board.cards[pos] = 7
        board.visible[pos] = True

    env._resolve_columns(0)
    assert env.column_clear_used_this_round[0] is True
    assert all(board.removed[pos] for pos in (0, 4, 8))
    assert all(not board.removed[pos] for pos in (1, 5, 9))

    # Further checks in the same round should not clear another column.
    env._resolve_columns(0)
    assert all(not board.removed[pos] for pos in (1, 5, 9))
