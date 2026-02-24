from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from web.backend.main import app


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
