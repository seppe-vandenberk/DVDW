from contextlib import nullcontext
from copy import deepcopy

from fastapi.testclient import TestClient
import torch

from connect4.api import create_app
from connect4.game import empty_board, drop
from connect4.simulation_service import create_app as create_viewer


def test_inspection_is_read_only_masks_columns_and_reports_search(tmp_path):
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        trainer = client.app.state.trainer
        agent = trainer.agent("shared")
        before = deepcopy(agent.online.state_dict())
        counters = agent.updates, agent.observations, agent.decisions, agent.rng.getstate()
        board = empty_board()
        board[0] = [1, -1, 1, -1, 1]
        response = client.post("/debug/analyze", json={"board": board, "player": -1, "simulations": 32})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["trains"] is False and data["column"] != 0
        assert data["columns"][0] == {"column": 0, "legal": False, "prior": 0, "visits": 0, "value": None}
        assert sum(c["visits"] for c in data["columns"]) == 32
        assert abs(sum(c["prior"] for c in data["columns"]) - 1) < 1e-6
        assert not trainer.games and trainer.workers["shared"].incoming.empty()
        assert counters == (agent.updates, agent.observations, agent.decisions, agent.rng.getstate())
        assert all(torch.equal(before[k], v) for k, v in agent.online.state_dict().items())


def test_debug_validation_and_unknown_models_do_not_create_learners(tmp_path):
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        trainer = client.app.state.trainer
        assert client.get("/debug/state/not-loaded").status_code == 404
        assert "not-loaded" not in trainer.agents
        payload = {"board": empty_board(), "player": -1, "simulations": 32}
        assert client.post("/debug/analyze", json=payload).status_code == 409
        payload.update(player=1, simulations=9999)
        assert client.post("/debug/analyze", json=payload).status_code == 422
        payload.update(simulations=32, board=[[0]])
        assert client.post("/debug/analyze", json=payload).status_code == 422


def test_debug_controls_are_non_destructive_and_audited(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        worker = client.app.state.trainer.workers["shared"]
        assert client.post("/debug/control", json={"action": "pause"}).status_code == 409
        monkeypatch.setattr(worker.thread, "is_alive", lambda: True)
        for action in ("pause", "resume", "evaluate", "checkpoint"):
            response = client.post("/debug/control", json={"action": action})
            assert response.status_code == 200, response.text
        snapshot = client.get("/debug/state/shared").json()
        assert snapshot["background"]["evaluation_requested"]
        assert snapshot["background"]["checkpoint_requested"]
        assert not snapshot["background"]["pause_requested"]
        assert len(snapshot["background"]["events"]) == 4
        assert client.post("/debug/control", json={"action": "reset"}).status_code == 422


def test_debug_sessions_show_pending_confirmations(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SEARCH_SIMULATIONS", "8")
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        assert client.post("/move", json={"game_id": "camera-debug", "board": empty_board(), "player": 1}).status_code == 200
        snapshot = client.get("/debug/state/shared").json()
        session = snapshot["sessions"][0]
        assert session["game_id"] == "camera-debug" and session["pending_players"] == [1]
        assert session["bindings"]["1"]["train"]
        assert snapshot["requests"][-1]["path"] == "/move"
        assert client.delete("/games/camera-debug").status_code == 200
        events = client.get("/debug/state/shared").json()["background"]["events"]
        assert events[-1]["kind"] == "session" and events[-1]["game_id"] == "camera-debug"


def test_debug_viewer_proxies_api_and_serves_assets(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        with TestClient(create_viewer(run_worker=False)) as viewer:
            assert viewer.get("/debug").status_code == 200
            assert "What is the AI doing?" in viewer.get("/debug").text
            assert viewer.get("/static/debug.js").status_code == 200
            assert viewer.get("/static/debug.css").status_code == 200
            assert viewer.get("/debug/data").json()["architecture"] == "policy-value-7x5-v1"
            assert viewer.get("/debug/data?model=../bad").status_code == 422
            assert viewer.post("/debug/analyze", json={"board":empty_board(),"player":1,"simulations":8}).status_code == 200
            assert viewer.post("/debug/control", json={"action":"pause"}).status_code == 409
