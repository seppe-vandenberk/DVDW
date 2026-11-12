from contextlib import nullcontext
from threading import Event, Thread

from fastapi.testclient import TestClient
import httpx
import pytest

from connect4.api import create_app as create_api
from connect4.game import empty_board, outcome, validate_board
from connect4.simulation_service import Simulation, create_app
from connect4.simulator import play_game


def test_viewer_and_snapshot_are_read_only():
    app = create_app(run_worker=False)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "Can you beat the AI?" in client.get("/").text
        assert client.get("/static/viewer.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/play.js").status_code == 200
        assert client.get("/static/play.css").status_code == 200
        assert "Play &amp; exhibit" not in client.get("/").text
        assert 'id="simulation-config"' not in client.get("/").text
        assert client.get("/health").json()["simulation"] == "starting"
        assert client.get("/state").json()["board"] == empty_board()
        snapshot = app.state.simulation.snapshot()
        snapshot["board"][0][-1] = 1
        assert client.get("/state").json()["board"][0][-1] == 0


def test_live_frames_are_real_simulator_moves(tmp_path):
    frames = []
    with TestClient(create_api(tmp_path, device="cpu")) as client:
        winner = play_game(client, interval=0, on_update=frames.append)
    assert frames[0]["board"] == empty_board()
    assert frames[0]["move_count"] == 0
    for previous, frame in zip(frames, frames[1:]):
        validate_board(frame["board"])
        assert frame["move_count"] == previous["move_count"] + 1
        changes = [(c, r) for c in range(7) for r in range(5)
                   if previous["board"][c][r] != frame["board"][c][r]]
        assert len(changes) == 1
        assert changes[0][0] == frame["last_column"]
    assert frames[-1]["done"]
    assert frames[-1]["winner"] == outcome(frames[-1]["board"]) == winner
    assert frames[-1]["next_player"] is None


def test_worker_records_scores_and_finishes_cleanly(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    simulation = Simulation("http://testserver", interval=0, result_delay=0)
    with TestClient(create_api(tmp_path, device="cpu")) as client:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: client)
        original_update = simulation.update

        def update(changes):
            original_update(changes)
            if changes.get("status") == "finished":
                simulation.stop.set()

        monkeypatch.setattr(simulation, "update", update)
        simulation.run()
        state = simulation.snapshot()
        assert state["status"] == "stopped"
        assert state["games"] == 1
        assert sum(state["scores"].values()) == 1
        assert not client.app.state.trainer.games


def test_cancel_discards_incomplete_game(tmp_path):
    stop = Event()
    with TestClient(create_api(tmp_path, device="cpu")) as client:
        def update(frame):
            if frame["move_count"] == 1:
                stop.set()
        with pytest.raises(InterruptedError):
            play_game(client, interval=0, on_update=update, stop_event=stop)
        assert not client.app.state.trainer.games
        assert not client.app.state.trainer.agents["shared"].memory


def test_worker_exposes_connection_error(monkeypatch):
    import connect4.simulation_service as module
    simulation = Simulation("http://unavailable", interval=0)
    simulation.curriculum = False

    def failure(*args, **kwargs):
        simulation.stop.set()
        raise httpx.ConnectError("API offline")

    monkeypatch.setattr(module, "play_game", failure)
    monkeypatch.setattr(module, "post", lambda *args: {})
    simulation.run()
    assert "API offline" in simulation.snapshot()["error"]
    assert simulation.snapshot()["games"] == 0


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_bad_simulation_interval_rejected(interval):
    with pytest.raises(ValueError):
        Simulation("http://api", interval=interval)


@pytest.fixture
def human_service(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    with TestClient(create_api(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        with TestClient(create_app(run_worker=False)) as viewer:
            yield viewer, api.app.state.trainer


def test_human_win_trains_bot_and_stale_moves_are_rejected(human_service):
    viewer, trainer = human_service
    state = viewer.post("/mode", json={"mode": "human"}).json()
    agent = trainer.agent("shared")
    agent.choose = lambda *args: (6, 0)
    for column in (0, 1, 2, 3):
        payload = {"game_id": state["game_id"], "move_count": state["move_count"], "column": column}
        response = viewer.post("/human/move", json=payload)
        assert response.status_code == 200, response.text
        state = response.json()
        assert viewer.post("/human/move", json=payload).status_code == 409
    assert state["winner"] == 1 and state["status"] == "finished"
    assert state["games"] == state["scores"]["1"] == 1
    assert len(agent.memory) == 3
    assert agent.memory[-1][2] == -0.97
    assert agent.decisions == 0  # Greedy play, while train=true still records experience.
    assert not trainer.games
    new = viewer.post("/human/new").json()
    assert new["game_id"] != state["game_id"] and new["board"] == empty_board()
    assert viewer.post("/human/move", json=payload).status_code == 409
    assert viewer.post("/mode", json={"mode": "simulation"}).json()["mode"] == "simulation"


def test_bot_win_and_abandoned_game_cleanup(human_service):
    viewer, trainer = human_service
    state = viewer.post("/human/new").json()
    agent = trainer.agent("shared")
    agent.batch_size = 1
    agent.choose = lambda *args: (6, 0)
    for column in (0, 1, 0, 1):
        state = viewer.post("/human/move", json={"game_id": state["game_id"],
                           "move_count": state["move_count"], "column": column}).json()
    assert state["winner"] == -1 and state["status"] == "finished"
    assert trainer.agents["shared"].memory[-1][2] == 0.96
    assert agent.updates > 0 and agent.path.exists()
    state = viewer.post("/human/new").json()
    viewer.post("/human/move", json={"game_id": state["game_id"], "move_count": 0, "column": 0})
    assert trainer.games
    viewer.post("/mode", json={"mode": "simulation"})
    assert not trainer.games


def test_human_full_column_and_invalid_requests(human_service):
    viewer, trainer = human_service
    assert viewer.post("/mode", json={"mode": "invalid"}).status_code == 422
    state = viewer.post("/human/new").json()
    trainer.agent("shared").choose = lambda *args: (0, 0)
    for count in (0, 2):
        assert viewer.post("/human/move", json={"game_id": state["game_id"],
                           "move_count": count, "column": 0}).status_code == 200
    # Fill the last slot with the human disc, then let the bot play elsewhere.
    trainer.agent("shared").choose = lambda *args: (1, 0)
    viewer.post("/human/move", json={"game_id": state["game_id"], "move_count": 4, "column": 0})
    payload = {"game_id": state["game_id"], "move_count": 6, "column": 0}
    assert viewer.post("/human/move", json=payload).status_code == 409
    assert viewer.post("/human/move", json={**payload, "column": 7}).status_code == 422
    assert viewer.post("/human/move", json={**payload, "column": True}).status_code == 422


def test_failed_bot_move_can_retry_without_repeating_human_disc(human_service, monkeypatch):
    import connect4.simulation_service as module
    viewer, trainer = human_service
    state = viewer.post("/human/new").json()
    original_post = module.post

    def failure(*args):
        raise httpx.ConnectError("API offline")

    monkeypatch.setattr(module, "post", failure)
    assert viewer.post("/human/move", json={"game_id": state["game_id"],
                       "move_count": 0, "column": 0}).status_code == 502
    failed = viewer.get("/state").json()
    assert failed["move_count"] == 1 and failed["status"] == "human_error"
    monkeypatch.setattr(module, "post", original_post)
    retry = {"game_id": state["game_id"], "move_count": 1}
    recovered = viewer.post("/human/retry", json=retry).json()
    assert recovered["move_count"] == 2 and recovered["status"] == "human_turn"
    assert viewer.post("/human/retry", json=retry).status_code == 409


def test_failed_finish_retries_terminal_reward_once(human_service, monkeypatch):
    import connect4.simulation_service as module
    viewer, trainer = human_service
    state = viewer.post("/human/new").json()
    agent = trainer.agent("shared")
    agent.choose = lambda *args: (6, 0)
    original_post = module.post

    def lose_finish_response(client, path, payload):
        result = original_post(client, path, payload)
        if path == "/finish":
            raise httpx.ReadError("Response lost after finish was accepted")
        return result

    monkeypatch.setattr(module, "post", lose_finish_response)
    for column in (0, 1, 2, 3):
        response = viewer.post("/human/move", json={"game_id": state["game_id"],
                              "move_count": state["move_count"], "column": column})
        state = viewer.get("/state").json()
    assert response.status_code == 502 and state["done"]
    assert state["games"] == 0 and len(agent.memory) == 3
    monkeypatch.setattr(module, "post", original_post)
    response = viewer.post("/human/retry", json={"game_id": state["game_id"],
                           "move_count": state["move_count"]})
    assert response.json()["games"] == 1
    assert len(agent.memory) == 3 and not trainer.games


def test_switching_mode_pauses_worker_and_resumes_self_play(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    simulation = Simulation("http://testserver", interval=0.05, result_delay=0)
    moved = Event()
    original_update = simulation.update

    def update(changes):
        original_update(changes)
        if changes.get("move_count") == 1:
            moved.set()

    monkeypatch.setattr(simulation, "update", update)
    with TestClient(create_api(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        thread = Thread(target=simulation.run)
        thread.start()
        try:
            assert moved.wait(5)
            human = simulation.set_mode("human")
            assert human["mode"] == "human" and human["move_count"] == 0
            assert not api.app.state.trainer.games
            moved.clear()
            # While human mode is active, only an explicit human action advances play.
            assert not moved.wait(0.2)
            assert simulation.snapshot() == human
            simulation.human_move(human["game_id"], 0, 0)
            moved.clear()
            simulation.set_mode("simulation")
            assert moved.wait(5)
        finally:
            simulation.stop.set()
            simulation.game_cancel.set()
            thread.join(5)
        assert not thread.is_alive()
        assert not api.app.state.trainer.games
