from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from connect4.event_game import EventGame, Setup, ManualMove, stamp
from connect4.game import drop


@pytest.fixture
def game(monkeypatch, tmp_path):
    import connect4.event_game as module
    jobs = []
    class DeferredThread:
        def __init__(self, target, args=(), **kwargs):
            self.target, self.args = target, args
        def start(self):
            jobs.append((self.target, self.args))
    monkeypatch.setattr(module, "Thread", DeferredThread)
    instance = EventGame("http://model", score_file=tmp_path / "scores.txt")
    instance.jobs = jobs
    return instance


def frame(game, board=None, **changes):
    return {**game.mock_frame(), **({"board": board} if board else {}), **changes}


def test_same_board_and_new_timestamp_never_repeat_move(game):
    game.reset(Setup(first="robot"))
    game.accept(frame(game))
    assert len(game.jobs) == 1
    assert game.status == "thinking"
    assert 0.8 < game.thinking_until - datetime.now(timezone.utc).timestamp() <= 1
    game.busy = False
    game.status = "suggestion"
    revision = game.revision
    game.accept(frame(game, timestamp=stamp()))
    assert len(game.jobs) == 1
    assert game.revision == revision
    assert game.status == "suggestion"
    game.retry()
    assert len(game.jobs) == 2


@pytest.mark.parametrize("robot_color", ["B", "R"])
@pytest.mark.parametrize("first", ["human", "robot"])
def test_camera_transpose_colors_and_first_player(game, robot_color, first):
    game.reset(Setup(robot_color=robot_color, first=first))
    board = [["X"]*7 for _ in range(5)]
    board[4][2] = robot_color if first == "robot" else ("R" if robot_color == "B" else "B")
    game.accept(frame(game, board))
    assert game.board[2][4] == 1
    assert game.snapshot()["board"][2][4] == board[4][2]
    assert len(game.jobs) == (1 if first == "human" else 0)


@pytest.mark.parametrize("kind", ["invalid", "shape", "floating", "counts", "stale", "future"])
def test_bad_observation_cannot_replace_board_or_call_model(game, kind):
    payload = frame(game)
    if kind == "invalid": payload["valid"] = False
    if kind == "shape": payload["board"] = [["X"]*7]*4
    if kind == "floating": payload["board"][0][0] = "R"
    if kind == "counts": payload["board"][4][0] = "B"
    if kind in ("stale", "future"):
        payload["timestamp"] = (datetime.now(timezone.utc)+timedelta(seconds=-30 if kind == "stale" else 30)).isoformat()
    before = deepcopy(game.board)
    with pytest.raises(ValueError): game.accept(payload)
    assert game.board == before
    assert not game.jobs


def test_manual_takeover_turn_checks_revision_and_resume(game):
    revision = game.revision
    with pytest.raises(HTTPException):
        game.manual_move(ManualMove(column=0, player="robot", revision=revision))
    state = game.manual_move(ManualMove(column=1, player="human", revision=revision))
    assert state["manual"] and state["board"][1][4] == "R"
    with pytest.raises(HTTPException):
        game.manual_move(ManualMove(column=0, player="robot", revision=revision))
    game.accept(frame(game))
    assert game.board[1][4] == 1  # Camera cannot overwrite manual takeover.
    state = game.resume()
    assert not state["manual"]
    assert state["move_count"] == 0


def test_old_response_is_discarded_and_retry_error_is_explicit(game, monkeypatch):
    import connect4.event_game as module
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"column": 3}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, path, json): return Response()
    monkeypatch.setattr(module.httpx, "Client", Client)
    game.reset(Setup(first="robot"))
    game.accept(frame(game))
    target, args = game.jobs[-1]
    game.reset()
    target(*args[:-1], 0)
    assert game.suggestion is None and game.status == "waiting"
    game.reset(Setup(first="robot"))
    game.accept(frame(game))
    target, args = game.jobs[-1]
    target(*args[:-1], 0)
    assert game.suggestion == 3 and game.model["ok"]
    class Broken(Client):
        def post(self, *args, **kwargs): raise RuntimeError("Offline")
    monkeypatch.setattr(module.httpx, "Client", Broken)
    game.retry()
    target, args = game.jobs[-1]
    target(*args[:-1], 0)
    assert game.status == "move_error" and game.model["error"] == "Offline"
    count = len(game.jobs)
    game.accept(frame(game))
    assert len(game.jobs) == count  # Failures need explicit retry, too.


def test_changed_camera_game_resets_identity_and_deduplication(game):
    game.reset(Setup(first="robot"))
    game.accept(frame(game, game_id=1))
    session = game.session
    game.busy = False
    game.accept(frame(game, game_id=2))
    assert game.session != session
    assert len(game.jobs) == 2


def test_terminal_board_confirms_outcome_instead_of_requesting_move(game):
    board = [["X"]*7 for _ in range(5)]
    for col in range(4): board[4][col] = "R"
    for col in range(3): board[3][col] = "B"
    game.accept(frame(game, board))
    assert game.status == "finished"
    assert game.snapshot()["winner"] == "human"
    assert len(game.jobs) == 1 and game.jobs[0][0] == game.finish


def test_event_routes_share_game_and_validate_requests(monkeypatch):
    from connect4.simulation_service import create_app
    monkeypatch.setattr(EventGame, "start", lambda self: None)
    with TestClient(create_app()) as client:
        for page in ("/participant", "/instructor"):
            assert client.get(page).status_code == 200
        assert client.post("/event/new", json={"robot_color":"R", "first":"robot"}).status_code == 200
        assert client.get("/event/state").json()["setup"]["robot_color"] == "R"
        assert client.post("/event/new", json={"robot_color":"X"}).status_code == 422
        before = client.get("/event/state").json()
        response=client.post("/event/drop", json={"column":2,"player":"robot","revision":before["revision"]})
        assert response.status_code == 200
        assert client.get("/event/state").json()["board"][2][4] == "R"
        assert client.post("/event/drop", json={"column":8,"player":"human","revision":before["revision"]}).status_code == 422
        mock = client.get("/event/camera/mock").json()
        assert len(mock["board"]) == 5 and len(mock["board"][0]) == 7
        assert client.post("/event/camera/mock", json={**mock,"valid":False}).status_code == 200


@pytest.mark.parametrize("robot_color,first", [("B", "human"), ("R", "robot")])
def test_real_model_api_accepts_mapped_board_and_confirmed_reply(game, monkeypatch, tmp_path, robot_color, first):
    from contextlib import nullcontext
    import connect4.event_game as module
    from connect4.api import create_app
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        game.reset(Setup(robot_color=robot_color, first=first))
        if first == "human":
            game.manual_move(ManualMove(column=0, player="human", revision=game.revision))
        else:
            game.accept(frame(game))
        target, args = game.jobs[-1]
        target(*args[:-1], 0)
        assert game.status == "suggestion", game.model
        game.manual_move(ManualMove(column=game.suggestion, player="robot", revision=game.revision))
        game.manual_move(ManualMove(column=1, player="human", revision=game.revision))
        target, args = game.jobs[-1]
        target(*args[:-1], 0)
        assert game.status == "suggestion", game.model


def test_score_counts_each_finished_game_once_and_survives_new_game(game):
    board = [["X"]*7 for _ in range(5)]
    for col in range(4): board[4][col] = "R"
    for col in range(3): board[3][col] = "B"
    game.accept(frame(game, board))
    assert game.snapshot()["robot_score"] == -1
    game.accept(frame(game, board))
    game.retry()
    assert len(game.score_history) == 2
    game.reset(Setup(robot_color="R", first="robot"))
    assert game.snapshot()["robot_score"] == -1
    game.accept(frame(game, board))
    assert game.snapshot()["robot_score"] == 0
    assert game.score_history[-1] == {"game":2,"score":0,"points":1}


def test_score_text_file_survives_restart(game):
    assert game.record_score(1)
    assert game.record_score(-1)
    assert game.record_score(0)
    assert game.score_file.read_text() == "+1\n-1\n0\n"
    restored = EventGame("http://model", score_file=game.score_file)
    assert restored.score_history == game.score_history
    assert restored.snapshot()["robot_score"] == 0


def test_failed_score_save_can_retry_without_duplicate(game, monkeypatch):
    from pathlib import Path
    original = Path.replace
    def fail(*args, **kwargs): raise OSError("Disk unavailable")
    monkeypatch.setattr(Path, "replace", fail)
    assert not game.record_score(1)
    assert len(game.score_history) == 1
    assert game.score_error
    monkeypatch.setattr(Path, "replace", original)
    assert game.record_score(1)
    assert game.score_file.read_text() == "+1\n"
    assert game.score_error is None


def test_previews_leave_board_and_scores_untouched(monkeypatch, tmp_path):
    from connect4.simulation_service import create_app
    monkeypatch.setenv("EVENT_SCORE_FILE", str(tmp_path / "preview-scores.txt"))
    monkeypatch.setattr(EventGame, "start", lambda self: None)
    with TestClient(create_app()) as client:
        before = client.get("/event/state").json()
        for winner in ("human", "robot", None):
            state = client.post("/event/preview", json={"winner":winner}).json()
            assert state["preview_winner"] == winner
            assert state["board"] == before["board"]
            assert state["score_history"] == before["score_history"]
        assert not (tmp_path / "preview-scores.txt").exists()


def test_board_size_is_shared_and_survives_new_game(monkeypatch, tmp_path):
    from connect4.simulation_service import create_app
    monkeypatch.setenv("EVENT_SCORE_FILE", str(tmp_path / "scores.txt"))
    monkeypatch.setattr(EventGame, "start", lambda self: None)
    with TestClient(create_app()) as client:
        before = client.get("/event/state").json()
        state = client.post("/event/size", json={"scale":125}).json()
        assert state["board_scale"] == 125
        assert state["board"] == before["board"]
        assert state["score_history"] == before["score_history"]
        assert client.get("/event/state").json()["board_scale"] == 125
        assert client.post("/event/new", json={}).json()["board_scale"] == 125
        for value in (49, 176, 100.5, True):
            assert client.post("/event/size", json={"scale":value}).status_code == 422
