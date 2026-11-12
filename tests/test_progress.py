from contextlib import nullcontext
from copy import deepcopy
import random

from fastapi.testclient import TestClient
import pytest
import torch

from connect4.api import create_app
from connect4.dqn import Agent
from connect4.evaluation import EVALUATION_GAMES, EVALUATION_INTERVAL, evaluate, opponent_move
from connect4.game import drop, empty_board, encode
from connect4.simulation_service import create_app as create_viewer


def test_tactical_opponent_wins_before_blocking_and_blocks_threats():
    board = empty_board()
    for index, column in enumerate([0, 6, 1, 6, 2, 5]):
        board = drop(board, column, 1 if index % 2 == 0 else -1)
    assert opponent_move(board, 1, random.Random(0)) == 3
    assert opponent_move(board, -1, random.Random(0)) == 3
    board = empty_board()
    for index, column in enumerate([0, 6, 1, 6, 2, 6, 4]):
        board = drop(board, column, 1 if index % 2 == 0 else -1)
    assert opponent_move(board, -1, random.Random(0)) == 6


def test_benchmark_plays_25_games_in_each_position(tmp_path):
    agent = Agent(tmp_path / "balanced.pt", device="cpu")
    starts = []
    choose = agent.choose

    def record(board, player, explore):
        if sum(v != 0 for col in board for v in col) <= 1:
            starts.append(player)
        assert explore is False
        return choose(board, player, explore)

    agent.choose = record
    result = evaluate(agent)
    assert result["games"] == 50
    assert starts.count(1) == starts.count(-1) == 25


def test_benchmark_is_repeatable_and_does_not_train_or_change_exploration(tmp_path):
    agent = Agent(tmp_path / "benchmark.pt", device="cpu")
    agent.decisions = 5000
    agent.remember(encode(empty_board(), 1), 0, -0.01, empty_board(), 1, False)
    weights = [p.detach().clone() for p in agent.online.parameters()]
    memory = list(agent.memory)
    rng_state = random.getstate()
    first, second = evaluate(agent), evaluate(agent)
    for key in first:
        if key != "timestamp":
            assert first[key] == second[key]
    assert first["wins"] + first["draws"] + first["losses"] == EVALUATION_GAMES
    assert first["score"] == (first["wins"] + 0.5 * first["draws"]) / EVALUATION_GAMES
    assert first["win_rate"] == first["wins"] / EVALUATION_GAMES
    assert agent.decisions == 5000 and agent.updates == 0
    assert list(agent.memory) == memory and random.getstate() == rng_state
    assert all(torch.equal(a, b) for a, b in zip(weights, agent.online.parameters()))


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.75, 1.0])
def test_exploration_reset_preserves_learning_active_games_and_persists(tmp_path, rate):
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        trainer = client.app.state.trainer
        client.post("/move", json={"game_id": "active", "board": empty_board(), "player": 1})
        agent = trainer.agent("shared")
        agent.decisions = 10000
        agent.updates = 42
        agent.remember(encode(empty_board(), 1), 0, -0.01, empty_board(), 1, False)
        weights = [p.detach().clone() for p in agent.online.parameters()]
        memory = list(agent.memory)
        pending = deepcopy(trainer.games["active"].pending)
        response = client.post("/exploration/reset", json={"model_id": "shared", "rate": rate})
        assert response.status_code == 200
        assert response.json()["exploration_rate"] == rate
        assert agent.decisions == 0 and agent.updates == 42
        assert list(agent.memory) == memory
        assert trainer.games["active"].pending == pending
        assert all(torch.equal(a, b) for a, b in zip(weights, agent.online.parameters()))
        restored = Agent(agent.path, device="cpu")
        assert restored.decisions == 0 and restored.updates == 42
        assert restored.exploration_rate == rate
        assert list(restored.memory) == memory
        assert client.post("/exploration/reset", json={"model_id": "../bad"}).status_code == 422


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.75, 1.0])
def test_selected_exploration_rate_decays_and_resumes_from_checkpoint(tmp_path, rate):
    agent = Agent(tmp_path / "shared.pt", device="cpu")
    agent.reset_exploration(rate)
    for index in range(3):
        _, epsilon = agent.choose(empty_board(), 1, True)
        assert epsilon == pytest.approx(rate - index / 50000)
    agent.save()
    resumed = Agent(agent.path, device="cpu")
    assert resumed.decisions == 3 and resumed.exploration_start == rate
    assert resumed.exploration_rate == pytest.approx(rate - 0.00006)
    resumed.decisions = 50000
    assert resumed.exploration_rate == 0.05


def test_legacy_checkpoint_preserves_existing_exploration_schedule(tmp_path):
    agent = Agent(tmp_path / "old.pt", device="cpu")
    agent.decisions = 5000
    agent.save()
    data = torch.load(agent.path, weights_only=True)
    del data["exploration_start"]
    torch.save(data, agent.path)
    restored = Agent(agent.path, device="cpu")
    assert restored.decisions == 5000 and restored.exploration_rate == 0.9


@pytest.mark.parametrize("rate", [-1, 0, 1.01, True, "0.5"])
def test_invalid_exploration_rates_are_rejected(tmp_path, rate):
    with TestClient(create_app(tmp_path, device="cpu")) as api:
        assert api.post("/exploration/reset", json={"rate": rate}).status_code == 422
        assert not api.app.state.trainer.agents
    with TestClient(create_viewer(run_worker=False)) as viewer:
        assert viewer.post("/exploration/reset", json={"rate": rate}).status_code == 422


def test_progress_persists_and_only_measures_when_due(tmp_path, monkeypatch):
    import connect4.api as module
    calls = []

    def benchmark(agent):
        calls.append(agent.updates)
        return {"benchmark": module.BENCHMARK, "updates": agent.updates, "score": 0.5}

    monkeypatch.setattr(module, "evaluate", benchmark)
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        agent = client.app.state.trainer.agent("shared")
        first = client.get("/models/shared/progress").json()
        assert len(first["history"]) == 1 and calls == [0]
        writes = []
        monkeypatch.setattr(agent, "save", lambda: writes.append(agent.updates))
        client.get("/models/shared/progress")
        assert calls == [0] and not writes
        agent.updates = EVALUATION_INTERVAL - 1
        client.get("/models/shared/progress")
        assert calls == [0]
        agent.updates += 1
        progress = client.get("/models/shared/progress").json()
        assert calls == [0, EVALUATION_INTERVAL] and len(progress["history"]) == 2
        client.post("/evaluate", json={"model_id": "shared"})
        assert len(agent.evaluations) == 2  # Same update replaces its previous result.
        for update in range(101):
            agent.updates += EVALUATION_INTERVAL
            client.get("/models/shared/progress")
        assert len(agent.evaluations) == 100
        monkeypatch.undo()
        agent.save()
        restored = Agent(agent.path, device="cpu")
        assert list(restored.evaluations) == list(agent.evaluations)


def test_training_automatically_measures_after_interval(tmp_path, monkeypatch):
    import connect4.api as module
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        agent = client.app.state.trainer.agent("shared")
        agent.evaluations.append({"benchmark": module.BENCHMARK, "updates": 0})
        agent.updates = EVALUATION_INTERVAL - 1
        agent.batch_size = 1
        agent.remember(encode(empty_board(), 1), 0, -0.01, empty_board(), 1, True)
        measured = []

        def benchmark(model):
            measured.append(model.updates)
            return {"benchmark": module.BENCHMARK, "updates": model.updates}

        monkeypatch.setattr(module, "evaluate", benchmark)
        response = client.post("/move", json={"game_id": "auto", "board": empty_board(), "player": 1})
        assert response.status_code == 200
        assert measured == [EVALUATION_INTERVAL]


def test_changed_benchmark_starts_a_separate_chart(tmp_path, monkeypatch):
    import connect4.api as module
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        agent = client.app.state.trainer.agent("shared")
        agent.evaluations.append({"benchmark": "rules-v1", "updates": 0, "score": 1.0})
        monkeypatch.setattr(module, "evaluate", lambda agent: {
            "benchmark": module.BENCHMARK, "updates": agent.updates, "score": 0.0})
        history = client.get("/models/shared/progress").json()["history"]
        assert len(history) == 1 and history[0]["benchmark"] == module.BENCHMARK
        assert len(agent.evaluations) == 2


def test_viewer_controls_target_each_configured_model_once(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    with TestClient(create_app(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        with TestClient(create_viewer(run_worker=False)) as viewer:
            response = viewer.get("/progress")
            assert response.status_code == 200
            assert list(response.json()["models"]) == ["shared"]
            agent = api.app.state.trainer.agent("shared")
            assert len(agent.evaluations) == 1
            agent.decisions = 9999
            response = viewer.post("/exploration/reset")
            assert response.json()["models"]["shared"]["exploration_rate"] == 1.0
            assert len(agent.evaluations) == 1
            for rate in (0.25, 0.5, 0.75):
                response = viewer.post("/exploration/reset", json={"rate": rate})
                assert response.status_code == 200
                assert response.json()["models"]["shared"]["exploration_rate"] == rate
                assert agent.decisions == 0 and len(agent.evaluations) == 1
            response = viewer.post("/evaluation")
            assert response.status_code == 200 and len(agent.evaluations) == 1
            assert viewer.get("/state").json()["board"] == empty_board()
            monkeypatch.setenv("MODEL_TWO", "other")
            with TestClient(create_viewer(run_worker=False)) as two_models:
                response = two_models.post("/exploration/reset")
                assert set(response.json()["models"]) == {"shared", "other"}
                assert api.app.state.trainer.agents["other"].decisions == 0
