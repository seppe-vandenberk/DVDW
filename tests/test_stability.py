from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from connect4.api import create_app, Session
from connect4.dqn import Agent, GAMMA, tactical_margin_loss
from connect4.game import drop, empty_board, encode


def test_margin_stops_pushing_once_preferred_move_is_ahead():
    legal = torch.ones(1, 3, dtype=torch.bool)
    hints = torch.tensor([[True, False, False]])
    q = torch.tensor([[0.5, 0.1, -0.2]], requires_grad=True)
    loss = tactical_margin_loss(q, hints, legal)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(q.grad, torch.zeros_like(q))
    assert tactical_margin_loss(torch.tensor([[0.0, 0.4, 100.0]]), hints,
                                torch.tensor([[True, True, False]])).item() == pytest.approx(0.6)


def test_checkpoint_retries_locked_destination_and_preserves_old_file(tmp_path, monkeypatch):
    agent = Agent(tmp_path / "a.pt", device="cpu")
    agent.save()
    original = Path.replace
    calls = []

    def locked_then_ok(path, destination):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("destination locked")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", locked_then_ok)
    monkeypatch.setattr("connect4.dqn.time.sleep", lambda _: None)
    agent.updates = 5
    agent.save()
    assert len(calls) == 3
    assert torch.load(agent.path, weights_only=True)["updates"] == 5
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(Path, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("locked")))
    agent.updates = 6
    with pytest.raises(PermissionError):
        agent.save()
    assert torch.load(agent.path, weights_only=True)["updates"] == 5
    assert agent.checkpoint_errors == 1 and agent.checkpoint_error
    assert not list(tmp_path.glob("*.tmp"))


def test_save_failure_never_repeats_game_learning(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        trainer = client.app.state.trainer
        agent = trainer.agent("shared")
        original = agent.save
        monkeypatch.setattr(agent, "save", lambda: (_ for _ in ()).throw(PermissionError("locked")))
        moves = iter([0, 6, 1, 6, 2, 5, 3])
        monkeypatch.setattr(agent, "choose", lambda *args: (next(moves), 0))
        board = empty_board()
        for index in range(7):
            payload = {"game_id": "retry", "board": board, "player": 1 if index % 2 == 0 else -1}
            first = client.post("/move", json=payload)
            assert first.status_code == 200
            count = agent.observations
            assert client.post("/move", json=payload).json() == first.json()
            assert agent.observations == count
            board = first.json()["predicted_board"]
        payload = {"game_id": "retry", "board": board}
        assert client.post("/finish", json=payload).status_code == 200
        updates = agent.updates
        assert agent.observations == 7
        assert client.post("/finish", json=payload).status_code == 200
        assert agent.observations == 7 and agent.updates == updates
        assert trainer.errors["checkpoint"] > 0
        monkeypatch.setattr(agent, "save", original)


def test_three_steps_do_not_mix_games_or_colors_and_flush_terminal(tmp_path):
    trainer = create_app(tmp_path, device="cpu").state.trainer
    agent = trainer.agent("shared")
    board = empty_board()
    # Three settled own-turn transitions for one player, interleaved with another session.
    first, other = Session(board), Session(board)
    for i in range(3):
        first.pending[1] = ("shared", encode(board, 1), i)
        trainer.settle(first, board, 1, False)
        other.pending[-1] = ("shared", encode(board, -1), 6)
        trainer.settle(other, board, -1, False)
    assert len(agent.memory) == 2
    assert agent.memory[0][1] == 0 and agent.memory[1][1] == 6
    assert agent.memory[0][5] == pytest.approx(GAMMA ** 3)
    assert len(first.trajectories[1]) == len(other.trajectories[-1]) == 2
    for i, col in enumerate([0, 6, 1, 6, 2, 5, 3]):
        board = drop(board, col, 1 if i % 2 == 0 else -1)
    first.pending[1] = ("shared", encode(empty_board(), 1), 3)
    trainer.settle(first, board, 1, True)
    assert [x[2] for x in list(agent.memory)[-3:]] == pytest.approx([0.96 * GAMMA**2, 0.96 * GAMMA, 0.96])
    assert all(x[-1] for x in list(agent.memory)[-3:])
    assert not first.trajectories[1] and len(other.trajectories[-1]) == 2


def test_curriculum_uses_fixed_benchmarks_and_does_not_train(tmp_path):
    from connect4.evaluation import BENCHMARK
    with TestClient(create_app(tmp_path, device="cpu")) as client:
        trainer = client.app.state.trainer
        assert client.get("/models/shared/curriculum").json()["rule_probability"] == 0.5
        agent = trainer.agent("shared")
        agent.evaluations.append({"benchmark": BENCHMARK, "score": 0.3})
        assert client.get("/models/shared/curriculum").json()["rule_probability"] == 0.75
        agent.evaluations.clear()
        agent.evaluations.append({"benchmark": BENCHMARK, "score": 0.8})
        assert client.get("/models/shared/curriculum").json()["rule_probability"] == 1
        assert agent.observations == agent.decisions == agent.updates == 0
