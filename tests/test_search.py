from copy import deepcopy
import math
import random

from fastapi.testclient import TestClient
import pytest
import torch

from connect4.api import create_app
from connect4.az import Agent, GAMMA, PolicyValue, search_many
from connect4.bitboard import Position
from connect4.game import empty_board, drop, outcome, encode, legal_moves
from connect4.learning import generate, match, Worker


def test_bitboards_match_reference_engine_through_random_games():
    rng = random.Random(18)
    for _ in range(100):
        board, player, position = empty_board(), 1, Position()
        while outcome(board) is None:
            assert position.legal() == legal_moves(board)
            assert position.board() == encode(board, player)
            assert Position.from_board(board, player) == position
            assert position.terminal() is None
            column = rng.choice(legal_moves(board))
            board = drop(board, column, player)
            position = position.play(column)
            player *= -1
        expected = 0 if outcome(board) == 0 else -1
        assert position.terminal() == expected


def tactical_board(moves):
    board = empty_board()
    for i, c in enumerate(moves):
        board = drop(board, c, 1 if i % 2 == 0 else -1)
    return board


def test_search_discovers_win_and_block_without_teacher_labels():
    torch.set_num_threads(1)
    model = PolicyValue()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    boards = [tactical_board([0, 6, 1, 6, 2, 5]),
              tactical_board([6, 0, 6, 1, 5, 2])]
    policies = search_many(model, [Position.from_board(b, 1) for b in boards], simulations=128)
    assert [max(range(7), key=lambda c: p[c]) for p in policies] == [3, 3]
    assert all(sum(p) == pytest.approx(1) for p in policies)


def test_search_masks_full_column_and_batch_matches_single():
    torch.manual_seed(13)
    model = PolicyValue()
    board = empty_board()
    board[0] = [1, -1, 1, -1, 1]
    position = Position.from_board(board, -1)
    batch = search_many(model, [position, position], simulations=8)
    single = search_many(model, [position], simulations=8)[0]
    assert batch[0] == batch[1] == single
    assert single[0] == 0


def test_episode_rewards_training_and_checkpoint_round_trip(tmp_path):
    agent = Agent(tmp_path / "az.pt", device="cpu", batch_size=2)
    board = empty_board()
    pi = [0, 0, 0, 1, 0, 0, 0]
    records = [(board, pi, 1, 0), (encode(drop(board, 3, 1), -1), pi, -1, 1)]
    agent.add_episode(records, 1, final_ply=6)
    assert [sample[2] for sample in agent.memory] == pytest.approx([GAMMA ** 6, -GAMMA ** 5])
    weights = deepcopy(agent.online.state_dict())
    assert math.isfinite(agent.train_step())
    assert any(not torch.equal(weights[k], v) for k, v in agent.online.state_dict().items())
    agent.save()
    restored = Agent(agent.path, device="cpu", batch_size=2)
    assert restored.updates == 1 and list(restored.memory) == list(agent.memory)
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), restored.online.parameters()))


def test_local_generation_produces_outcome_targets_and_updates(tmp_path):
    agent = Agent(tmp_path / "local.pt", device="cpu", batch_size=8)
    results = generate(agent, games=4, simulations=4)
    assert len(results) == 4 and agent.observations >= 8
    assert all(-1 <= sample[2] <= 1 for sample in agent.memory)
    assert all(sum(sample[1]) == pytest.approx(1) for sample in agent.memory)
    assert math.isfinite(agent.train_step())


def test_camera_outcomes_are_queued_once_and_retries_are_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SEARCH_SIMULATIONS", "8")
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        trainer = client.app.state.trainer
        agent = trainer.agent("shared")
        choices = iter([6, 6, 5])
        def choose(*args):
            action = next(choices)
            agent.last_policy = [float(c == action) for c in range(7)]
            return action, 0
        agent.choose = choose
        board = empty_board()
        for c in (0, 1, 2):
            board = drop(board, c, 1)
            payload = {"game_id": "physical", "board": board, "player": -1}
            response = client.post("/move", json=payload)
            assert response.status_code == 200, response.text
            assert client.post("/move", json=payload).json() == response.json()
            board = response.json()["predicted_board"]
        board = drop(board, 3, 1)
        payload = {"game_id": "physical", "board": board}
        assert client.post("/finish", json=payload).status_code == 200
        assert client.post("/finish", json=payload).status_code == 200
        worker = trainer.workers["shared"]
        assert worker.incoming.qsize() == 1
        records, winner, final_ply = worker.incoming.get_nowait()
        assert len(records) == 3 and winner == 1 and final_ply == 6
        assert agent.updates == 0 and not agent.path.exists()
        assert client.post("/evaluate", json={}).json()["evaluation_pending"]


def test_cancelling_camera_game_never_assigns_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SEARCH_SIMULATIONS", "2")
    with TestClient(create_app(tmp_path, device="cpu", backend="az", background=False)) as client:
        assert client.post("/move", json={"game_id": "cancel", "board": empty_board(), "player": 1}).status_code == 200
        assert client.delete("/games/cancel").status_code == 200
        assert client.app.state.trainer.workers["shared"].incoming.empty()


def test_match_is_balanced_repeatable_and_does_not_change_weights():
    torch.manual_seed(4)
    model = PolicyValue()
    weights = deepcopy(model.state_dict())
    a = match(model, "random", "cpu", games=4, simulations=4)
    b = match(model, "random", "cpu", games=4, simulations=4)
    assert a == b
    assert sum(a["positions"]["first"].values()) == 2
    assert sum(a["positions"]["second"].values()) == 2
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
