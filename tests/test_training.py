import math

from fastapi.testclient import TestClient
import pytest
import torch

from connect4.api import create_app
from connect4.dqn import Agent, DISC_PENALTY, outcome_reward, select_device
from connect4.game import drop, empty_board, encode, outcome, validate_board, winners
from connect4.simulator import play_game


@pytest.fixture
def service(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        yield client, app.state.trainer


@pytest.mark.parametrize("dc,dr", [(1, 0), (0, 1), (1, 1), (1, -1)])
def test_four_directions(dc, dr):
    board = empty_board()
    for i in range(4):
        board[i * dc][(3 if dr < 0 else 0) + i * dr] = -1
    assert winners(board) == {-1}


def test_board_validation_and_full_column():
    board = empty_board()
    board[0][0] = 1
    with pytest.raises(ValueError, match="Floating"):
        validate_board(board)
    board[0] = [1, -1, 1, -1, 1]
    with pytest.raises(ValueError, match="full"):
        drop(board, 0, -1)


def test_rewards_retries_and_own_turn_transitions(service):
    client, trainer = service
    agent = trainer.agent("shared")
    moves = iter([0, 0, 1, 1, 2, 2, 3])
    agent.choose = lambda *args: (next(moves), 0.0)
    board = empty_board()
    first = {"game_id": "test", "board": board, "player": 1}
    response = client.post("/move", json=first)
    assert response.status_code == 200
    assert client.post("/move", json=first).json() == response.json()
    board = drop(board, response.json()["column"], 1)
    for player in [-1, 1, -1, 1, -1, 1]:
        response = client.post("/move", json={"game_id": "test", "board": board, "player": player})
        assert response.status_code == 200, response.text
        board = drop(board, response.json()["column"], player)
    assert response.json()["done"]
    assert outcome(board) == 1
    before = len(agent.memory)
    assert client.post("/finish", json={"game_id": "test", "board": board}).status_code == 200
    assert len(agent.memory) == before + 6 == 7
    assert [entry[2] for entry in list(agent.memory)[-3:]] == pytest.approx([-0.97 * 0.99**2, -0.97 * 0.99, -0.97])
    assert agent.memory[0][2] == 0.0
    assert all(entry[-1] for entry in list(agent.memory)[1:])
    expected = empty_board()
    for i, column in enumerate([0, 0, 1, 1, 2, 2]):
        expected = drop(expected, column, 1 if i % 2 == 0 else -1)
    assert agent.memory[0][3] == encode(expected, 1)
    assert client.post("/finish", json={"game_id": "test", "board": board}).status_code == 200
    assert len(agent.memory) == 7


def test_external_human_win_teaches_bot_loss(service):
    client, trainer = service
    agent = trainer.agent("robot")
    agent.choose = lambda *args: (6, 0)
    board = empty_board()
    for column in (0, 1, 2):
        board = drop(board, column, 1)
        response = client.post("/move", json={"game_id": "camera", "board": board,
                                               "player": -1, "model_id": "robot"})
        assert response.status_code == 200, response.text
        board = drop(board, response.json()["column"], -1)
    board = drop(board, 3, 1)
    assert client.post("/finish", json={"game_id": "camera", "board": board}).status_code == 200
    assert agent.memory[-1][2] == -1.0 + 3 * DISC_PENALTY
    assert agent.memory[-1][-1]


def test_draw(service):
    client, trainer = service
    board = [[1, -1, 1, -1, 1] if c % 4 < 2 else [-1, 1, -1, 1, -1] for c in range(7)]
    assert outcome(board) == 0
    validate_board(board)
    board[0][0] = 0
    agent = trainer.agent("shared")
    agent.choose = lambda *args: (0, 0)
    response = client.post("/move", json={"game_id": "draw", "board": board, "player": 1})
    assert response.status_code == 200, response.text
    board = drop(board, 0, 1)
    assert client.post("/finish", json={"game_id": "draw", "board": board}).status_code == 200
    assert agent.memory[-1][2] == 0.0
    assert agent.memory[-1][-1]


def test_invalid_requests_and_unconfirmed_action(service):
    client, trainer = service
    payload = {"game_id": "bad", "board": empty_board(), "player": -1}
    assert client.post("/move", json=payload).status_code == 409
    payload["player"] = 1
    payload["model_id"] = "../bad"
    assert client.post("/move", json=payload).status_code == 422
    payload["model_id"] = "shared"
    assert client.post("/move", json=payload).status_code == 200
    changed = drop(drop(empty_board(), 0, 1), 1, -1)
    # Replace the confirmed action with a different first disk.
    expected = trainer.games["bad"].expected
    wrong_col = (next(c for c in range(7) if expected[c][-1] == 1) + 1) % 7
    changed = drop(drop(empty_board(), wrong_col, 1), wrong_col, -1)
    assert client.post("/move", json={**payload, "board": changed}).status_code == 409
    assert client.post("/finish", json={"game_id": "bad", "board": expected}).status_code == 409


def test_training_mask_and_checkpoint(tmp_path):
    torch.manual_seed(1)
    agent = Agent(tmp_path / "test.pt", batch_size=2)
    board = empty_board()
    board[0] = [1, -1, 1, -1, 1]
    # Make illegal column 0 the greedy favorite; it must still be excluded.
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.column_bias[0] = 100
    assert agent.choose(board, -1, False)[0] != 0
    agent.remember(encode(board, -1), 1, 0, board, -1, False)
    agent.remember(encode(board, 1), 1, 1, board, 1, True)
    assert math.isfinite(agent.train_step())
    assert agent.updates == 1
    agent.save()
    restored = Agent(agent.path, batch_size=2)
    assert restored.updates == 1 and len(restored.memory) == 2
    for a, b in zip(agent.online.parameters(), restored.online.parameters()):
        assert torch.equal(a, b)
    assert math.isfinite(restored.train_step())


def test_simulator_trains_shared_and_separate_models(service):
    client, trainer = service
    for _ in range(4):
        assert play_game(client, interval=0) in (-1, 0, 1)
    assert trainer.agents["shared"].updates > 0
    assert len(trainer.agents["shared"].memory) >= 32
    assert play_game(client, "a", "b", interval=0) in (-1, 0, 1)
    assert len(trainer.agents["a"].memory) > 0
    assert len(trainer.agents["b"].memory) > 0
    assert not trainer.games


def test_inference_only_does_not_learn(service):
    client, trainer = service
    response = client.post("/move", json={"game_id": "eval", "board": empty_board(),
                                           "player": 1, "train": False})
    assert response.status_code == 200
    assert response.json()["epsilon"] == 0
    assert trainer.agents["shared"].updates == 0
    assert not trainer.games["eval"].pending


def test_every_training_move_saves_even_during_warmup(service):
    client, trainer = service
    response = client.post("/move", json={"game_id": "save", "board": empty_board(),
                                           "player": 1, "explore": True})
    assert response.status_code == 200
    agent = trainer.agents["shared"]
    assert agent.path.exists()
    restored = Agent(agent.path, device="cpu")
    assert restored.decisions == 1
    assert restored.updates == 0
    assert restored.device.type == "cpu"
    assert response.json()["device"] == str(agent.device)


def test_device_selection(monkeypatch):
    assert select_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        select_device("cuda")


@pytest.mark.parametrize("version", [0, 1])
def test_old_checkpoint_rewards_migrate_only_once(tmp_path, version):
    agent = Agent(tmp_path / "legacy.pt", device="cpu")
    board = empty_board()
    for i, column in enumerate([0, 6, 1, 6, 2, 5, 3]):
        board = drop(board, column, 1 if i % 2 == 0 else -1)
    for player in (1, -1):
        agent.remember(encode(empty_board(), player), 0,
                       player - (DISC_PENALTY if version else 0), board, player, True)
    agent.remember(encode(empty_board(), 1), 0, -0.01, empty_board(), 1, False)
    agent.save()
    data = torch.load(agent.path, weights_only=True)
    data["reward_version"] = version
    torch.save(data, agent.path)
    restored = Agent(agent.path, device="cpu")
    assert [entry[2] for entry in restored.memory] == pytest.approx([0.96, -0.97, 0])
    restored.save()
    again = Agent(agent.path, device="cpu")
    assert list(again.memory) == list(restored.memory)


def test_faster_win_has_larger_total_reward(service):
    client, trainer = service
    totals = []
    for game_id, moves in (("fast", [0, 6, 1, 6, 2, 5, 3]),
                           ("slow", [0, 6, 1, 6, 2, 5, 4, 5, 3])):
        agent = trainer.agent(game_id)
        choices = iter(moves)
        agent.choose = lambda *args: (next(choices), 0)
        board = empty_board()
        for index in range(len(moves)):
            result = client.post("/move", json={"game_id": game_id, "board": board,
                                 "player": 1 if index % 2 == 0 else -1, "model_id": game_id})
            assert result.status_code == 200, result.text
            board = result.json()["predicted_board"]
        assert outcome(board) == 1
        assert client.post("/finish", json={"game_id": game_id, "board": board}).status_code == 200
        # The acting player's own disks are +1 in the encoded state.
        totals.append(sum(entry[2] for entry in agent.memory if sum(v for col in entry[0] for v in col) == 0))
    assert totals[0] == pytest.approx(0.96 * (1 + 0.99 + 0.99**2))
    assert totals[1] == pytest.approx(0.95 * (1 + 0.99 + 0.99**2))


def test_long_loss_better_than_short_loss_but_worse_than_win():
    rewards = []
    for moves in ([0, 6, 1, 6, 2, 5, 3], [0, 6, 1, 6, 2, 5, 4, 5, 3]):
        board = empty_board()
        for i, column in enumerate(moves):
            board = drop(board, column, 1 if i % 2 == 0 else -1)
        rewards.append((outcome_reward(board, 1, True), outcome_reward(board, -1, True)))
    assert rewards[0][1] < rewards[1][1] < 0 < rewards[1][0] < rewards[0][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_training_and_cpu_checkpoint_restore(tmp_path):
    agent = Agent(tmp_path / "gpu.pt", batch_size=2, device="cuda")
    board = empty_board()
    assert agent.online.layers[0].weight.is_cuda
    assert agent.target.layers[0].weight.is_cuda
    assert agent.choose(board, 1, False)[0] in range(7)
    for done in (True, False):
        agent.remember(encode(board, 1), 3, 0, board, 1, done)
    assert math.isfinite(agent.train_step())
    agent.save()
    cpu = Agent(agent.path, batch_size=2, device="cpu")
    assert math.isfinite(cpu.train_step())
    cpu.save()
    gpu = Agent(agent.path, batch_size=2, device="cuda")
    assert math.isfinite(gpu.train_step())
