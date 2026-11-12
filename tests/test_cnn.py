import math

import pytest
import torch
from torch import nn

from connect4.dqn import ARCHITECTURE, Agent, DQN, tactical_actions
from connect4.game import drop, empty_board, encode
from connect4.train import demonstrations, tactical_suite


def test_matrix_encoding_preserves_column_row_order_and_perspective():
    board = drop(drop(empty_board(), 2, 1), 4, -1)
    state = encode(board, -1)
    assert len(state) == 7 and all(len(column) == 5 for column in state)
    assert state[2][4] == -1 and state[4][4] == 1
    state[2][4] = 0
    assert board[2][4] == 1


def test_inference_and_replay_share_one_color_perspective(tmp_path):
    agent = Agent(tmp_path / "perspective.pt", device="cpu")
    board = drop(drop(empty_board(), 2, 1), 4, -1)
    inverted = [[-v for v in column] for column in board]
    seen = []
    hook = agent.online.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    first = agent.choose(board, 1, False)
    second = agent.choose(inverted, -1, False)
    hook.remove()
    assert first == second and torch.equal(seen[0], seen[1])
    for actual, player in ((board, 1), (inverted, -1)):
        agent.remember(encode(actual, player), 3, 0, drop(actual, 3, player), player, False)
    assert agent.memory[0] == agent.memory[1]


def test_tiny_cnn_shapes_parameters_and_occupancy_planes():
    model = DQN()
    assert sum(p.numel() for p in model.parameters()) == 5112
    assert model(torch.zeros(7, 5)).shape == (7,)
    assert model(torch.zeros(4, 7, 5)).shape == (4, 7)
    with pytest.raises(ValueError, match="7x5"):
        model(torch.zeros(35))
    seen = []
    hook = model.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    board = torch.zeros(7, 5)
    board[2, 4], board[4, 4] = 1, -1
    model(board)
    hook.remove()
    assert seen[0].shape == (1, 3, 7, 5)
    assert seen[0][0, 0, 2, 4] == seen[0][0, 1, 4, 4] == 1
    assert seen[0][0, 2, 2, 3] == seen[0][0, 2, 4, 3] == 1
    assert seen[0][0, 2].sum() == 7


def test_tactical_teacher_is_training_only(tmp_path):
    board = empty_board()
    for index, column in enumerate([0, 6, 1, 6, 2, 6]):
        board = drop(board, column, 1 if index % 2 == 0 else -1)
    state = encode(board, 1)
    assert tactical_actions(tuple(tuple(col) for col in state)) == (3,)
    agent = Agent(tmp_path / "cnn.pt", device="cpu")
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.column_bias[0] = 10
    # Choosing a move uses only learned Q-values and the legal-column mask.
    assert agent.choose(board, 1, False)[0] == 0


def test_reflection_keeps_state_next_state_and_action_in_sync(tmp_path, monkeypatch):
    agent = Agent(tmp_path / "reflection.pt", batch_size=1, device="cpu")
    board = drop(empty_board(), 0, 1)
    next_board = drop(board, 2, -1)
    state = encode(board, 1)
    agent.remember(state, 0, 0, next_board, 1, True)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.column_bias.copy_(torch.arange(7))
    seen = []
    hook = agent.online.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs))
    assert math.isfinite(agent.train_step())
    hook.remove()
    assert torch.equal(seen[0][0], torch.tensor(state[::-1], dtype=torch.float32))
    assert torch.equal(seen[1][0], torch.tensor(encode(next_board, 1)[::-1], dtype=torch.float32))
    assert agent.last_td_loss == pytest.approx(5.5)  # Original action 0 becomes column 6.


def test_incompatible_dense_checkpoint_is_discarded(tmp_path):
    path = tmp_path / "shared.pt"
    dense = nn.Sequential(nn.Linear(35, 64), nn.ReLU(), nn.Linear(64, 7))
    torch.save({"online": dense.state_dict(), "updates": 999, "decisions": 9500,
                "memory": ["legacy experience"]}, path)
    agent = Agent(path, device="cpu")
    assert agent.updates == agent.decisions == agent.observations == 0
    assert not agent.memory and not agent.evaluations and not agent.optimizer.state
    assert torch.load(path, weights_only=True)["architecture"] == ARCHITECTURE


def test_staged_trained_model_replaces_old_checkpoint_once(tmp_path):
    path = tmp_path / "shared.pt"
    torch.save({"old": True}, path)
    path.with_suffix(".reset").write_text("Old reset request")
    candidate = Agent(path.with_suffix(".next.pt"), device="cpu")
    candidate.observations = 10000
    candidate.pretraining = {"method": "rule-demonstrations", "positions": 10000}
    candidate.reset_exploration(0.25)
    candidate.save()
    activated = Agent(path, device="cpu")
    assert activated.observations == 10000 and activated.exploration_rate == 0.25
    assert activated.pretraining == candidate.pretraining
    assert not path.with_suffix(".next.pt").exists() and not path.with_suffix(".reset").exists()
    assert all(torch.equal(a, b) for a, b in zip(candidate.online.parameters(), activated.online.parameters()))
    activated.decisions = 5
    activated.save()
    assert Agent(path, device="cpu").decisions == 5


def test_demonstration_budget_labels_and_held_out_exclusion():
    suite = tactical_suite(per_kind=4)
    forbidden = {tuple(tuple(col) for col in state) for cases in suite.values() for state, _ in cases}
    data = demonstrations(100, forbidden=forbidden, seed=73)
    assert len(data) == 100
    states = [tuple(tuple(col) for col in state) for state, _, _ in data]
    assert len(set(states)) == 100 and not set(states) & forbidden
    for state, targets, weight in data:
        assert len(state) == 7 and all(len(col) == 5 for col in state)
        assert any(targets)
        assert all(not target or state[column][0] == 0 for column, target in enumerate(targets))
        hints = tactical_actions(tuple(tuple(col) for col in state))
        if weight == 2:
            assert {i for i, target in enumerate(targets) if target} == set(hints)
