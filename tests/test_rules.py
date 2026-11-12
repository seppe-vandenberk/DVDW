from contextlib import nullcontext
import random

from fastapi.testclient import TestClient
import pytest
import torch

from connect4.api import create_app
from connect4.dqn import Agent
from connect4.game import drop, empty_board, encode, legal_moves, outcome, validate_board
from connect4.rules import choose_rule_move, sequence_value, winning_moves
from connect4.simulation_service import Simulation
from connect4.simulator import play_game


def board_after(moves):
    board = empty_board()
    for index, column in enumerate(moves):
        board = drop(board, column, 1 if index % 2 == 0 else -1)
    validate_board(board)
    return board


@pytest.mark.parametrize("moves,column", [
    ([0, 6, 1, 6, 2, 5], 3),
    ([0, 6, 0, 6, 0, 5], 0),
    ([0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3], 3),
    ([6, 5, 5, 4, 2, 4, 4, 3, 2, 3, 1, 3], 3),
])
def test_rules_complete_four_in_every_direction(moves, column):
    board = board_after(moves)
    chosen, reason = choose_rule_move(board, 1, random.Random(0))
    assert chosen == column
    assert outcome(drop(board, chosen, 1)) == 1
    assert reason == "Complete four in a row"


def test_win_has_priority_over_block():
    board = board_after([0, 6, 1, 6, 2, 6])
    assert winning_moves(board, -1) == [6]
    assert choose_rule_move(board, 1)[0] == 3


def test_block_immediate_win():
    board = board_after([6, 0, 6, 1, 5, 2])
    assert choose_rule_move(board, 1)[0] == 3


def test_defend_pair_when_own_discs_are_singles():
    board = board_after([0, 5, 2, 6])
    assert sequence_value(board, 1)[0] == 1
    assert sequence_value(board, -1)[0] == 2
    column, reason = choose_rule_move(board, 1)
    assert column in (3, 4)
    assert "opposing pair" in reason
    assert sequence_value(drop(board, column, 1), -1)[1] < sequence_value(board, -1)[1]


def test_extend_own_pair_and_prefer_center():
    board = board_after([1, 6, 2, 6])
    column, reason = choose_rule_move(board, 1)
    assert column == 3
    assert sequence_value(drop(board, column, 1), 1)[0] == 3
    assert "Extend" in reason


@pytest.mark.parametrize("player,board", [(1, empty_board()), (-1, board_after([3]))])
def test_rules_randomize_first_move_for_either_starting_side(player, board):
    openings = set()
    for seed in range(40):
        result = choose_rule_move(board, player, random.Random(seed))
        assert result == choose_rule_move(board, player, random.Random(seed))
        column, reason = result
        assert column in legal_moves(board) and reason == "Random opening move"
        openings.add(column)
    assert openings == set(range(7))
    # After placing its first disc, the player goes back to the tactical rules.
    after = drop(board, 0, player)
    after = drop(after, 6, -player)
    assert choose_rule_move(after, player, random.Random(0))[1] != "Random opening move"


def test_ignore_dead_sequences_and_unplayable_floating_finishes():
    board = empty_board()
    for column, value in enumerate([-1, 1, 1, 1, -1]):
        board[column][-1] = value
    assert sequence_value(board, 1)[0] == 1
    board = empty_board()
    for column in (0, 1, 2):
        board[column][-2:] = [1, -1]
    assert winning_moves(board, 1) == []
    column, reason = choose_rule_move(board, 1)
    assert column == 3 and reason == "Block an immediate win"


def test_avoid_supporting_an_opponents_floating_win():
    board = empty_board()
    board[0][-2:] = [-1, 1]
    board[1][-2:] = [-1, 1]
    board[2][-2:] = [-1, -1]
    board[4][-1] = board[5][-1] = 1
    validate_board(board)
    assert winning_moves(board, -1) == []
    assert winning_moves(drop(board, 3, 1), -1) == [3]
    assert choose_rule_move(board, 1)[0] != 3


def test_full_columns_are_excluded_and_finished_games_rejected():
    board = board_after([0, 0, 0, 0, 0])
    assert choose_rule_move(board, -1)[0] in legal_moves(board)
    assert choose_rule_move(board, -1)[0] != 0
    with pytest.raises(ValueError, match="finished"):
        choose_rule_move(board_after([0, 6, 1, 6, 2, 5, 3]), -1)


@pytest.mark.parametrize("models", [("shared", "rules"), ("rules", "shared")])
def test_rule_games_train_only_ai_and_settle_final_reward(tmp_path, models):
    frames = []
    with TestClient(create_app(tmp_path, device="cpu")) as api:
        winner = play_game(api, *models, interval=0, on_update=frames.append)
        trainer = api.app.state.trainer
        assert set(trainer.agents) == {"shared"}
        assert not trainer.games
        agent = trainer.agents["shared"]
        assert len(agent.memory) == sum(frame.get("actor") == "shared" for frame in frames)
        ai_player = 1 if models[0] == "shared" else -1
        discs = sum(v == ai_player for col in frames[-1]["board"] for v in col)
        expected = (0 if winner == 0 else 1 if winner == ai_player else -1) * (1 - 0.01 * discs)
        assert agent.memory[-1][2] == pytest.approx(expected)
        assert agent.memory[-1][-1]
        assert agent.decisions == len(agent.memory)
        for frame in frames[1:]:
            validate_board(frame["board"])
            if frame["actor"] == "rules":
                assert frame["move_reason"] and "epsilon" not in frame
        rule_frames = [frame for frame in frames if frame.get("actor") == "rules"]
        assert rule_frames[0]["move_reason"] == "Random opening move"
        assert all(frame["move_reason"] != "Random opening move" for frame in rule_frames[1:])
        assert not (tmp_path / "rules.pt").exists()


def test_worker_alternates_ai_and_rules_colors(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    simulation = Simulation("http://testserver", interval=0, result_delay=0)
    starts = []
    original_update = simulation.update

    def update(changes):
        original_update(changes)
        if changes.get("move_count") == 0:
            starts.append((changes["model_one"], changes["model_two"]))
        if changes.get("games") == 2:
            simulation.stop.set()

    monkeypatch.setattr(simulation, "update", update)
    with TestClient(create_app(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        simulation.run()
        assert starts == [("shared", "rules"), ("rules", "shared")]
        assert set(api.app.state.trainer.agents) == {"shared"}


def test_queued_reset_deletes_old_training_state_once(tmp_path):
    path = tmp_path / "shared.pt"
    agent = Agent(path, device="cpu")
    agent.updates, agent.decisions = 500, 9000
    agent.remember(encode(empty_board(), 1), 0, -0.01, empty_board(), 1, False)
    agent.evaluations.append({"benchmark": "rules-v1", "updates": 500})
    agent.save()
    previous_weights = [parameter.detach().clone() for parameter in agent.online.parameters()]
    marker = path.with_suffix(".reset")
    marker.write_text("Start from scratch on next API load")
    fresh = Agent(path, device="cpu")
    assert fresh.updates == fresh.decisions == 0
    assert not fresh.memory and not fresh.evaluations and not fresh.optimizer.state
    assert any(not torch.equal(a, b) for a, b in zip(previous_weights, fresh.online.parameters()))
    assert path.exists() and not marker.exists()
    fresh.decisions = 7
    fresh.save()
    resumed = Agent(path, device="cpu")
    assert resumed.decisions == 7
    assert all(torch.equal(a, b) for a, b in zip(fresh.online.parameters(), resumed.online.parameters()))
