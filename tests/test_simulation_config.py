from contextlib import nullcontext
from threading import Event, Thread

from fastapi.testclient import TestClient
import pytest
import torch

from connect4.api import create_app as create_api
from connect4.simulation_service import Simulation, SimulationConfig, create_app


@pytest.fixture
def services(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    with TestClient(create_api(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        with TestClient(create_app(run_worker=False)) as viewer:
            yield viewer, api.app.state.trainer


def test_configure_both_ais_or_either_side_as_rules_preserves_learning(services):
    viewer, trainer = services
    learner = trainer.agent("shared")
    learner.decisions, learner.updates = 100, 25
    weights = [parameter.detach().clone() for parameter in learner.online.parameters()]
    for one, two, swap in [("shared", "shared", False), ("alpha", "beta", True),
                           ("rules", "shared", False), ("shared", "rules", True)]:
        config = {"model_one": one, "model_two": two, "swap_sides": swap}
        response = viewer.post("/simulation/config", json=config)
        assert response.status_code == 200, response.text
        state = response.json()
        assert state["config"] == SimulationConfig(**config).model_dump()
        assert state["mode"] == "simulation"
        assert state["model_one"] == one and state["model_two"] == two
        assert viewer.get("/state").json()["config"] == SimulationConfig(**config).model_dump()
    assert learner.decisions == 100 and learner.updates == 25
    assert all(torch.equal(a, b) for a, b in zip(weights, learner.online.parameters()))


@pytest.mark.parametrize("change", [
    {"model_one": "rules", "model_two": "rules"},
    {"model_one": "../outside"}, {"model_two": ""}, {"model_two": "a" * 65},
    {"swap_sides": "false"}, {"swap_sides": 1}, {"unknown": True},
])
def test_invalid_configuration_leaves_active_game_untouched(services, change):
    viewer, _ = services
    before = viewer.post("/human/new").json()
    config = {"model_one": "shared", "model_two": "rules", "swap_sides": True, **change}
    assert viewer.post("/simulation/config", json=config).status_code == 422
    assert viewer.get("/state").json() == before


def test_configuration_discards_human_session_and_uses_selected_ai(services):
    viewer, trainer = services
    game = viewer.post("/human/new").json()
    response = viewer.post("/human/move", json={"game_id": game["game_id"], "move_count": 0, "column": 0})
    assert response.status_code == 200
    assert trainer.games and not trainer.agent("shared").memory
    config = {"model_one": "rules", "model_two": "new-ai", "swap_sides": False}
    assert viewer.post("/simulation/config", json=config).status_code == 200
    assert not trainer.games and not trainer.agent("shared").memory
    human = viewer.post("/human/new").json()
    assert human["model_two"] == "new-ai" and human["config"] == SimulationConfig(**config).model_dump()
    assert viewer.post("/human/move", json={"game_id": human["game_id"], "move_count": 0, "column": 0}).status_code == 200
    assert trainer.games[human["game_id"]].bindings[-1] == ("new-ai", True)


@pytest.mark.parametrize("models,swap,expected", [
    (("alpha", "beta"), True, [("alpha", "beta"), ("beta", "alpha")]),
    (("alpha", "beta"), False, [("alpha", "beta"), ("alpha", "beta")]),
    (("rules", "alpha"), False, [("rules", "alpha"), ("rules", "alpha")]),
    (("rules", "alpha"), True, [("rules", "alpha"), ("alpha", "rules")]),
])
def test_worker_respects_player_selection_and_switching(tmp_path, monkeypatch, models, swap, expected):
    import connect4.simulation_service as module
    simulation = Simulation("http://testserver", *models, interval=0, result_delay=0, swap_sides=swap)
    # Prior human results must not determine which simulation side starts.
    simulation.update({"games": 7})
    starts = []
    original_update = simulation.update

    def update(changes):
        original_update(changes)
        if changes.get("move_count") == 0:
            starts.append((changes["model_one"], changes["model_two"]))
        if changes.get("games") == 9:
            simulation.stop.set()

    monkeypatch.setattr(simulation, "update", update)
    with TestClient(create_api(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        simulation.run()
        assert starts == expected
        assert not api.app.state.trainer.games
        assert set(api.app.state.trainer.agents) == set(models) - {"rules"}


def test_live_reconfiguration_cancels_old_game_before_new_players_start(tmp_path, monkeypatch):
    import connect4.simulation_service as module
    simulation = Simulation("http://testserver", interval=0.05, result_delay=0)
    old_move, new_start = Event(), Event()
    original_update = simulation.update

    def update(changes):
        original_update(changes)
        if changes.get("move_count") == 1:
            old_move.set()
        if changes.get("move_count") == 0 and changes.get("model_one") == "alpha":
            new_start.set()

    monkeypatch.setattr(simulation, "update", update)
    with TestClient(create_api(tmp_path, device="cpu")) as api:
        monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(api))
        worker = Thread(target=simulation.run)
        worker.start()
        try:
            assert old_move.wait(5)
            config = SimulationConfig(model_one="alpha", model_two="beta", swap_sides=False)
            result = simulation.configure(config)
            assert result["config"] == config.model_dump()
            assert new_start.wait(5)
            assert simulation.snapshot()["model_one"] == "alpha"
            with api.app.state.trainer.lock:
                assert all(binding[0] != "shared" for game in api.app.state.trainer.games.values()
                           for binding in game.bindings.values())
        finally:
            simulation.stop.set()
            simulation.game_cancel.set()
            worker.join(5)
        assert not worker.is_alive()
        assert not api.app.state.trainer.games


def test_configure_restarts_switching_phase(services):
    viewer, _ = services
    simulation = viewer.app.state.simulation
    simulation.simulation_games = 5
    config = {"model_one": "rules", "model_two": "shared", "swap_sides": True}
    viewer.post("/simulation/config", json=config)
    assert simulation.simulation_games == 0
    assert simulation.models == ("rules", "shared")


def test_switching_environment_default(monkeypatch):
    monkeypatch.setenv("SIM_SWAP_SIDES", "false")
    with TestClient(create_app(run_worker=False)) as viewer:
        assert viewer.get("/state").json()["config"]["swap_sides"] is False


def test_round_robin_covers_pairs_in_15_game_blocks_and_pauses():
    simulation = Simulation("http://unused")
    config = SimulationConfig(model_one="alpha", model_two="beta", model_three="rules",
                              round_robin=True, swap_sides=True)
    simulation.configure(config)
    blocks = []
    for game in range(135):
        simulation.simulation_games = game
        pair = simulation.select_pair()
        if game % 15 == 0:
            blocks.append(pair)
        assert pair == blocks[-1]
        assert simulation.select_pair() == pair  # Retries/unfinished games cannot advance.
        assert simulation.snapshot()["rotation"]["game"] == game % 15 + 1
    for start in (0, 3, 6):
        assert set(blocks[start:start + 3]) == {("alpha", "beta"), ("alpha", "rules"), ("beta", "rules")}
    assert blocks[2] != blocks[3] and blocks[5] != blocks[6]
    simulation.set_mode("human")
    assert simulation.simulation_games == 134
    assert simulation.snapshot()["rotation"]["pair"] == list(blocks[-1])


@pytest.mark.parametrize("third", [None, "rules", "../bad"])
def test_round_robin_invalid_roster_rejected(services, third):
    viewer, _ = services
    before = viewer.get("/state").json()
    response = viewer.post("/simulation/config", json={"model_one": "shared", "model_two": "rules",
        "model_three": third, "round_robin": True, "swap_sides": True})
    assert response.status_code == 422
    assert viewer.get("/state").json() == before


def test_round_robin_progress_includes_idle_third_ai(services):
    viewer, _ = services
    viewer.post("/simulation/config", json={"model_one": "alpha", "model_two": "beta",
        "model_three": "gamma", "round_robin": True, "swap_sides": True})
    response = viewer.post("/exploration/reset", json={"rate": 0.25})
    assert set(response.json()["models"]) == {"alpha", "beta", "gamma"}


@pytest.mark.parametrize("swap", [False, True])
def test_shared_round_robin_worker_rotation_results_and_interruptions(monkeypatch, swap):
    import connect4.simulation_service as module
    simulation = Simulation("http://unused", result_delay=0)
    simulation.configure(SimulationConfig(model_one="shared", model_two="rules", model_three="shared",
                                          round_robin=True, swap_sides=swap, curriculum=False))
    monkeypatch.setattr(module.random, "shuffle", lambda queue: None)
    # Start a known round after configure's random first selection.
    simulation.pair_queue = []
    simulation.pair_block = -1
    starts = []
    interrupted = False

    def play(client, one, two, **kwargs):
        nonlocal interrupted
        # An aborted game at the block boundary must neither count nor rotate twice.
        if len(starts) == 15 and not interrupted:
            interrupted = True
            raise InterruptedError()
        starts.append((one, two, tuple(simulation.active_slots)))
        if len(starts) == 45:
            simulation.stop.set()
        return (1, 0, -1)[(len(starts) - 1) % 3]

    monkeypatch.setattr(module, "play_game", play)
    monkeypatch.setattr(module, "post", lambda *args: {})
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: nullcontext(None))
    simulation.run()
    assert simulation.simulation_games == 45 and interrupted
    assert len({starts[i][2] for i in (0, 15, 30)}) == 3
    for i, (one, two, slots) in enumerate(starts):
        assert slots == starts[i // 15 * 15][2]
        expected = tuple(simulation.models[j] for j in slots)
        assert (one, two) == (expected[::-1] if swap and i % 2 else expected)
    matrix = simulation.snapshot()["matchups"]
    assert sum(sum(counts.values()) for row in matrix.values() for counts in row.values()) == 45
    assert matrix["shared"]["shared"] == {"1": 5, "0": 5, "-1": 5}
    simulation.configure(SimulationConfig(model_one="shared", model_two="rules", swap_sides=True))
    assert simulation.snapshot()["matchups"] == {}
