"""Exercise the actual background thread without a long strength tournament."""
from threading import RLock
import time

import torch

from connect4.az import Agent
from connect4.game import empty_board
from connect4.learning import Worker


def test_worker_trains_publishes_and_persists_human_outcomes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    def tournament(*args, **kwargs):
        return {"games": 4, "wins": 3, "draws": 0, "losses": 1, "score": 0.75,
                "win_rate": 0.75, "positions": {"first": {"wins": 2, "draws": 0, "losses": 0},
                "second": {"wins": 1, "draws": 0, "losses": 1}}}
    monkeypatch.setattr("connect4.learning.match", tournament)
    serving = Agent(tmp_path / "az-shared.pt", device="cpu", simulations=2)
    worker = Worker(serving, RLock(), batch_games=8, simulations=2, interval=0.01, gate_games=4, move_delay=0)
    worker.submit([(empty_board(), [0, 0, 0, 1, 0, 0, 0], 1, 0)], 1, 6)
    worker.start()
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline and serving.generation == 0 and worker.thread.is_alive():
            worker.stop.wait(0.02)
    finally:
        worker.close()
    assert not worker.thread.is_alive()
    assert worker.status["error"] is None
    assert serving.generation > 0 and serving.updates > 0
    assert any(p.get("promoted") for p in serving.evaluations)
    restored = Agent(serving.path.with_suffix(".learner.pt"), device="cpu")
    assert len(restored.human_memory) == 1
    assert restored.updates > 0


def test_failed_promotion_keeps_published_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr("connect4.learning.match", lambda *args, **kwargs:
        {"games": 4, "wins": 0, "draws": 0, "losses": 4, "score": 0.0, "win_rate": 0.0})
    serving = Agent(tmp_path / "az-shared.pt", device="cpu", simulations=2)
    before = {k: v.clone() for k, v in serving.online.state_dict().items()}
    worker = Worker(serving, RLock(), batch_games=8, simulations=2, interval=0.01, gate_games=4, move_delay=0)
    worker.start()
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline and len(serving.evaluations) < 2 and worker.thread.is_alive():
            worker.stop.wait(0.02)
    finally:
        worker.close()
    assert worker.status["error"] is None and len(serving.evaluations) >= 2
    assert serving.generation == 0
    assert all(torch.equal(before[k], v) for k, v in serving.online.state_dict().items())


def test_pause_preserves_weights_and_checkpoint_works_while_paused(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    serving = Agent(tmp_path / "az-paused.pt", device="cpu", simulations=2)
    serving.evaluations.append({"stage": "baseline", "updates": 0, "timestamp": time.time(), "promoted": True})
    worker = Worker(serving, RLock(), batch_games=4, simulations=2, move_delay=0)
    worker.paused.set()
    worker.start()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and worker.snapshot()["phase"] != "paused":
            worker.stop.wait(0.02)
        assert worker.snapshot()["phase"] == "paused"
        assert serving.updates == 0 and worker.status["games"] == 0
        worker.checkpoint.set()
        while time.monotonic() < deadline and not worker.snapshot()["learner_checkpoint"]["saved_at"]:
            worker.stop.wait(0.02)
        assert worker.snapshot()["learner_checkpoint"]["saved_at"] is not None
        assert serving.updates == 0 and worker.status["games"] == 0
        worker.paused.clear()
        while time.monotonic() < deadline and worker.snapshot()["games"] == 0:
            worker.stop.wait(0.02)
        assert worker.snapshot()["games"] > 0
    finally:
        worker.close()
    assert worker.status["error"] is None
