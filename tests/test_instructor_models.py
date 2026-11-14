import hashlib
import time
from threading import RLock, Thread, Event

import torch
from fastapi.testclient import TestClient

from connect4.az import Agent
from connect4.game import empty_board
from connect4.learning import Worker, generate
from connect4.model_catalog import Catalog


def test_every_training_move_uses_the_pacing_callback(tmp_path):
    agent = Agent(tmp_path / 'az-shared.pt', device='cpu', simulations=2)
    waits, positions = [], []
    result = generate(agent, games=2, simulations=2,
                      before_move=lambda: waits.append(1) or True,
                      on_position=lambda s: positions.append(s['ply']))
    assert len(result) == 2
    assert len(waits) == len(positions) > 2


def test_pause_and_delay_changes_apply_during_a_game(tmp_path):
    worker = Worker(Agent(tmp_path / 'az-shared.pt', device='cpu'), RLock(), move_delay=0.5)
    done = Event()
    worker.paused.set()
    thread = Thread(target=lambda: worker.wait_before_move() and done.set())
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while worker.snapshot()['phase'] != 'paused' and time.monotonic() < deadline:
            time.sleep(.01)
        worker.move_delay = 0
        assert not done.wait(.1)  # A delay change cannot resume paused learning.
        worker.paused.clear()
        assert done.wait(1)
    finally:
        worker.stop.set()
        thread.join(2)


def test_generations_are_archived_and_saved_models_never_train(tmp_path, monkeypatch):
    monkeypatch.setenv('REPO_MODEL_RUNTIME', str(tmp_path / 'absent'))
    from connect4.api import create_app
    model = Agent(tmp_path / 'az-shared.pt', device='cpu')
    model.save_generation()
    model.generation, model.published_updates = 1, 42
    model.save_generation()
    archives = sorted((tmp_path / 'published').rglob('*.pt'))
    assert len(archives) == 2
    original = hashlib.sha256(archives[-1].read_bytes()).hexdigest()
    model.save_generation()
    assert hashlib.sha256(archives[-1].read_bytes()).hexdigest() == original
    catalog = Catalog(tmp_path)
    row = next(r for r in catalog.scan() if r['generation'] == 1)
    with TestClient(create_app(model_dir=tmp_path, device='cpu', backend='az', background=False)) as client:
        listed = client.get('/model-catalog').json()
        assert row['id'] in [m['id'] for m in listed['models']]
        response = client.post('/move', json={'game_id':'frozen-test','board':empty_board(),
                                              'player':1,'model_id':row['id'],'train':True})
        assert response.status_code == 200, response.text
        assert response.json()['column'] in range(7)
        assert response.json()['diagnostics']['read_only']
        assert client.post('/checkpoint').status_code == 200
    assert hashlib.sha256(archives[-1].read_bytes()).hexdigest() == original


def test_legacy_model_is_pickable_without_migration(tmp_path, monkeypatch):
    from connect4.dqn import Agent as LegacyAgent
    monkeypatch.setenv('REPO_MODEL_RUNTIME', str(tmp_path / 'absent'))
    legacy = LegacyAgent(tmp_path / 'shared.pt', device='cpu')
    legacy.save()
    original = legacy.path.read_bytes()
    catalog = Catalog(tmp_path)
    row = catalog.scan()[0]
    frozen = catalog.load(row['id'], 'cpu', 2)
    assert frozen.choose(empty_board(), 1, False)[0] in range(7)
    frozen.save()
    assert legacy.path.read_bytes() == original
