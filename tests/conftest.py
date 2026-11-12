"""Keep the pre-existing DQN regression suite on its explicit legacy backend.

Policy/value integration tests pass backend="az" to test the new default engine.
"""
import pytest


@pytest.fixture(autouse=True)
def legacy_backend(monkeypatch):
    monkeypatch.setenv("AI_BACKEND", "dqn")
