"""One process owns all weights and serializes inference/training with a lock."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import deque
import logging
import os
import time
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
import torch

from .dqn import ARCHITECTURE, GAMMA, Agent, outcome_reward, select_device
from .evaluation import BENCHMARK, EVALUATION_INTERVAL, evaluate
from .game import drop, encode, legal_moves, next_player, outcome, validate_board

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
log = logging.getLogger(__name__)


class BoardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: Name
    board: list[list[StrictInt]]

    @field_validator("board")
    @classmethod
    def valid_board(cls, board):
        validate_board(board)
        return board


class MoveRequest(BoardRequest):
    player: Literal[1, -1]
    model_id: Name = "shared"
    train: bool = True
    explore: bool = False


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Name = "shared"


class ExplorationResetRequest(ModelRequest):
    rate: float = Field(default=1.0, strict=True, ge=0.05, le=1.0, allow_inf_nan=False)


class InspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Name = "shared"
    board: list[list[StrictInt]]
    player: Literal[1, -1]
    simulations: int = Field(default=64, strict=True, ge=8, le=512)

    @field_validator("board")
    @classmethod
    def valid_board(cls, board):
        validate_board(board)
        return board


class DebugControlRequest(ModelRequest):
    action: Literal["pause", "resume", "evaluate", "checkpoint", "delay"]
    delay_ms: int = Field(default=500, strict=True, ge=0, le=60000)


@dataclass
class Session:
    expected: list
    bindings: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    responses: dict = field(default_factory=dict)
    finished: bool = False
    trajectories: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class Trainer:
    def __init__(self, model_dir, device="auto", n_steps=3):
        self.model_dir = Path(model_dir)
        self.device = select_device(device)
        self.agents = {}
        self.games = {}
        self.lock = RLock()
        self.n_steps = n_steps
        self.errors = {"training": 0, "checkpoint": 0, "evaluation": 0, "requests": 0}
        self.last_error = None
        self.request_metrics = deque(maxlen=100)

    def attempt(self, kind, operation):
        try:
            return operation()
        except Exception as exc:
            self.errors[kind] += 1
            self.last_error = f"{kind}: {type(exc).__name__}: {exc}"
            log.exception("%s operation failed; game state remains usable", kind)
            return None

    def agent(self, name):
        if name not in self.agents:
            if len(self.agents) >= 32:
                raise HTTPException(409, "32 models already loaded; restart to unload them")
            self.agents[name] = Agent(self.model_dir / f"{name}.pt", device=self.device)
        return self.agents[name]

    def observe(self, session, board):
        if board == session.expected:
            return
        expected = session.expected
        if outcome(expected) is None:
            player = next_player(expected)
            if any(drop(expected, c, player) == board for c in legal_moves(expected)):
                return  # One external/human move since the last API action.
        raise HTTPException(409, "Board does not confirm the last action plus at most one opponent move")

    def settle(self, session, board, player, terminal):
        pending = session.pending.pop(player, None)
        if pending is not None:
            model_id, state, action = pending
            reward = outcome_reward(board, player, terminal)
            queue = session.trajectories.setdefault(player, deque())
            queue.append((state, action, reward, board, terminal))
            while len(queue) >= self.n_steps or (terminal and queue):
                window = list(queue)[:self.n_steps]
                first, last = window[0], window[-1]
                total = sum(GAMMA ** i * item[2] for i, item in enumerate(window))
                self.agent(model_id).remember(first[0], first[1], total, last[3], player,
                                               last[4], steps=len(window))
                queue.popleft()

    def train_models(self, names):
        return {name: self.attempt("training", self.agent(name).train_step) for name in sorted(names)}

    def checkpoint_agent(self, agent):
        self.attempt("checkpoint", agent.save)

    def train_agent(self, agent):
        return self.attempt("training", agent.train_step)

    def measure(self, agent, force=False):
        previous = agent.evaluations[-1] if agent.evaluations else None
        if (previous is None or previous["benchmark"] != BENCHMARK
                or force or agent.updates - previous["updates"] >= EVALUATION_INTERVAL):
            point = evaluate(agent)
            # A repeated measurement at the same weights replaces its existing point.
            if previous and previous["updates"] == point["updates"] and previous["benchmark"] == BENCHMARK:
                agent.evaluations.pop()
            agent.evaluations.append(point)
            return True
        return False

    def progress(self, name, force=False):
        with self.lock:
            agent = self.agent(name)
            if self.attempt("evaluation", lambda: self.measure(agent, force)):
                self.attempt("checkpoint", agent.save)
            return {"model_id": name, "updates": agent.updates,
                    "architecture": ARCHITECTURE, "input_shape": [7, 5],
                    "observations": agent.observations,
                    "pretraining": agent.pretraining,
                    "diagnostics": agent.diagnostics(), "service_errors": dict(self.errors),
                    "last_service_error": self.last_error,
                    "exploration_rate": agent.exploration_rate,
                    "exploration_decisions": agent.decisions,
                    "evaluation_interval": EVALUATION_INTERVAL,
                    "history": [point for point in agent.evaluations if point["benchmark"] == BENCHMARK]}

    def move(self, request):
        with self.lock:
            key = request.model_dump_json()
            session = self.games.get(request.game_id)
            if session and key in session.responses:
                return session.responses[key]  # Camera/network retries never add experience twice.
            if session and session.finished:
                raise HTTPException(409, "Game finished; use a new game_id")
            if outcome(request.board) is not None:
                raise HTTPException(409, "Game is over; send this board to /finish")
            if next_player(request.board) != request.player:
                raise HTTPException(409, "Wrong player: player 1 always starts")
            if request.explore and not request.train:
                raise HTTPException(422, "Exploration requires train=true")
            if session:
                self.observe(session, request.board)
                binding = session.bindings.get(request.player)
                if binding is not None and binding != (request.model_id, request.train):
                    raise HTTPException(409, "Keep each player's model_id and train setting fixed within a game")
            else:
                if len(self.games) >= 1000:
                    raise HTTPException(409, "1000 games retained; DELETE old games to free sessions")
                session = Session(request.board)
            agent = self.agent(request.model_id)
            self.games[request.game_id] = session
            session.bindings[request.player] = (request.model_id, request.train)
            # Transition spans our action AND the opponent's reply: next state is our turn.
            self.settle(session, request.board, request.player, terminal=False)
            loss = self.train_agent(agent) if request.train else None
            if request.train and agent.evaluations:
                self.attempt("evaluation", lambda: self.measure(agent))
            column, epsilon = agent.choose(request.board, request.player, request.explore)
            if request.train:
                session.pending[request.player] = (request.model_id, encode(request.board, request.player), column)
            predicted = drop(request.board, column, request.player)
            session.expected = predicted
            result = {"game_id": request.game_id, "model_id": request.model_id,
                      "player": request.player, "column": column, "epsilon": epsilon,
                      "predicted_board": predicted, "done": outcome(predicted) is not None,
                      "winner": outcome(predicted), "loss": loss,
                      "replay_size": len(agent.memory), "updates": agent.updates,
                      "device": str(agent.device)}
            session.responses[key] = result
            if request.train:
                self.checkpoint_agent(agent)
            result["diagnostics"] = agent.diagnostics()
            result["service_errors"] = dict(self.errors)
            return result

    def finish(self, request):
        with self.lock:
            session = self.games.get(request.game_id)
            if session is None:
                raise HTTPException(404, "Unknown game_id")
            if outcome(request.board) is None:
                raise HTTPException(409, "The final board must be a win or draw")
            self.observe(session, request.board)
            if not session.finished:
                names = {pending[0] for pending in session.pending.values()}
                for player in (1, -1):
                    self.settle(session, request.board, player, terminal=True)
                self.train_models(names)
                session.expected = request.board
                session.finished = True
                for name in names:
                    if self.agent(name).evaluations:
                        self.attempt("evaluation", lambda: self.measure(self.agent(name)))
                    self.checkpoint_agent(self.agent(name))
            return {"game_id": request.game_id, "done": True, "winner": outcome(request.board)}

    def save(self):
        with self.lock:
            for agent in self.agents.values():
                self.attempt("checkpoint", agent.save)


def create_app(model_dir=None, device=None, backend=None, background=None):
    torch.set_num_threads(1)
    backend = backend or os.getenv("AI_BACKEND", "az")
    if backend not in ("az", "dqn"):
        raise ValueError("AI_BACKEND must be az or dqn")
    trainer_type = Trainer
    if backend == "az":
        from .search_trainer import SearchTrainer
        trainer_type = SearchTrainer
    trainer = trainer_type(model_dir or os.getenv("MODEL_DIR", "models"),
                           device if device is not None else os.getenv("DQN_DEVICE", "auto"))
    trainer.background_enabled = (os.getenv("AI_BACKGROUND", "true").lower() == "true"
                                  if background is None else background)

    @asynccontextmanager
    async def lifespan(app):
        if backend == "az":
            trainer.start()
        try:
            yield
        finally:
            if backend == "az":
                trainer.close()
            trainer.save()

    app = FastAPI(title="Connect Four Learning AI", lifespan=lifespan)
    app.state.trainer = trainer

    @app.middleware("http")
    async def record_errors(request, call_next):
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
            if request.url.path in ("/move", "/finish", "/debug/analyze"):
                with trainer.lock:
                    trainer.request_metrics.append({"timestamp": time.time(), "path": request.url.path,
                        "status": response.status_code, "milliseconds": (time.perf_counter() - started_at) * 1000})
            return response
        except Exception as exc:
            with trainer.lock:
                trainer.errors["requests"] += 1
                trainer.last_error = f"{request.url.path}: {type(exc).__name__}: {exc}"
            log.exception("Request failed: %s", request.url.path)
            raise

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(trainer.device),
                "cuda_available": torch.cuda.is_available(), "backend": backend,
                "background": trainer.background_status() if backend == "az" else None}

    @app.get("/models/{model_id}/curriculum")
    def curriculum(model_id: Name):
        with trainer.lock:
            points = [p for p in trainer.agent(model_id).evaluations if p["benchmark"] == BENCHMARK][-5:]
            score = sum(p["score"] for p in points) / len(points) if points else 0.0
            return {"rule_probability": 0.5 + 0.5 * min(score / 0.6, 1.0), "benchmark_score": score}

    @app.get("/debug/state/{model_id}")
    def debug_state(model_id: Name):
        if backend != "az":
            raise HTTPException(409, "Debug console requires AI_BACKEND=az")
        return trainer.debug_state(model_id)

    @app.post("/debug/control")
    def debug_control(request: DebugControlRequest):
        if backend != "az":
            raise HTTPException(409, "Debug console requires AI_BACKEND=az")
        return trainer.control(request.model_id, request.action, request.delay_ms)

    @app.post("/debug/analyze")
    def analyze(request: InspectRequest):
        if backend != "az":
            raise HTTPException(409, "Move inspection requires AI_BACKEND=az")
        if outcome(request.board) is not None or next_player(request.board) != request.player:
            raise HTTPException(409, "Choose an unfinished board and its next player")
        return trainer.analyze(request.model_id, request.board, request.player, request.simulations)

    @app.post("/move")
    def move(request: MoveRequest):
        return trainer.move(request)

    @app.post("/finish")
    def finish(request: BoardRequest):
        return trainer.finish(request)

    @app.delete("/games/{game_id}")
    def discard(game_id: str):
        with trainer.lock:
            session = trainer.games.pop(game_id, None)
            if session is None:
                raise HTTPException(404, "Unknown game_id")
            if backend == "az":
                trainer.record_discard(game_id, session)
        return {"deleted": game_id}

    @app.get("/model-catalog")
    def model_catalog():
        if backend != "az":
            raise HTTPException(409, "Model catalog requires the search backend")
        rows = trainer.catalog.scan()
        with trainer.lock:
            agent = trainer.agent("shared")
            published = [p for p in agent.evaluations if p.get("updates") == agent.published_updates and p.get("promoted", True)]
            rows.insert(0, {"id":"shared","label":"Current published model (live)","playable":True,
                            "generation":agent.generation,"updates":agent.published_updates,
                            "benchmark":published[-1] if published else None})
            training = trainer.workers["shared"].snapshot()
        return {"models":rows,"training":training,"training_model":"shared"}

    @app.get("/models")
    def models():
        with trainer.lock:
            return {name: {"updates": a.updates, "replay_size": len(a.memory),
                           "exploration_decisions": a.decisions, "device": str(a.device)}
                    for name, a in trainer.agents.items()}

    @app.post("/checkpoint")
    def checkpoint():
        trainer.save()
        return {"saved": all(a.checkpoint_error is None for a in trainer.agents.values()),
                "errors": dict(trainer.errors),
                "learner_checkpoint_pending": any(w.checkpoint.is_set() for w in trainer.workers.values())
                    if backend == "az" else False}

    @app.get("/models/{model_id}/progress")
    def progress(model_id: Name):
        return trainer.progress(model_id)

    @app.post("/evaluate")
    def measure(request: ModelRequest):
        return trainer.progress(request.model_id, force=True)

    @app.post("/exploration/reset")
    def reset_exploration(request: ExplorationResetRequest):
        with trainer.lock:
            agent = trainer.agent(request.model_id)
            agent.reset_exploration(request.rate)
            trainer.checkpoint_agent(agent)
            return {"model_id": request.model_id, "exploration_decisions": 0,
                    "exploration_rate": agent.exploration_rate,
                    "checkpoint_saved": agent.checkpoint_error is None}

    return app


app = create_app()
