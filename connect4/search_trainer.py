"""Search learner behind the existing validated, idempotent camera contract."""
import os
import queue
import time
from copy import deepcopy

from fastapi import HTTPException

from .api import Trainer
from .az import Agent, ARCHITECTURE, search_many
from .bitboard import Position
from .learning import Worker, BENCHMARK
from .game import outcome


class SearchTrainer(Trainer):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        self.workers = {}
        self.started = False
        self.background_enabled = True

    def agent(self, name):
        if name not in self.agents:
            if len(self.agents) >= 4:
                raise HTTPException(409, "At most four search models can be loaded")
            agent = Agent(self.model_dir / f"az-{name}.pt", device=self.device,
                          simulations=int(os.getenv("AI_SEARCH_SIMULATIONS", "64")))
            self.agents[name] = agent
            worker = Worker(agent, self.lock,
                            batch_games=int(os.getenv("AI_BATCH_GAMES", "16")),
                            simulations=int(os.getenv("AI_TRAIN_SIMULATIONS", "32")),
                            interval=float(os.getenv("AI_EVALUATION_SECONDS", "300")),
                            gate_games=int(os.getenv("AI_GATE_GAMES", "40")))
            self.workers[name] = worker
            if self.started and self.background_enabled:
                worker.start()
        return self.agents[name]

    def start(self):
        self.started = True
        self.agent("shared")
        if self.background_enabled:
            for worker in self.workers.values():
                if worker.thread.ident is None:
                    worker.start()

    def close(self):
        for worker in self.workers.values():
            worker.close()

    def save(self):
        for worker in self.workers.values():
            if worker.thread.is_alive():
                worker.checkpoint.set()
        super().save()

    def background_status(self):
        return {name: worker.snapshot() for name, worker in self.workers.items()}

    def checkpoint_agent(self, agent):
        # Physical requests never serialize replay or optimizer state.
        pass

    def train_agent(self, agent):
        return None

    def measure(self, agent, force=False):
        return False

    def train_models(self, names):
        return {name: None for name in names}

    def settle(self, session, board, player, terminal):
        pending = session.pending.pop(player, None)
        if pending is not None:
            model_id, state, action = pending
            policies = getattr(session, "search_policies", {})
            policy = policies.pop(player, None)
            if policy is None:
                policy = [float(c == action) for c in range(7)]
            ply = sum(v != 0 for col in state for v in col)
            session.trajectories.setdefault(player, []).append((state, policy, player, ply))
        if terminal:
            records = session.trajectories.pop(player, [])
            if records:
                model_id = session.bindings[player][0]
                final_ply = sum(v != 0 for col in board for v in col) - 1
                try:
                    self.workers[model_id].submit(records, outcome(board), final_ply,
                                                 getattr(session, "game_id", None))
                except queue.Full:
                    # Report a bounded queue overflow without invalidating a confirmed move.
                    self.errors["training"] += 1
                    self.last_error = "Human experience queue full; episode was not learned"

    def move(self, request):
        with self.lock:
            previous = self.games.get(request.game_id)
            if previous and request.model_dump_json() in previous.responses:
                return previous.responses[request.model_dump_json()]
            result = super().move(request)
            session = self.games[request.game_id]
            session.game_id = request.game_id
            if request.train and request.player in session.pending:
                if not hasattr(session, "search_policies"):
                    session.search_policies = {}
                # A retry can arrive after the model published; keep its original policy.
                session.search_policies.setdefault(request.player,
                    list(self.agent(request.model_id).last_policy))
            result["architecture"] = ARCHITECTURE
            result["replay_size"] = self.workers[request.model_id].status.get("replay_size", 0)
            return result

    def progress(self, name, force=False):
        with self.lock:
            agent = self.agent(name)
            worker = self.workers[name]
            if force:
                worker.measure.set()
            return {"model_id": name, "architecture": ARCHITECTURE, "input_shape": [7, 5],
                    "updates": agent.updates, "observations": agent.observations,
                    "exploration_rate": agent.exploration_rate,
                    "exploration_decisions": agent.decisions, "pretraining": None,
                    "diagnostics": agent.diagnostics(), "service_errors": dict(self.errors),
                    "last_service_error": self.last_error, "evaluation_interval": None,
                    "history": list(agent.evaluations), "background": worker.snapshot(),
                    "evaluation_pending": worker.measure.is_set(), "benchmark": BENCHMARK}

    def control(self, name, action):
        with self.lock:
            if name not in self.workers:
                raise HTTPException(404, "Model is not loaded")
            worker = self.workers[name]
            if not worker.thread.is_alive():
                raise HTTPException(409, "Background learner is not running; check its error and restart the API")
            if action == "pause":
                worker.paused.set()
            elif action == "resume":
                worker.paused.clear()
            elif action == "evaluate":
                worker.measure.set()
            elif action == "checkpoint":
                worker.checkpoint.set()
            worker.event("control", f"Operator requested {action}")
            return {"accepted": True, "action": action,
                    "note": "Applied between batches; evaluation waits while learning is paused."}

    def record_discard(self, game_id, session):
        if session.finished or not session.bindings:
            return
        for name, train in set(session.bindings.values()):
            if train:
                self.workers[name].event("session", "Unfinished external game discarded without reward", game_id=game_id)

    def debug_state(self, name):
        with self.lock:
            if name not in self.agents:
                raise HTTPException(404, "Model is not loaded; select an existing model")
            agent, worker = self.agents[name], self.workers[name]
            sessions = [{"game_id": game_id, "created_at": s.created_at,
                         "age_seconds": time.time() - s.created_at, "finished": s.finished,
                         "moves": sum(v != 0 for col in s.expected for v in col),
                         "bindings": {str(p): {"model": b[0], "train": b[1]} for p, b in s.bindings.items()},
                         "pending_players": list(s.pending),
                         "recorded_positions": sum(len(t) for t in s.trajectories.values()),
                         "board": deepcopy(s.expected)}
                        for game_id, s in self.games.items() if any(b[0] == name for b in s.bindings.values())]
            return {"timestamp": time.time(), "model_id": name, "models": list(self.agents),
                    "architecture": ARCHITECTURE, "device": str(agent.device),
                    "parameters": sum(p.numel() for p in agent.online.parameters()),
                    "published": {"generation": agent.generation, "weights_update": agent.published_updates,
                        "published_at": agent.last_published_at, "checkpoint_path": str(agent.path),
                        "checkpoint_saved_at": agent.last_saved_at, "checkpoint_error": agent.checkpoint_error,
                        "search_simulations": agent.simulations},
                    "background": worker.debug_snapshot(), "evaluations": list(agent.evaluations),
                    "sessions": sessions, "requests": list(self.request_metrics),
                    "errors": dict(self.errors), "last_error": self.last_error}

    def analyze(self, name, board, player, simulations):
        # Inspect a copied published network without holding the gameplay lock during search.
        with self.lock:
            if name not in self.agents:
                raise HTTPException(404, "Model is not loaded")
            agent = self.agents[name]
            network, generation = deepcopy(agent.online).eval(), agent.generation
            weights_update = agent.published_updates
        started = time.perf_counter()
        analysis = search_many(network, [Position.from_board(board, player)], agent.device,
                               simulations, details=True)[0]
        return {**analysis, "generation": generation, "weights_update": weights_update,
                "column": max(range(7), key=lambda c: analysis["policy"][c]),
                "milliseconds": (time.perf_counter() - started) * 1000,
                "board": board, "player": player, "trains": False}
