"""Self-play and human training games with a shared web board; no PyTorch dependency."""
from contextlib import asynccontextmanager
from copy import deepcopy
import logging
import math
import os
import random
from itertools import combinations
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from .game import drop, empty_board, outcome
from .event_game import EventGame, install_event_routes
from .rules import RULE_MODEL
from .simulator import play_game, post

log = logging.getLogger(__name__)


class Simulation:
    def __init__(self, api_url, model_one="shared", model_two=RULE_MODEL, interval=0,
                 result_delay=0.0001, swap_sides=True):
        if any(not math.isfinite(v) or v < 0 for v in (interval, result_delay)):
            raise ValueError("SIM_INTERVAL and SIM_RESULT_DELAY must be finite and nonnegative")
        self.api_url = api_url
        self.models = (model_one, model_two)
        self.round_robin = False
        self.curriculum = True
        self.pair_queue = []
        self.active_pair = self.models
        self.active_slots = (0, 1)
        self.pair_block = -1
        if model_one == model_two == RULE_MODEL:
            raise ValueError("At least one player must be a learning AI")
        self.ai_model = next(model for model in reversed(self.models) if model != RULE_MODEL)
        self.swap_sides = swap_sides
        self.simulation_games = 0
        self.interval, self.result_delay = interval, result_delay
        self.stop = Event()
        self.human_mode = Event()
        self.game_cancel = Event()
        self.operation_lock = Lock()
        self.control_lock = Lock()
        self.lock = Lock()
        self.state = {"mode": "simulation", "human_player": 1,
                      "status": "starting", "board": empty_board(), "game_id": None,
                      "model_one": model_one, "model_two": model_two, "interval": interval,
                      "config": {"model_one": model_one, "model_two": model_two,
                                 "swap_sides": swap_sides},
                      "rotation": None,
                      "rule_probability": 0.5, "api_errors": 0,
                      "matchups": {},
                      "games": 0, "scores": {"1": 0, "-1": 0, "0": 0},
                      "move_count": 0, "next_player": 1, "done": False,
                      "winner": None, "last_column": None, "last_player": None,
                      "device": None, "epsilon": None, "loss": None,
                      "updates": 0, "replay_size": 0, "error": None,
                      "updated_at": time.time()}

    def update(self, changes):
        with self.lock:
            self.state.update(changes)
            self.state["updated_at"] = time.time()

    def snapshot(self):
        with self.lock:
            return deepcopy(self.state)

    def configure(self, config):
        with self.control_lock:
            # The same handoff as a mode change: drain the current move and cancel
            # the old game before changing player bindings or the switching phase.
            self.human_mode.set()
            self.game_cancel.set()
            with self.operation_lock:
                state = self.snapshot()
                if state["mode"] == "human":
                    with httpx.Client(base_url=self.api_url, timeout=10) as client:
                        self.discard(client, state["game_id"])
                self.models = (config.model_one, config.model_two)
                self.round_robin = config.round_robin
                self.curriculum = config.curriculum
                if self.round_robin:
                    self.models += (config.model_three,)
                self.pair_queue = []
                self.pair_block = -1
                self.active_pair = self.models[:2]
                self.active_slots = (0, 1)
                self.ai_model = next(model for model in reversed(self.models) if model != RULE_MODEL)
                self.swap_sides = config.swap_sides
                self.simulation_games = 0
                self.update({"config": config.model_dump(), "rotation": None, "matchups": {}})
                self.select_pair()
                self.new_board("simulation")
                self.human_mode.clear()
            return self.snapshot()

    def select_pair(self):
        """Shuffle all three pairings per round, advancing only on completed games."""
        if not self.round_robin:
            return self.models
        block = self.simulation_games // 15
        if block != self.pair_block:
            if not self.pair_queue:
                self.pair_queue = list(combinations(range(3), 2))
                random.shuffle(self.pair_queue)
                if self.pair_block >= 0 and self.pair_queue[-1] == self.active_slots:
                    self.pair_queue[0], self.pair_queue[-1] = self.pair_queue[-1], self.pair_queue[0]
            self.active_slots = self.pair_queue.pop()
            self.active_pair = tuple(self.models[index] for index in self.active_slots)
            self.pair_block = block
        self.update({"rotation": {"pair": list(self.active_pair),
                     "slots": [index + 1 for index in self.active_slots],
                     "round": block // 3 + 1, "game": self.simulation_games % 15 + 1,
                     "games_per_pair": 15}})
        return self.active_pair

    def record_matchup(self, models, winner):
        with self.lock:
            counts = self.state["matchups"].setdefault(models[0], {}).setdefault(
                models[1], {"1": 0, "0": 0, "-1": 0})
            counts[str(winner)] += 1

    def training_progress(self, action=None, exploration_rate=1.0):
        # The model API serializes these controls with training; no game is interrupted.
        try:
            with httpx.Client(base_url=self.api_url, timeout=60) as client:
                models = {}
                for name in dict.fromkeys(model for model in self.models if model != RULE_MODEL):
                    if action == "reset":
                        post(client, "/exploration/reset", {"model_id": name, "rate": exploration_rate})
                    if action == "evaluate":
                        models[name] = post(client, "/evaluate", {"model_id": name})
                    else:
                        response = client.get(f"/models/{name}/progress")
                        response.raise_for_status()
                        models[name] = response.json()
                return {"models": models}
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Could not reach the model API for training progress") from exc

    def discard(self, client, game_id):
        if game_id:
            try:
                response = client.delete(f"/games/{game_id}")
                if response.status_code != 404:
                    response.raise_for_status()
            except httpx.HTTPError:
                log.warning("Could not delete session %s", game_id)

    def new_board(self, mode):
        self.update({"mode": mode, "status": "human_turn" if mode == "human" else "starting",
                     "game_id": f"human-{uuid4().hex}" if mode == "human" else None,
                     "board": empty_board(), "move_count": 0, "next_player": 1,
                     "model_one": "human" if mode == "human" else self.active_pair[0],
                     "model_two": self.ai_model if mode == "human" else self.active_pair[1],
                     "actor": None, "move_reason": None,
                     "done": False, "winner": None, "last_column": None,
                     "last_player": None, "error": None})

    def set_mode(self, mode, restart=False):
        with self.control_lock:
            if self.snapshot()["mode"] == mode and not restart:
                return self.snapshot()
            # Interrupt waits and let an in-flight API call and session cleanup finish
            # before replacing the board. The worker cannot start another game.
            self.human_mode.set()
            self.game_cancel.set()
            with self.operation_lock:
                state = self.snapshot()
                if state["mode"] == "human":
                    with httpx.Client(base_url=self.api_url, timeout=10) as client:
                        self.discard(client, state["game_id"])
                self.new_board(mode)
                if mode == "simulation":
                    self.human_mode.clear()
            return self.snapshot()

    def human_move(self, game_id, move_count, column=None):
        with self.control_lock, self.operation_lock:
            state = self.snapshot()
            if state["mode"] != "human" or state["game_id"] != game_id:
                raise HTTPException(409, "This human game is no longer active")
            if state["move_count"] != move_count:
                raise HTTPException(409, "The board changed; refresh before playing")
            if column is not None:
                if state["done"] or state["next_player"] != 1:
                    raise HTTPException(409, "It is not your turn")
                try:
                    board = drop(state["board"], column, 1)
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
                self.show_human_board(board, column, 1)
            elif state["status"] != "human_error":
                raise HTTPException(409, "There is no failed bot action to retry")
            try:
                with httpx.Client(base_url=self.api_url, timeout=10) as client:
                    state = self.snapshot()
                    if not state["done"]:
                        result = post(client, "/move", {
                            "game_id": game_id, "board": state["board"], "player": -1,
                            "model_id": self.ai_model, "train": True, "explore": False,
                        })
                        self.update({key: result[key] for key in
                                     ("model_id", "device", "epsilon", "loss", "updates", "replay_size")})
                        self.update({key: result[key] for key in ("diagnostics", "service_errors") if key in result})
                        self.show_human_board(drop(state["board"], result["column"], -1),
                                              result["column"], -1)
                    state = self.snapshot()
                    if state["done"]:
                        post(client, "/finish", {"game_id": game_id, "board": state["board"]})
                        scores = state["scores"]
                        scores[str(state["winner"])] += 1
                        self.update({"status": "finished", "games": state["games"] + 1,
                                     "scores": scores, "error": None})
                        self.discard(client, game_id)
                    else:
                        self.update({"status": "human_turn", "error": None})
            except httpx.HTTPError as exc:
                self.update({"status": "human_error", "error": str(exc)})
                raise HTTPException(502, "Model API unavailable. Retry the bot action or start a new game.") from exc
            return self.snapshot()

    def show_human_board(self, board, column, player):
        winner = outcome(board)
        self.update({"board": board, "last_column": column, "last_player": player,
                     "actor": "human" if player == 1 else self.ai_model,
                     "move_reason": "Human move" if player == 1 else "AI best legal move",
                     "move_count": sum(v != 0 for col in board for v in col),
                     "next_player": None if winner is not None else -player,
                     "done": winner is not None, "winner": winner,
                     "status": "bot_turn" if player == 1 else "human_turn", "error": None})

    def run(self):
        logging.getLogger("httpx").setLevel(logging.WARNING)
        with httpx.Client(base_url=self.api_url, timeout=10) as client:
            try:
                while not self.stop.is_set():
                    if self.human_mode.is_set():
                        self.stop.wait(0.1)
                        continue
                    delay = self.result_delay
                    with self.operation_lock:
                        if self.human_mode.is_set() or self.stop.is_set():
                            continue
                        self.game_cancel.clear()
                        try:
                            self.update({"error": None})
                            models = self.select_pair()
                            if self.swap_sides and self.simulation_games % 2:
                                models = models[::-1]
                            probability = 1.0
                            if self.curriculum and RULE_MODEL in models:
                                learner = next(model for model in models if model != RULE_MODEL)
                                response = client.get(f"/models/{learner}/curriculum")
                                response.raise_for_status()
                                probability = response.json()["rule_probability"]
                            self.update({"rule_probability": probability})
                            winner = play_game(client, *models, interval=self.interval,
                                               on_update=self.update, stop_event=self.game_cancel,
                                               rule_probability=probability)
                            state = self.snapshot()
                            scores = state["scores"]
                            scores[str(winner)] += 1
                            self.simulation_games += 1
                            self.record_matchup(models, winner)
                            self.update({"status": "finished", "games": state["games"] + 1,
                                         "scores": scores})
                            log.info("Completed game %d: winner=%s", state["games"] + 1, winner)
                        except InterruptedError:
                            continue
                        except httpx.HTTPError as exc:
                            log.warning("API unavailable or rejected game: %s", exc)
                            self.update({"status": "reconnecting", "error": str(exc),
                                         "api_errors": self.snapshot()["api_errors"] + 1})
                            delay = 2
                    self.game_cancel.wait(delay)
            finally:
                try:
                    post(client, "/checkpoint", {})
                except httpx.HTTPError:
                    log.warning("Could not request final checkpoint")
                if self.snapshot()["mode"] == "human":
                    self.discard(client, self.snapshot()["game_id"])
                self.update({"status": "stopped"})


class ModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["simulation", "human"]


class ExplorationResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate: float = Field(default=1.0, strict=True, ge=0.05, le=1.0, allow_inf_nan=False)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_one: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    model_two: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    swap_sides: StrictBool
    round_robin: StrictBool = False
    model_three: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    curriculum: StrictBool = True

    @model_validator(mode="after")
    def require_ai(self):
        if self.round_robin:
            if self.model_three is None:
                raise ValueError("Round robin requires three player entries")
            if (self.model_one, self.model_two, self.model_three).count(RULE_MODEL) > 1:
                raise ValueError("Round robin allows at most one rule player")
        if self.model_one == self.model_two == RULE_MODEL:
            raise ValueError("Choose at least one learning AI")
        return self


class HumanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str
    move_count: StrictInt = Field(ge=0, le=35)


class HumanMoveRequest(HumanRequest):
    column: StrictInt = Field(ge=0, le=6)


def create_app(api_url=None, interval=None, result_delay=None, run_worker=False):
    simulation = Simulation(api_url or os.getenv("API_URL", "http://127.0.0.1:8000"),
                            os.getenv("MODEL_ONE", "shared"), os.getenv("MODEL_TWO", RULE_MODEL),
                            float(os.getenv("SIM_INTERVAL", "0")) if interval is None else interval,
                            float(os.getenv("SIM_RESULT_DELAY", "0.0001")) if result_delay is None else result_delay,
                            swap_sides=os.getenv("SIM_SWAP_SIDES", "true").lower() in ("true", "1", "yes", "on"))

    event_game = EventGame(simulation.api_url)

    @asynccontextmanager
    async def lifespan(app):
        event_game.start()
        thread = None
        if run_worker:
            thread = Thread(target=simulation.run, name="self-play", daemon=True)
            thread.start()
        yield
        event_game.stop.set()
        simulation.stop.set()
        simulation.game_cancel.set()
        if thread:
            # Joining in a worker thread keeps the event loop responsive during shutdown.
            import asyncio
            await asyncio.to_thread(thread.join, 25)

    app = FastAPI(title="Connect Four simulator", lifespan=lifespan)
    app.state.simulation = simulation
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")
    install_event_routes(app, event_game, static)

    @app.get("/")
    def index():
        return FileResponse(static / "play.html")

    @app.get("/state")
    def state():
        return simulation.snapshot()

    @app.post("/mode")
    def mode(request: ModeRequest):
        return simulation.set_mode(request.mode)

    @app.post("/simulation/config")
    def configure(request: SimulationConfig):
        return simulation.configure(request)

    @app.post("/human/new")
    def new_human_game():
        return simulation.set_mode("human", restart=True)

    @app.post("/human/move")
    def human_move(request: HumanMoveRequest):
        return simulation.human_move(request.game_id, request.move_count, request.column)

    @app.post("/human/retry")
    def retry(request: HumanRequest):
        return simulation.human_move(request.game_id, request.move_count)

    @app.get("/health")
    def health():
        return {"status": "ok", "simulation": simulation.snapshot()["status"]}

    @app.get("/progress")
    def progress():
        return simulation.training_progress()

    @app.post("/exploration/reset")
    def reset_exploration(request: ExplorationResetRequest | None = None):
        return simulation.training_progress("reset", request.rate if request else 1.0)

    @app.post("/evaluation")
    def evaluate():
        return simulation.training_progress("evaluate")

    @app.get("/debug")
    def debug_page():
        return FileResponse(Path(__file__).parent / "static" / "debug.html")

    @app.get("/explain")
    def explanation_page():
        return FileResponse(Path(__file__).parent / "static" / "explain.html")

    def debug_forward(method, path, payload=None):
        try:
            with httpx.Client(base_url=simulation.api_url, timeout=30) as client:
                response = client.request(method, path, json=payload) if payload is not None else client.request(method, path)
                if not response.is_success:
                    raise HTTPException(response.status_code, response.json().get("detail", "API request failed"))
                return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Model API unavailable: {exc}") from exc

    @app.get("/debug/data")
    def debug_data(model: str = Query(default="shared", pattern=r"^[A-Za-z0-9_-]{1,64}$")):
        return debug_forward("GET", f"/debug/state/{model}")

    @app.post("/debug/control")
    def debug_control(payload: dict):
        return debug_forward("POST", "/debug/control", payload)

    @app.post("/debug/analyze")
    def debug_analyze(payload: dict):
        return debug_forward("POST", "/debug/analyze", payload)

    return app


app = create_app()
