"""Shared physical STEM game. Camera observations, never predictions, confirm discs."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Event, RLock, Thread
import time
from uuid import uuid4

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictBool
from typing import Literal
from .game import empty_board, drop, next_player, outcome, validate_board


def stamp():
    return datetime.now(timezone.utc).isoformat()


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(default="shared", pattern=r"^(shared|saved-[a-f0-9]{24})$")
    robot_color: Literal["B", "R"] = "B"
    first: Literal["human", "robot"] = "human"
    robot_character: Literal["orbit", "spark", "pixel"] = "orbit"
    human_character: Literal["astronaut", "scientist", "explorer"] = "astronaut"


class ManualMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: StrictInt = Field(ge=0, le=6)
    player: Literal["human", "robot"]
    revision: StrictInt


class BoardSize(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scale: StrictInt = Field(ge=50, le=175)


class CameraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delay_ms: StrictInt = Field(ge=100, le=60000)


class ResultPreview(BaseModel):
    winner: Literal["human", "robot"] | None = None


class CameraFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    board: list[list[Literal["X", "B", "R"]]]
    game_id: str | int
    timestamp: datetime
    valid: StrictBool


class EventGame:
    def __init__(self, api_url, score_file=None):
        self.api_url = api_url
        self.camera_url = os.getenv("CAMERA_API_URL", "")
        self.camera_poll_delay_ms = int(os.getenv("CAMERA_POLL_DELAY_MS", "500"))
        CameraSettings(delay_ms=self.camera_poll_delay_ms)
        self.camera_schedule_changed = Event()
        self.lock = RLock()
        self.stop = Event()
        self.setup = Setup()
        self.revision = 0
        self.generation = 0
        self.busy = False
        self.attempted = None
        self.thread = None
        self.board_scale = 100
        self.score_file = Path(score_file or os.getenv("EVENT_SCORE_FILE", "runtime/robot_scores.txt"))
        self.score_error = None
        self.score_history = [{"game": 0, "score": 0, "points": 0}]
        if self.score_file.exists():
            for line in self.score_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if line.strip() not in ("+1", "1", "-1", "0"):
                    raise ValueError(f"Invalid score in {self.score_file}: {line!r}")
                points = int(line)
                self.score_history.append({"game": len(self.score_history),
                    "score": self.score_history[-1]["score"] + points, "points": points})
        self.reset()

    def reset(self, setup=None):
        with self.lock:
            if setup is not None:
                self.setup = setup
            self.generation += 1
            self.revision += 1
            self.session = "stem-" + uuid4().hex
            self.board = empty_board()
            self.camera_game = None
            self.override = False
            self.attempted = None
            self.busy = False
            self.status = "waiting"
            self.suggestion = None
            self.thinking_until = 0
            self.camera = {"ok": False, "last_success": None, "error": None, "latency_ms": None}
            self.model = {"ok": None, "last_success": None, "error": None, "latency_ms": None}
            self.finished = False
            self.scored = False
            self.preview_winner = None
            self.mock = {"board": [["X"] * 7 for _ in range(5)], "game_id": self.session,
                         "timestamp": stamp(), "valid": True}
        return self.snapshot()

    def player_number(self, actor):
        return 1 if self.setup.first == actor else -1

    def color_number(self):
        robot = self.player_number("robot")
        return {"X": 0, self.setup.robot_color: robot,
                "R" if self.setup.robot_color == "B" else "B": -robot}

    def snapshot(self):
        with self.lock:
            reverse = {v: k for k, v in self.color_number().items()}
            result = outcome(self.board)
            return deepcopy({"setup": self.setup.model_dump(), "board": [[reverse[v] for v in col] for col in self.board],
                "game_id": self.session, "revision": self.revision, "status": self.status,
                "suggestion": self.suggestion, "thinking_until": self.thinking_until,
                "next": "robot" if next_player(self.board) == self.player_number("robot") else "human",
                "winner": None if result is None else "draw" if result == 0 else
                    "robot" if result == self.player_number("robot") else "human",
                "move_count": sum(v != 0 for col in self.board for v in col),
                "board_scale": self.board_scale,
                "preview_winner": self.preview_winner,
                "score_error": self.score_error,
                "score_history": self.score_history,
                "robot_score": self.score_history[-1]["score"],
                "camera": self.camera, "model": self.model, "manual": self.override,
                "camera_source": "external" if self.camera_url else "mock",
                "camera_poll_delay_ms": self.camera_poll_delay_ms})

    def mock_frame(self):
        with self.lock:
            return deepcopy({**self.mock, "timestamp": stamp()})

    def accept(self, payload):
        frame = CameraFrame.model_validate(payload)
        if not frame.valid:
            raise ValueError("Camera marked this board invalid")
        if len(frame.board) != 5 or any(len(row) != 7 for row in frame.board):
            raise ValueError("Camera board must contain 5 rows of 7 cells")
        age = time.time() - frame.timestamp.timestamp()
        if age > 10 or age < -10:
            raise ValueError("Camera timestamp is stale or more than 10 seconds in the future")
        with self.lock:
            values = self.color_number()
            board = [[values[frame.board[r][c]] for r in range(5)] for c in range(7)]
            validate_board(board)
            if self.override:
                return
            changed_game = self.camera_game is not None and str(frame.game_id) != self.camera_game
            if changed_game:
                self.generation += 1
                self.session = "stem-" + uuid4().hex
                self.attempted = None
                self.finished = False
                self.scored = False
                self.busy = False
            self.camera_game = str(frame.game_id)
            if board != self.board or changed_game:
                self.board = board
                self.revision += 1
                self.suggestion = None
                self.status = "waiting"
                # A result from the previous observation must never be displayed.
                self.generation += 1
                self.busy = False
            self.maybe_move()

    def poll_camera(self):
        start = time.monotonic()
        try:
            if self.camera_url:
                with httpx.Client(timeout=1.8) as client:
                    response = client.get(self.camera_url)
                    response.raise_for_status()
                    payload = response.json()
            else:
                payload = self.mock_frame()
            self.accept(payload)
            with self.lock:
                self.camera.update(ok=True, last_success=stamp(), error=None,
                                   latency_ms=round((time.monotonic()-start)*1000))
        except Exception as exc:
            with self.lock:
                self.camera.update(ok=False, error=str(exc))

    def record_score(self, points):
        record = {"game": self.score_history[-1]["game"] + 1,
                  "score": self.score_history[-1]["score"] + points, "points": points}
        pending = self.score_file.with_name(self.score_file.name + ".tmp")
        try:
            self.score_file.parent.mkdir(parents=True, exist_ok=True)
            with pending.open("w", encoding="utf-8", newline="\n") as stream:
                for item in [*self.score_history[1:], record]:
                    stream.write(("+1" if item["points"] == 1 else str(item["points"])) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(self.score_file)
            self.score_history.append(record)
            self.score_error = None
            return True
        except OSError as exc:
            self.score_error = "Score kon niet worden opgeslagen: " + str(exc)
            return False
        finally:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass

    def maybe_move(self, force=False):
        # Caller holds the reentrant lock. Only board identity, not timestamp, triggers work.
        key = (self.session, tuple(tuple(col) for col in self.board))
        terminal = outcome(self.board)
        if terminal is not None:
            if not self.scored:
                points = 0 if terminal == 0 else 1 if terminal == self.player_number("robot") else -1
                self.scored = self.record_score(points)
            self.status = "finished"
            if not self.finished and not self.busy:
                self.finished = True
                self.busy = True
                Thread(target=self.finish, args=(self.generation, self.session, deepcopy(self.board)), daemon=True).start()
            return
        if next_player(self.board) != self.player_number("robot") or self.busy:
            return
        if not force and self.attempted == key:
            return
        self.attempted = key
        self.busy = True
        self.status = "thinking"
        self.suggestion = None
        count = sum(v != 0 for col in self.board for v in col)
        self.thinking_until = time.time() + 1 + 2 * count / 34
        Thread(target=self.compute, args=(self.generation, self.session, deepcopy(self.board),
               self.player_number("robot"), self.thinking_until, self.setup.model_id), daemon=True).start()

    def compute(self, generation, session, board, player, deadline, model_id=None):
        start = time.monotonic()
        try:
            with httpx.Client(base_url=self.api_url, timeout=20) as client:
                response = client.post("/move", json={"game_id": session, "board": board, "player": player,
                    "model_id": model_id or self.setup.model_id, "train": (model_id or self.setup.model_id) == "shared", "explore": False})
                response.raise_for_status()
                data = response.json()
            column = data["column"]
            if type(column) is not int or column not in range(7) or board[column][0] != 0:
                raise ValueError("Move API returned an illegal column")
            elapsed = round((time.monotonic()-start)*1000)
            self.stop.wait(max(0, deadline-time.time()))
            with self.lock:
                if generation != self.generation or self.stop.is_set():
                    return
                self.model.update(ok=True, error=None, last_success=stamp(), latency_ms=elapsed)
                self.suggestion = column
                self.status = "suggestion"
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.model.update(ok=False, error=str(exc))
                    self.status = "move_error"
        finally:
            with self.lock:
                if generation == self.generation:
                    self.busy = False

    def finish(self, generation, session, board):
        try:
            with httpx.Client(base_url=self.api_url, timeout=20) as client:
                response = client.post("/finish", json={"game_id": session, "board": board})
                response.raise_for_status()
                response = client.delete("/games/" + session)
                response.raise_for_status()
            with self.lock:
                if generation == self.generation:
                    self.model.update(ok=True, error=None, last_success=stamp())
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.finished = False
                    self.model.update(ok=False, error="Could not confirm outcome: " + str(exc))
        finally:
            with self.lock:
                if generation == self.generation:
                    self.busy = False

    def manual_move(self, request):
        with self.lock:
            if request.revision != self.revision:
                raise HTTPException(409, "Board changed. Check the latest board and try again.")
            if outcome(self.board) is not None:
                raise HTTPException(409, "Game is finished")
            player = self.player_number(request.player)
            if next_player(self.board) != player:
                raise HTTPException(409, "It is the other player's turn")
            try:
                board = drop(self.board, request.column, player)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            # Manual play explicitly takes over until Resume camera is selected.
            # If the operator overrides a predicted move, use a new inference session.
            if request.player == "robot" and request.column != self.suggestion:
                self.session = "stem-" + uuid4().hex
            self.override = True
            self.generation += 1
            self.busy = False
            self.board = board
            self.revision += 1
            self.suggestion = None
            self.status = "waiting"
            self.maybe_move()
        return self.snapshot()

    def retry(self):
        with self.lock:
            self.maybe_move(force=True)
        return self.snapshot()

    def resume(self):
        with self.lock:
            # Corrections may differ from the model's last predicted board.
            self.session = "stem-" + uuid4().hex
            self.generation += 1
            self.busy = False
            self.attempted = None
            self.override = False
            self.suggestion = None
            self.status = "waiting"
        self.poll_camera()
        return self.snapshot()

    def start(self):
        def run():
            while not self.stop.is_set():
                started = time.monotonic()
                self.poll_camera()
                self.wait_for_camera_poll(started)
        self.thread = Thread(target=run, name="stem-camera", daemon=True)
        self.thread.start()

    def set_camera_settings(self, settings):
        with self.lock:
            self.camera_poll_delay_ms = settings.delay_ms
        self.camera_schedule_changed.set()
        return self.snapshot()

    def wait_for_camera_poll(self, started):
        # Recalculate an in-progress wait when the operator changes the interval.
        # Slow requests complete before another starts; no overlapping pollers.
        while not self.stop.is_set():
            with self.lock:
                remaining = self.camera_poll_delay_ms / 1000 - (time.monotonic() - started)
            if remaining <= 0:
                return
            self.camera_schedule_changed.wait(min(remaining, 0.2))
            self.camera_schedule_changed.clear()


def install_event_routes(app, game, static):
    from fastapi.responses import FileResponse
    app.state.event_game = game

    @app.get("/participant")
    @app.get("/instructor")
    def event_page():
        return FileResponse(static / "event.html")

    @app.get("/event/state")
    def state():
        return game.snapshot()

    @app.get("/event/camera/mock")
    def camera():
        return game.mock_frame()

    @app.post("/event/camera/settings")
    def camera_settings(settings: CameraSettings):
        return game.set_camera_settings(settings)

    @app.post("/event/camera/mock")
    def set_camera(frame: CameraFrame):
        with game.lock:
            game.mock = frame.model_dump(mode="json")
        return game.mock_frame()

    @app.post("/event/size")
    def board_size(request: BoardSize):
        with game.lock:
            game.board_scale = request.scale
        return game.snapshot()

    @app.post("/event/preview")
    def preview(request: ResultPreview):
        with game.lock:
            game.preview_winner = request.winner
        return game.snapshot()

    @app.post("/event/new")
    def new_game(setup: Setup):
        return game.reset(setup)

    @app.post("/event/drop")
    def manual(request: ManualMove):
        return game.manual_move(request)

    @app.post("/event/retry")
    def retry():
        return game.retry()

    @app.post("/event/resume")
    def resume():
        return game.resume()
