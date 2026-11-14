"""Local batched self-play, independent of the physical-game request loop."""
from copy import deepcopy
from collections import deque
import logging
import queue
import random
from threading import Event, Thread
import time

import torch

from .az import Agent, search_many
from .bitboard import Position, reference_move
from .game import empty_board, drop, outcome, encode
from .rules import choose_rule_move

log = logging.getLogger(__name__)
BENCHMARK = "search-v1"


def generate(agent, games=16, simulations=32, history=(), stop=None, on_position=None, before_move=None):
    """Play a mixture of complete games; no rule labels enter training."""
    rng = agent.rng
    pool = history or [agent.online]
    states = []
    for i in range(games):
        draw = rng.random()
        kind = "self" if draw < 0.60 else "rules" if draw < 0.85 else "random"
        states.append({"board": empty_board(), "player": 1, "learner": 1 if i % 2 == 0 else -1,
                       "kind": kind, "opponent": rng.choice(pool), "records": [], "ply": 0})
    completed = []
    while states and not (stop and stop.is_set()):
        groups = {}
        for state in states:
            ours = state["player"] == state["learner"]
            if ours or state["kind"] == "self":
                model = agent.online if ours else state["opponent"]
                groups.setdefault(id(model), (model, []))[1].append(state)
            else:
                board, player = state["board"], state["player"]
                state["action"] = (choose_rule_move(board, player, rng)[0]
                                   if state["kind"] == "rules" else rng.choice(Position.from_board(board, player).legal()))
        for model, group in groups.values():
            policies = search_many(model, [Position.from_board(s["board"], s["player"]) for s in group],
                                   agent.device, simulations, noise=True, rng=rng)
            for state, policy in zip(group, policies):
                # Explore openings; use strongest visit count in later positions.
                state["action"] = (rng.choices(range(7), weights=policy)[0] if state["ply"] < 10
                                   else max(range(7), key=lambda c: policy[c]))
                if state["player"] == state["learner"]:
                    state["records"].append((encode(state["board"], state["player"]), policy,
                                             state["player"], state["ply"]))
                    agent.decisions += 1
        pending = []
        for state in states:
            if before_move and not before_move():
                return completed
            state["board"] = drop(state["board"], state["action"], state["player"])
            state["ply"] += 1
            winner = outcome(state["board"])
            if on_position:
                on_position(state)
            if winner is None:
                state["player"] *= -1
                pending.append(state)
            else:
                # Include the actual final ply even when the opponent ended the game.
                final_ply = state["ply"] - 1
                agent.add_episode(state["records"], winner, final_ply=final_ply)
                completed.append((state["kind"], winner, state["learner"]))
        states = pending
    return completed


def match(model, opponent, device, games=20, simulations=32, seed=91027, stop=None):
    """Paired seeded openings; no exploration noise, learning, or replay changes."""
    wins = draws = losses = 0
    positions = {"first": {"wins": 0, "draws": 0, "losses": 0},
                 "second": {"wins": 0, "draws": 0, "losses": 0}}
    for game in range(games):
        if stop and stop.is_set():
            return None
        rng = random.Random(seed + game // 2)
        board, player, bot, ply = empty_board(), 1, 1 if game % 2 == 0 else -1, 0
        while outcome(board) is None:
            if stop and stop.is_set():
                return None
            if ply < 2:
                action = rng.choice(Position.from_board(board, player).legal())
            elif player != bot and isinstance(opponent, str):
                if opponent == "search":
                    action = reference_move(Position.from_board(board, player))
                elif opponent == "rules":
                    action = choose_rule_move(board, player, rng)[0]
                elif opponent == "random":
                    action = rng.choice(Position.from_board(board, player).legal())
                else:
                    raise ValueError("Unknown benchmark opponent")
            else:
                network = model if player == bot else opponent
                policy = search_many(network, [Position.from_board(board, player)], device, simulations)[0]
                action = max(range(7), key=lambda c: policy[c])
            board = drop(board, action, player)
            player *= -1
            ply += 1
        winner = outcome(board)
        result = "draws" if winner == 0 else "wins" if winner == bot else "losses"
        positions["first" if bot == 1 else "second"][result] += 1
        wins += result == "wins"
        draws += result == "draws"
        losses += result == "losses"
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": (wins + draws * 0.5) / games, "win_rate": wins / games,
            "positions": positions}


class Worker:
    def __init__(self, serving, lock, batch_games=16, simulations=32, interval=300,
                 gate_games=40, move_delay=0.5):
        if min(batch_games, simulations, gate_games) < 1 or interval <= 0 or not 0 <= move_delay <= 60:
            raise ValueError("Background training settings must be positive")
        self.serving, self.lock = serving, lock
        self.batch_games, self.simulations = batch_games, simulations
        self.interval, self.gate_games = interval, gate_games
        self.move_delay = move_delay
        self.stop = Event()
        self.paused = Event()
        self.measure = Event()
        self.checkpoint = Event()
        self.incoming = queue.Queue(maxsize=1000)
        self.status = {"running": False, "games": 0, "batches": 0, "error": None,
                       "last_evaluation": None, "latest_game": None,
                       "phase": "starting", "phase_since": time.time(), "started_at": time.time(),
                       "heartbeat": time.time(), "evaluation_due_at": time.time() + interval,
                       "external_received": 0, "external_consumed": 0, "external_positions": 0,
                       "opponents": {kind: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
                                     for kind in ("self", "rules", "random")},
                       "learner_checkpoint": {"path": str(serving.path.with_suffix(".learner.pt")),
                                              "saved_at": None, "error": None}}
        self.events = deque(maxlen=200)
        self.series = deque(maxlen=600)
        self.thread = Thread(target=self.run, name="connect4-learner", daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {**deepcopy(self.status), "queue_size": self.incoming.qsize(),
                    "pause_requested": self.paused.is_set(), "evaluation_requested": self.measure.is_set(),
                    "checkpoint_requested": self.checkpoint.is_set(), "move_delay_ms": round(self.move_delay * 1000)}

    def debug_snapshot(self):
        with self.lock:
            return {**self.snapshot(), "events": list(self.events), "series": list(self.series),
                    "config": {"batch_games": self.batch_games, "training_simulations": self.simulations,
                               "serving_simulations": self.serving.simulations,
                               "evaluation_seconds": self.interval, "gate_games": self.gate_games,
                               "promotion_score": 0.60, "mixture": {"self": 0.60, "rules": 0.25, "random": 0.15}}}

    def event(self, kind, message, **data):
        with self.lock:
            self.events.append({"timestamp": time.time(), "kind": kind, "message": message, **data})

    def phase(self, name):
        with self.lock:
            if self.status["phase"] != name:
                self.status.update(phase=name, phase_since=time.time())
            self.status["heartbeat"] = time.time()

    def submit(self, records, winner, final_ply, game_id=None):
        self.incoming.put_nowait((records, winner, final_ply))
        with self.lock:
            self.status["external_received"] += 1
        self.event("experience", "Confirmed external episode queued", positions=len(records), winner=winner,
                   game_id=game_id)

    def wait_before_move(self):
        elapsed = 0.0
        while not self.stop.is_set():
            if self.paused.is_set():
                self.phase("paused")
                if self.checkpoint.is_set() and hasattr(self, "candidate"):
                    self.save_candidate(self.candidate)
                self.stop.wait(0.05)
                continue
            self.phase("collecting experience")
            if elapsed >= self.move_delay:
                return True
            started = time.monotonic()
            self.stop.wait(min(0.05, self.move_delay - elapsed))
            elapsed += time.monotonic() - started
        return False

    def run(self):
        try:
            self._run()
        except Exception as exc:
            log.exception("Background learner stopped")
            with self.lock:
                self.status["error"] = f"{type(exc).__name__}: {exc}"
            self.event("error", self.status["error"])
        finally:
            with self.lock:
                self.status["running"] = False
            self.phase("error" if self.status["error"] else "stopped")

    def _run(self):
        serving = self.serving
        candidate = Agent(serving.path.with_suffix(".learner.pt"), device=serving.device,
                          simulations=self.simulations)
        if not candidate.path.exists():
            candidate.online.load_state_dict(serving.online.state_dict())
        self.candidate = candidate
        serving.save_generation()
        history = deque([deepcopy(serving.online).eval()], maxlen=4)
        last_check = last_save = time.monotonic()
        with self.lock:
            self.status["running"] = True
            self.status["learner_checkpoint"]["saved_at"] = candidate.last_saved_at
            self.status["replay_size"] = len(candidate.memory)
            self.status["human_replay_size"] = len(candidate.human_memory)
            checks = [p for p in serving.evaluations if p.get("stage") == "candidate"]
            self.status["last_evaluation"] = checks[-1] if checks else None
        if not serving.evaluations:
            self.phase("baseline evaluation")
            baseline = match(serving.online, "rules", serving.device, 20,
                             self.simulations, stop=self.stop)
            if baseline is not None:
                with self.lock:
                    serving.evaluations.append({**baseline, "benchmark": BENCHMARK,
                        "timestamp": time.time(), "updates": serving.updates,
                        "observations": serving.observations, "promoted": True,
                        "stage": "baseline", "winning_discs": None,
                        "search_simulations": self.simulations})
                self.event("evaluation", "Initial published model measured", score=baseline["score"])
        last_frame = [0.0]
        def show_position(state):
            if time.monotonic() - last_frame[0] < 0.2:
                return
            last_frame[0] = time.monotonic()
            with self.lock:
                self.status["latest_game"] = {"board": deepcopy(state["board"]),
                    "opponent": state["kind"], "next_player": -state["player"],
                    "move_count": state["ply"], "winner": outcome(state["board"]),
                    "updated_at": time.time()}
        try:
            while not self.stop.is_set():
                if self.paused.is_set():
                    self.phase("paused")
                    if self.checkpoint.is_set():
                        self.save_candidate(candidate)
                    self.stop.wait(0.2)
                    continue
                batch_start, observations_before = time.monotonic(), candidate.observations
                self.phase("collecting experience")
                while True:
                    try:
                        records, winner, final_ply = self.incoming.get_nowait()
                    except queue.Empty:
                        break
                    candidate.add_episode(records, winner, human=True, final_ply=final_ply)
                    with self.lock:
                        self.status["external_consumed"] += 1
                        self.status["external_positions"] += len(records)
                    self.event("experience", "External episode added to replay", positions=len(records))
                results = generate(candidate, self.batch_games, self.simulations, history, self.stop,
                                   on_position=show_position, before_move=self.wait_before_move)
                if self.stop.is_set():
                    break
                # A bounded number of updates per fresh batch limits replay overfitting.
                self.phase("optimizing")
                for _ in range(max(1, len(results))):
                    candidate.train_step()
                with self.lock:
                    self.status["games"] += len(results)
                    self.status["batches"] += 1
                    serving.updates, serving.observations = candidate.updates, candidate.observations
                    serving.decisions, serving.last_loss = candidate.decisions, candidate.last_loss
                    self.status["replay_size"] = len(candidate.memory)
                    self.status["human_replay_size"] = len(candidate.human_memory)
                    seconds = max(0.001, time.monotonic() - batch_start)
                    self.status["games_per_second"] = len(results) / seconds
                    self.status["observations_per_second"] = (candidate.observations - observations_before) / seconds
                    self.status["candidate"] = {"updates": candidate.updates,
                        "observations": candidate.observations, "loss": candidate.last_loss,
                        "policy_loss": candidate.last_policy_loss, "value_loss": candidate.last_value_loss,
                        "entropy": candidate.last_entropy, "gradient_norm": candidate.last_gradient_norm}
                    for kind, winner, learner in results:
                        counter = self.status["opponents"][kind]
                        counter["games"] += 1
                        counter["draws" if winner == 0 else "wins" if winner == learner else "losses"] += 1
                    if not self.series or time.time() - self.series[-1]["timestamp"] >= 1:
                        self.series.append({"timestamp": time.time(), "games": self.status["games"],
                            "updates": candidate.updates, "observations": candidate.observations,
                            "loss": candidate.last_loss, "policy_loss": candidate.last_policy_loss,
                            "value_loss": candidate.last_value_loss,
                            "games_per_second": self.status["games_per_second"]})
                now = time.monotonic()
                if candidate.updates > 0 and (now - last_check >= self.interval or self.measure.is_set()):
                    self.measure.clear()
                    self.phase("promotion evaluation")
                    self.event("evaluation", "Candidate evaluation started", updates=candidate.updates)
                    incumbent = deepcopy(serving.online).eval()
                    arena = match(candidate.online, incumbent, candidate.device, self.gate_games,
                                  self.simulations, seed=91027 + candidate.updates, stop=self.stop)
                    rules = match(candidate.online, "rules", candidate.device, 20,
                                  self.simulations, stop=self.stop)
                    random_result = match(candidate.online, "random", candidate.device, 20,
                                          self.simulations, stop=self.stop)
                    if arena is None or rules is None or random_result is None:
                        break
                    promoted = arena["score"] >= 0.60
                    point = {**rules, "benchmark": BENCHMARK, "timestamp": time.time(),
                             "updates": candidate.updates, "observations": candidate.observations,
                             "arena": arena, "random": random_result, "promoted": promoted,
                             "winning_discs": None, "search_simulations": self.simulations,
                             "stage": "candidate"}
                    with self.lock:
                        if promoted:
                            serving.online.load_state_dict(candidate.online.state_dict())
                            serving.generation += 1
                            serving.published_updates = candidate.updates
                            serving.last_published_at = time.time()
                        serving.evaluations.append(point)
                        self.status["last_evaluation"] = point
                    self.event("publication" if promoted else "evaluation",
                        "Candidate published" if promoted else "Candidate not promoted",
                        arena_score=arena["score"], rules_score=rules["score"], generation=serving.generation)
                    if promoted:
                        history.append(deepcopy(candidate.online).eval())
                    candidate.generation = serving.generation
                    candidate.evaluations = deepcopy(serving.evaluations)
                    try:
                        with self.lock:
                            serving.save()
                            if promoted:
                                serving.save_generation()
                    except OSError as exc:
                        self.event("error", f"Published checkpoint failed: {exc}")
                    last_check = time.monotonic()
                    with self.lock:
                        self.status["evaluation_due_at"] = time.time() + self.interval
                if time.monotonic() - last_save >= 60 or self.checkpoint.is_set():
                    self.save_candidate(candidate)
                    last_save = time.monotonic()
        finally:
            # Drain confirmed human games even if shutdown arrived during search.
            while not self.incoming.empty():
                records, winner, final_ply = self.incoming.get_nowait()
                candidate.add_episode(records, winner, human=True, final_ply=final_ply)
                with self.lock:
                    self.status["external_consumed"] += 1
                    self.status["external_positions"] += len(records)
            self.save_candidate(candidate)

    def save_candidate(self, candidate):
        self.checkpoint.clear()
        self.phase("saving learner")
        try:
            candidate.save()
            with self.lock:
                self.status["learner_checkpoint"].update(saved_at=time.time(), error=None)
            self.event("checkpoint", "Learner checkpoint saved", updates=candidate.updates)
        except OSError as exc:
            with self.lock:
                self.status["learner_checkpoint"]["error"] = str(exc)
            self.event("error", f"Learner checkpoint failed: {exc}")
            # Periodic failures should remain visible and retry, not kill learning.

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=40)
            if self.thread.is_alive():
                raise RuntimeError("Learner did not stop within 40 seconds")
