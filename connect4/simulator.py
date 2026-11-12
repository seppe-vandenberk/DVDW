"""Training client: learned AI versus rules, or optional two-model self-play."""
import argparse
import logging
import time
import random
from uuid import uuid4

import httpx

from .game import drop, empty_board, outcome, legal_moves
from .rules import RULE_MODEL, choose_rule_move

log = logging.getLogger(__name__)


def post(client, path, payload):
    # Safe to retry: the server caches identical move requests within a game.
    for attempt in range(3):
        try:
            response = client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(0.5)


def play_game(client, model_one="shared", model_two="shared", interval=0,
              on_update=None, stop_event=None, rule_probability=1.0):
    if not 0 <= rule_probability <= 1:
        raise ValueError("Rule probability must be between 0 and 1")
    if model_one == model_two == RULE_MODEL:
        raise ValueError("At least one player must be a learning AI")
    game_id = f"sim-{uuid4().hex}"
    board, player = empty_board(), 1
    last_move = None
    # Search-backend training runs locally. HTTP games are readable exhibitions;
    # physical/human callers still explicitly submit train=true outcomes.
    health = client.get("/health")
    health.raise_for_status()
    local_training = health.json().get("backend") == "az"
    try:
        if on_update:
            on_update({"game_id": game_id, "board": board, "next_player": player,
                       "model_one": model_one, "model_two": model_two,
                       "actor": None, "move_reason": None,
                       "last_column": None, "last_player": None, "move_count": 0,
                       "done": False, "winner": None, "status": "playing"})
        while outcome(board) is None:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Simulation stopped")
            if last_move is not None:
                delay = max(0.0, interval - (time.monotonic() - last_move))
                if stop_event is None:
                    time.sleep(delay)
                elif stop_event.wait(delay):
                    raise InterruptedError("Simulation stopped")
            last_move = time.monotonic()
            actor = model_one if player == 1 else model_two
            metrics = {}
            if actor == RULE_MODEL:
                if rule_probability < 1 and random.random() >= rule_probability:
                    column, reason = random.choice(legal_moves(board)), "Training curriculum: random legal move"
                else:
                    column, reason = choose_rule_move(board, player)
            else:
                result = post(client, "/move", {
                    "game_id": game_id, "board": board, "player": player,
                    "model_id": actor, "train": not local_training, "explore": not local_training,
                })
                column = result["column"]
                reason = "AI search exhibition; background learning continues" if local_training else "AI exploration and training"
                metrics = {key: result[key] for key in
                           ("model_id", "device", "epsilon", "loss", "updates", "replay_size")}
                metrics.update({key: result[key] for key in ("diagnostics", "service_errors") if key in result})
            board = drop(board, column, player)
            if on_update:
                done = outcome(board) is not None
                on_update({"game_id": game_id, "board": board,
                           "next_player": None if done else -player,
                           "last_column": column, "last_player": player,
                           "actor": actor, "move_reason": reason,
                           "move_count": sum(v != 0 for col in board for v in col),
                           "done": done, "winner": outcome(board), "status": "playing", **metrics})
            player = -player
        post(client, "/finish", {"game_id": game_id, "board": board})
        return outcome(board)
    finally:
        # Also discard unfinished games after an error. Never invent a terminal reward.
        try:
            response = client.delete(f"/games/{game_id}")
            if response.status_code != 404:
                response.raise_for_status()
        except httpx.HTTPError:
            log.warning("Could not delete session %s", game_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-one", default="shared")
    parser.add_argument("--model-two", default=RULE_MODEL, help="rules (default) or another model ID")
    parser.add_argument("--swap-sides", action=argparse.BooleanOptionalAction, default=True,
                        help="Swap player positions after each game (default: on)")
    parser.add_argument("--interval", type=float, default=0, help="Seconds between moves (default: 0, no delay)")
    parser.add_argument("--games", type=int, default=0, help="0 runs indefinitely")
    args = parser.parse_args()
    if args.interval < 0 or args.games < 0:
        parser.error("interval and games must be nonnegative")
    if args.model_one == args.model_two == RULE_MODEL:
        parser.error("At least one player must be a learning AI")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    scores = {1: 0, -1: 0, 0: 0}
    with httpx.Client(base_url=args.api_url, timeout=30) as client:
        count = 0
        try:
            while args.games == 0 or count < args.games:
                models = (args.model_one, args.model_two)
                if args.swap_sides and count % 2:
                    models = models[::-1]
                winner = play_game(client, *models, args.interval)
                scores[winner] += 1
                count += 1
                log.info("Game %d: winner=%s; totals=%s", count, winner, scores)
        except KeyboardInterrupt:
            log.info("Simulation stopped")
        finally:
            try:
                post(client, "/checkpoint", {})
            except httpx.HTTPError:
                log.warning("Could not request final checkpoint")


if __name__ == "__main__":
    main()
