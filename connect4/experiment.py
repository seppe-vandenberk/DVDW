"""Reproducible, isolated baseline/revised RL runs; never touches live checkpoints."""
import argparse
import json
from pathlib import Path
import random
import time

import torch

from . import dqn
from .api import Trainer, MoveRequest, BoardRequest
from .evaluation import evaluate
from .game import drop, empty_board, legal_moves, outcome
from .rules import choose_rule_move


def legacy_hint(q, preferred, legal, margin=0.2):
    taught = preferred.any(1)
    if not taught.any():
        return q.new_zeros(())
    return (q[taught].masked_fill(~legal[taught], -torch.inf).logsumexp(1)
            - q[taught].masked_fill(~preferred[taught], -torch.inf).logsumexp(1)).mean()


def run(directory, seed, budget, baseline=False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(directory, device="cpu", n_steps=1 if baseline else 3)
    agent = trainer.agent("shared")
    agent.save = lambda: None  # No per-move filesystem cost in either experiment.
    trainer.measure = lambda *args, **kwargs: False
    if baseline:
        agent.exploration_decay = 10_000
        for group in agent.optimizer.param_groups:
            group["lr"] = 0.001
    original_hint = dqn.tactical_margin_loss
    if baseline:
        dqn.tactical_margin_loss = legacy_hint
    remember = agent.remember

    def bounded_remember(*args, **kwargs):
        if agent.observations < budget:
            remember(*args, **kwargs)

    agent.remember = bounded_remember
    start = time.monotonic()
    history = []
    next_check = 0
    game = 0
    pairs = []
    pair = None
    probability = 1.0 if baseline else 0.5
    try:
        while agent.observations < budget:
            if agent.observations >= next_check:
                point = evaluate(agent)
                point["diagnostics"] = agent.diagnostics()
                history.append(point)
                next_check += 2500
                if not baseline:
                    score = sum(p["score"] for p in history[-5:]) / len(history[-5:])
                    probability = 0.5 + 0.5 * min(score / 0.6, 1)
                print(json.dumps({"seed": seed, "baseline": baseline, "observations": agent.observations,
                                  "score": point["score"], "tactics": point["tactics"]}), flush=True)
            if game % 15 == 0:
                if not pairs:
                    pairs = [(0, 1), (0, 2), (1, 2)]
                    random.shuffle(pairs)
                pair = pairs.pop()
            roster = ("shared", "rules", "shared")
            models = tuple(roster[i] for i in pair)
            if game % 2:
                models = models[::-1]
            board, player = empty_board(), 1
            game_id = f"experiment-{game}"
            while outcome(board) is None and agent.observations < budget:
                model = models[0 if player == 1 else 1]
                if model == "rules":
                    column = (random.choice(legal_moves(board)) if random.random() >= probability
                              else choose_rule_move(board, player)[0])
                    board = drop(board, column, player)
                else:
                    response = trainer.move(MoveRequest(game_id=game_id, board=board, player=player,
                                                        model_id="shared", train=True, explore=True))
                    board = response["predicted_board"]
                player = -player
            if outcome(board) is not None:
                trainer.finish(BoardRequest(game_id=game_id, board=board))
            trainer.games.pop(game_id, None)
            game += 1
        point = evaluate(agent)
        point["diagnostics"] = agent.diagnostics()
        history.append(point)
        dqn.Agent.save(agent)
        result = {"seed": seed, "profile": "baseline" if baseline else "revised", "observations": agent.observations,
                  "games": game, "updates": agent.updates, "errors": trainer.errors,
                  "seconds": time.monotonic() - start, "history": history}
        (directory / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"finished": str(directory), "score": point["score"], "tactics": point["tactics"],
                          "q_min": agent.last_q_min, "q_max": agent.last_q_max}), flush=True)
        return result
    finally:
        dqn.tactical_margin_loss = original_hint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.observations <= 0:
        parser.error("observations must be positive")
    results = []
    for seed in args.seeds:
        for baseline in (True, False):
            results.append(run(args.output / f'{"baseline" if baseline else "revised"}-{seed}',
                               seed, args.observations, baseline))
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
