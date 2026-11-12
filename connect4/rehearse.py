"""Time-budgeted, isolated STEM-day training rehearsal; never overwrites live models."""
import argparse
import json
from pathlib import Path
from threading import RLock
import time

import torch

from .az import Agent
from .learning import Worker, match


def run(directory, seconds, seed, device, batch_games, simulations, gate_games, interval, test_games):
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    agent = Agent(directory / "az-shared.pt", device=device, simulations=simulations)
    agent.rng.seed(seed)
    worker = Worker(agent, RLock(), batch_games, simulations, interval, gate_games)
    start = time.monotonic()
    worker.start()
    try:
        while time.monotonic() - start < seconds and worker.thread.is_alive():
            worker.stop.wait(min(10, max(0, seconds - (time.monotonic() - start))))
            status = worker.snapshot()
            print(json.dumps({"seed": seed, "seconds": round(time.monotonic() - start, 1),
                              "games": status["games"], "updates": agent.updates,
                              "observations": agent.observations, "error": status["error"]}), flush=True)
    finally:
        worker.close()
        agent.save()
    if worker.status["error"]:
        raise RuntimeError(worker.status["error"])
    elapsed = time.monotonic() - start
    candidate = Agent(agent.path.with_suffix(".learner.pt"), device=device, simulations=simulations)
    # Distinct seeds from the promotion checks; report candidate and published model separately.
    benchmarks = {}
    for label, model in (("published", agent.online), ("candidate", candidate.online)):
        benchmarks[label] = {opponent: match(model, opponent, agent.device, test_games,
                                             simulations, seed=771101)
                             for opponent in ("random", "rules", "search")}
    summary = {"seed": seed, "training_seconds": elapsed, "simulations": simulations,
               "batch_games": batch_games, "games": worker.status["games"],
               "observations": candidate.observations, "updates": candidate.updates,
               "games_per_second": worker.status["games"] / elapsed,
               "generation": agent.generation, "benchmarks": benchmarks,
               "history": list(agent.evaluations)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="New output directory")
    parser.add_argument("--seconds", type=float, default=18000, help="Training budget per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 43, 29])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--gate-games", type=int, default=40)
    parser.add_argument("--evaluation-seconds", type=float, default=300)
    parser.add_argument("--test-games", type=int, default=100)
    args = parser.parse_args()
    if (args.seconds <= 0 or args.evaluation_seconds <= 0
            or min(args.batch_games, args.simulations, args.gate_games, args.test_games) < 1):
        parser.error("Budgets and counts must be positive")
    if args.gate_games % 2 or args.test_games % 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use even game counts and distinct seeds")
    if args.directory.exists():
        parser.error("Choose a new directory to preserve previous rehearsals")
    torch.set_num_threads(1)
    summaries = [run(args.directory / str(seed), args.seconds, seed, args.device,
                     args.batch_games, args.simulations, args.gate_games,
                     args.evaluation_seconds, args.test_games) for seed in args.seeds]
    (args.directory / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
