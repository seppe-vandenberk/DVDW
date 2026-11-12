"""Warm-start the tiny CNN from at most 10,000 labeled, legal board positions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import torch

from .dqn import Agent
from .evaluation import evaluate
from .game import drop, empty_board, encode, legal_moves, outcome
from .rules import choose_rule_move, winning_moves


def tactical_suite(per_kind=128, seed=41027):
    """Held-out positions from legal random games; never used as training data."""
    rng = random.Random(seed)
    cases = {"win": [], "block": []}
    seen = set()
    while any(len(group) < per_kind for group in cases.values()):
        board, player = empty_board(), 1
        while outcome(board) is None:
            wins = winning_moves(board, player)
            threats = winning_moves(board, -player)
            kind = "win" if wins else "block" if len(threats) == 1 else None
            state = encode(board, player)
            key = tuple(tuple(col) for col in state)
            if kind and len(cases[kind]) < per_kind and key not in seen:
                cases[kind].append((state, wins if wins else threats))
                seen.add(key)
            board = drop(board, rng.choice(legal_moves(board)), player)
            player = -player
    return cases


def tactical_accuracy(agent, suite):
    scores = {}
    with torch.no_grad():
        for kind, cases in suite.items():
            states = torch.tensor([case[0] for case in cases], dtype=torch.float32, device=agent.device)
            choices = agent.online(states).masked_fill(states[:, :, 0] != 0, -torch.inf).argmax(1).tolist()
            scores[f"{kind}_accuracy"] = sum(action in case[1] for action, case in zip(choices, cases)) / len(cases)
    return scores


def demonstrations(count, forbidden=(), seed=2026, on_progress=None):
    rng = random.Random(seed)
    quotas = [count * 3 // 10, count * 3 // 10, count - 2 * (count * 3 // 10)]
    groups = [[], [], []]
    seen = set(forbidden)
    total = 0
    while total < count:
        board, player = empty_board(), 1
        while outcome(board) is None:
            state = encode(board, player)
            key = tuple(tuple(col) for col in state)
            wins = winning_moves(board, player)
            threats = winning_moves(board, -player)
            kind = 0 if wins else 1 if len(threats) == 1 else 2
            if (key not in seen and len(groups[kind]) < quotas[kind]
                    and (kind != 2 or not threats)):
                targets = wins if wins else threats
                if kind == 2:
                    targets = [choose_rule_move(board, player, rng)[0]]
                mask = [column in targets for column in range(7)]
                groups[kind].append((state, mask, 2.0 if kind < 2 else 1.0))
                seen.add(key)
                total += 1
                if on_progress and (total % 1000 == 0 or total == count):
                    on_progress({"positions": total, "by_kind": [len(group) for group in groups]})
            action = (rng.choice(legal_moves(board)) if rng.random() < 0.8
                      else choose_rule_move(board, player, rng)[0])
            board = drop(board, action, player)
            player = -player
    return [case for group in groups for case in group]


def train(agent, observations=10_000, epochs=100, on_report=None):
    start = time.monotonic()
    validation = tactical_suite(seed=41027)
    held_out = tactical_suite(per_kind=256, seed=937451)
    forbidden = set()
    for suite in (validation, held_out):
        for cases in suite.values():
            for state, _ in cases:
                key = tuple(tuple(col) for col in state)
                forbidden.update((key, key[::-1]))

    data = demonstrations(observations, forbidden, on_progress=lambda value: print(json.dumps(value), flush=True))
    states = torch.tensor([case[0] for case in data], dtype=torch.float32, device=agent.device)
    targets = torch.tensor([case[1] for case in data], dtype=torch.bool, device=agent.device)
    weights = torch.tensor([case[2] for case in data], device=agent.device)
    agent.observations = len(data)
    best_score, best = -1.0, None
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=50, gamma=0.3)

    def report(epoch, final=False):
        point = evaluate(agent)
        accuracy = tactical_accuracy(agent, held_out if final else validation)
        if final:
            agent.pretraining.update(accuracy)
        point.update({(key if final else "validation_" + key): value for key, value in accuracy.items()})
        point.update({"observations": agent.observations, "epoch": epoch, "final": final,
                      "elapsed_seconds": time.monotonic() - start})
        agent.target.load_state_dict(agent.online.state_dict())
        agent.evaluations.append(point)
        agent.save()
        if on_report:
            on_report(point)

    report(0)
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(len(data), device=agent.device)
        for batch in indices.split(256):
            x, preferred = states[batch], targets[batch]
            reflected = torch.rand(len(batch), device=agent.device) < 0.5
            x = torch.where(reflected[:, None, None], x.flip(1), x)
            preferred = torch.where(reflected[:, None], preferred.flip(1), preferred)
            q = agent.online(x)
            legal_logits = (4 * q).masked_fill(x[:, :, 0] != 0, -torch.inf)
            target_logits = (4 * q).masked_fill(~preferred, -torch.inf)
            losses = legal_logits.logsumexp(1) - target_logits.logsumexp(1)
            loss = (losses * weights[batch]).mean() + 0.01 * q.square().mean()
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 10)
            agent.optimizer.step()
            agent.updates += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            accuracy = tactical_accuracy(agent, validation)
            score = (accuracy["win_accuracy"] + accuracy["block_accuracy"]) / 2
            if score > best_score:
                best_score = score
                best = deepcopy((agent.online.state_dict(), agent.optimizer.state_dict(), agent.updates))
            report(epoch)
            if min(accuracy.values()) >= 0.98:
                break
    online, optimizer, updates = best
    agent.online.load_state_dict(online)
    agent.optimizer.load_state_dict(optimizer)
    agent.updates = updates
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.001
    agent.pretraining = {"method": "rule-demonstrations", "positions": len(data),
                         "held_out_cases_per_kind": 256}
    agent.reset_exploration(0.25)
    report(epoch, final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runtime/cnn-teacher-10000.pt"))
    parser.add_argument("--observations", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.observations, args.epochs) < 1:
        parser.error("Observation and epoch counts must be positive")
    if args.model.exists():
        parser.error("Choose a new output path for a fresh, bounded training run")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agent = Agent(args.model, device=args.device)
    reports = list(agent.evaluations)

    def report(point):
        reports.append(point)
        args.model.with_suffix(".json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps(point), flush=True)

    print(json.dumps({"device": str(agent.device), "parameters": sum(p.numel() for p in agent.online.parameters())}), flush=True)
    train(agent, args.observations, args.epochs, report)


if __name__ == "__main__":
    main()
