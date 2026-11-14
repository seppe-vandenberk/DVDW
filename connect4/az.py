"""Outcome-trained policy/value network and batched, zero-sum PUCT search."""
from collections import deque
from pathlib import Path
import os
import random
import tempfile
import time

import torch
from torch import nn

from .bitboard import Position
from .game import COLS, ROWS
from .dqn import select_device

ARCHITECTURE = "policy-value-7x5-v1"
GAMMA = 0.997


class PolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * COLS * ROWS, 128), nn.ReLU())
        self.policy = nn.Linear(128, COLS)
        self.value = nn.Linear(128, 1)

    def forward(self, boards):
        if boards.ndim == 2:
            boards = boards.unsqueeze(0)
        planes = torch.stack((boards == 1, boards == -1, boards == 0), 1).float()
        features = self.body(planes)
        return self.policy(features), self.value(features).tanh().squeeze(-1)


class Node:
    def __init__(self, position, prior=1.0):
        self.position, self.prior = position, prior
        self.visits, self.total = 0, 0.0
        self.children = {}

    @property
    def value(self):
        return self.total / self.visits if self.visits else 0.0

    def expand(self, probabilities):
        self.children = {c: Node(self.position.play(c), probabilities[c])
                         for c in self.position.legal()}


@torch.inference_mode()
def predict(model, positions, device):
    boards = torch.tensor([p.board() for p in positions], dtype=torch.float32, device=device)
    logits, values = model(boards)
    logits = logits.masked_fill(boards[:, :, 0] != 0, -torch.inf)
    return logits.softmax(1).cpu().tolist(), values.cpu().tolist()


def search_many(model, positions, device="cpu", simulations=64, noise=False, rng=None, details=False):
    """One network batch per search iteration, across independent game trees."""
    if simulations < 1 or not positions or any(p.terminal() is not None for p in positions):
        raise ValueError("Search needs live positions and a positive simulation count")
    rng = rng or random.Random(0)
    roots = [Node(p) for p in positions]
    priors, root_values = predict(model, positions, device)
    for root, probabilities in zip(roots, priors):
        root.expand(probabilities)
        if noise:
            samples = [rng.gammavariate(0.3, 1) for _ in root.children]
            total = sum(samples)
            for child, sample in zip(root.children.values(), samples):
                child.prior = 0.75 * child.prior + 0.25 * sample / total
    for _ in range(simulations):
        leaves, paths, values = [], [], []
        for root in roots:
            node, path = root, [root]
            while node.children:
                # Child values are in the opponent's perspective.
                node = max(node.children.values(), key=lambda n:
                           -GAMMA * n.value + 1.5 * n.prior * (node.visits + 1) ** 0.5 / (1 + n.visits))
                path.append(node)
            paths.append(path)
            value = node.position.terminal()
            values.append(value)
            if value is None:
                leaves.append(node)
        if leaves:
            probabilities, predictions = predict(model, [n.position for n in leaves], device)
            for node, policy in zip(leaves, probabilities):
                node.expand(policy)
            predictions = iter(predictions)
            values = [next(predictions) if value is None else value for value in values]
        for path, value in zip(paths, values):
            for node in reversed(path):
                node.visits += 1
                node.total += value
                value = -GAMMA * value
    policies = [[root.children[c].visits / simulations if c in root.children else 0.0
                 for c in range(COLS)] for root in roots]
    if not details:
        return policies
    return [{"policy": policy, "network_value": value, "search_value": root.value,
             "simulations": simulations,
             "columns": [{"column": c, "legal": c in root.children,
                          "prior": probabilities[c] if c in root.children else 0.0,
                          "visits": root.children[c].visits if c in root.children else 0,
                          "value": -GAMMA * root.children[c].value
                              if c in root.children and root.children[c].visits else None}
                         for c in range(COLS)]}
            for root, policy, probabilities, value in zip(roots, policies, priors, root_values)]


class Agent:
    def __init__(self, path, device="cpu", batch_size=128, simulations=64):
        self.path, self.device = Path(path), select_device(device)
        self.online = PolicyValue().to(self.device)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=0.001, weight_decay=0.0001)
        self.memory = deque(maxlen=100_000)
        self.human_memory = deque(maxlen=10_000)
        self.evaluations = deque(maxlen=100)
        self.batch_size, self.simulations = batch_size, simulations
        self.updates = self.decisions = self.observations = self.generation = 0
        self.last_loss = None
        self.last_policy_loss = self.last_value_loss = self.last_entropy = self.last_gradient_norm = None
        self.published_updates = 0
        self.last_published_at = None
        self.last_saved_at = None
        self.checkpoint_error = None
        self.checkpoint_errors = 0
        self.pretraining = None
        self.exploration_start = 1.0
        self.rng = random.Random(2026)
        if self.path.exists():
            data = torch.load(self.path, map_location=self.device, weights_only=True)
            if data.get("architecture") != ARCHITECTURE:
                raise ValueError("Incompatible policy/value checkpoint; use a new model directory")
            self.online.load_state_dict(data["online"])
            self.optimizer.load_state_dict(data["optimizer"])
            for name in ("updates", "decisions", "observations", "generation"):
                setattr(self, name, data.get(name, 0))
            self.memory.extend(data.get("memory", []))
            self.human_memory.extend(data.get("human_memory", []))
            self.evaluations.extend(data.get("evaluations", []))
            self.exploration_start = data.get("exploration_start", 1.0)
            if "rng_state" in data:
                self.rng.setstate(data["rng_state"])
            self.published_updates = data.get("published_updates", 0)
            self.last_published_at = data.get("last_published_at")
            if self.evaluations and "published_updates" not in data:
                promoted = [p for p in self.evaluations if p.get("promoted")]
                if promoted:
                    self.published_updates = promoted[-1]["updates"]
                    self.last_published_at = promoted[-1]["timestamp"]
            self.last_saved_at = self.path.stat().st_mtime

    @property
    def exploration_rate(self):
        # Search exploration is root noise plus early-game visit sampling.
        return self.exploration_start

    def reset_exploration(self, rate=1.0):
        self.exploration_start, self.decisions = rate, 0

    def choose(self, board, player, explore):
        policy = search_many(self.online, [Position.from_board(board, player)], self.device,
                             self.simulations, noise=explore, rng=self.rng)[0]
        self.last_policy = policy
        if explore:
            self.decisions += 1
        if explore and self.rng.random() < self.exploration_start:
            action = self.rng.choices(range(COLS), weights=policy)[0]
        else:
            action = max(range(COLS), key=lambda c: policy[c])
        return action, self.exploration_rate if explore else 0.0

    def add_episode(self, records, winner, human=False, final_ply=None):
        """Records are (normalized board, search policy, actual player, ply)."""
        target = self.human_memory if human else self.memory
        last_ply = final_ply if final_ply is not None else records[-1][3] if records else 0
        for board, policy, player, ply in records:
            result = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
            target.append((board, policy, result * GAMMA ** (last_ply - ply)))
            self.observations += 1

    def train_step(self):
        if len(self.memory) + len(self.human_memory) < self.batch_size:
            return None
        # Reserve up to 20% for human experience without duplicating it in replay.
        human_count = min(len(self.human_memory), self.batch_size // 5)
        if len(self.memory) < self.batch_size - human_count:
            human_count = self.batch_size - len(self.memory)
        batch = self.rng.sample(list(self.memory), self.batch_size - human_count)
        batch += self.rng.sample(list(self.human_memory), human_count)
        boards, policies, values = zip(*batch)
        x = torch.tensor(boards, dtype=torch.float32, device=self.device)
        pi = torch.tensor(policies, dtype=torch.float32, device=self.device)
        z = torch.tensor(values, dtype=torch.float32, device=self.device)
        reflected = torch.rand(len(batch), device=self.device) < 0.5
        x = torch.where(reflected[:, None, None], x.flip(1), x)
        pi = torch.where(reflected[:, None], pi.flip(1), pi)
        logits, value = self.online(x)
        # Finite mask keeps 0 * log(0) from producing NaNs in policy loss.
        logits = logits.masked_fill(x[:, :, 0] != 0, -1e9)
        policy_loss = -(pi * logits.log_softmax(1)).sum(1).mean()
        value_loss = (value - z).square().mean()
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy/value loss")
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 5, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.last_loss = loss.item()
        self.last_policy_loss, self.last_value_loss = policy_loss.item(), value_loss.item()
        self.last_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean().item()
        self.last_gradient_norm = gradient_norm.item()
        return self.last_loss

    def diagnostics(self):
        return {"architecture": ARCHITECTURE, "policy_value_loss": self.last_loss,
                "policy_loss": self.last_policy_loss, "value_loss": self.last_value_loss,
                "policy_entropy": self.last_entropy, "gradient_norm": self.last_gradient_norm,
                "published_updates": self.published_updates, "last_published_at": self.last_published_at,
                "generation": self.generation, "search_simulations": self.simulations,
                "human_replay_size": len(self.human_memory), "last_saved_at": self.last_saved_at,
                "checkpoint_error": self.checkpoint_error, "checkpoint_errors": self.checkpoint_errors}

    def save_generation(self):
        """Archive published weights once, never replacing an older generation."""
        directory = self.path.parent / "published" / self.path.stem
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"generation-{self.generation:06d}-update-{self.published_updates}.pt"
        if destination.exists():
            return
        original = self.path
        try:
            self.path = destination
            self.save()
        finally:
            self.path = original

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.path.stem + "-", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            torch.save({"architecture": ARCHITECTURE, "online": self.online.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "memory": list(self.memory),
                        "human_memory": list(self.human_memory), "evaluations": list(self.evaluations),
                        "updates": self.updates, "observations": self.observations,
                        "decisions": self.decisions, "generation": self.generation,
                        "exploration_start": self.exploration_start,
                        "rng_state": self.rng.getstate(), "published_updates": self.published_updates,
                        "last_published_at": self.last_published_at}, temporary)
            for attempt in range(5):
                try:
                    temporary.replace(self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.025 * 2 ** attempt)
            self.last_saved_at, self.checkpoint_error = time.time(), None
        except OSError as exc:
            self.checkpoint_errors += 1
            self.checkpoint_error = str(exc)
            raise
        finally:
            temporary.unlink(missing_ok=True)
