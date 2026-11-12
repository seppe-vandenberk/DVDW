from collections import deque
from functools import lru_cache
import logging
from pathlib import Path
import random
import os
import tempfile
import time

import torch
from torch import nn

from .game import COLS, ROWS, encode, legal_moves, outcome
from .rules import winning_moves

DISC_PENALTY = 0.01
REWARD_VERSION = 2
ARCHITECTURE = "tiny-cnn-7x5-v2"
LEARNING_RATE = 3e-4
GAMMA = 0.99
TRAINING_VERSION = 3
log = logging.getLogger(__name__)


def tactical_margin_loss(q, preferred, legal, margin=0.2):
    taught = preferred.any(1)
    if not taught.any():
        return q.new_zeros(())
    values, hints, allowed = q[taught], preferred[taught], legal[taught]
    best = values.masked_fill(~hints, -torch.inf).max(1).values
    competitors = (values + margin * (~hints)).masked_fill(~allowed, -torch.inf).max(1).values
    return (competitors - best).clamp_min(0).mean()


def outcome_reward(board, player, terminal):
    if not terminal:
        return 0.0
    winner = outcome(board)
    if winner in (None, 0):
        return 0.0
    discs = sum(value == player for column in board for value in column)
    magnitude = 1.0 - DISC_PENALTY * discs
    return magnitude if winner == player else -magnitude


@lru_cache(maxsize=20_000)
def tactical_actions(state):
    """Training-only labels: take a win, otherwise block a single winning threat."""
    board = [list(column) for column in state]
    wins = winning_moves(board, 1)
    if wins:
        return tuple(wins)
    threats = winning_moves(board, -1)
    return tuple(threats) if len(threats) == 1 else ()


def select_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in ("cuda", "cpu"):
        raise ValueError("DQN_DEVICE must be auto, cpu, cuda, or cuda:<index>")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA-enabled PyTorch and check the NVIDIA driver")
    return result


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 1, 1))
        self.column_bias = nn.Parameter(torch.zeros(COLS))

    def forward(self, x):
        if x.ndim not in (2, 3) or tuple(x.shape[-2:]) != (COLS, ROWS):
            raise ValueError("Expected a 7x5 board matrix or a batch of 7x5 matrices")
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        # Share one local pattern scorer across the board. The third plane marks
        # legal landing cells; no win/block rules run during inference.
        empty = x == 0
        supported = nn.functional.pad(x[:, :, 1:] != 0, (0, 1), value=True)
        landing = empty & supported
        planes = torch.stack((x == 1, x == -1, landing), dim=1).to(self.layers[0].weight.dtype)
        values = self.layers(planes).squeeze(1)
        rows = (empty.sum(2) - 1).clamp_min(0)
        q = values.gather(2, rows[:, :, None]).squeeze(2) + self.column_bias
        return q.squeeze(0) if single else q


class Agent:
    def __init__(self, path: Path, batch_size=32, device="auto"):
        self.path = path
        self.device = select_device(device)
        reset_marker = path.with_suffix(".reset")
        replacement = path.with_suffix(".next.pt")
        staged = None
        if replacement.exists():
            staged = torch.load(replacement, map_location=self.device, weights_only=True)
            if staged.get("architecture") != ARCHITECTURE:
                raise ValueError("Staged checkpoint has an incompatible architecture")
            replacement.replace(path)
            reset_marker.unlink(missing_ok=True)
        reset_requested = reset_marker.exists()
        if reset_requested:
            # A queued reset is consumed by the model-owning API, after restart.
            # An older running API may still have saved its old checkpoint meanwhile.
            path.unlink(missing_ok=True)
        self.online, self.target = DQN().to(self.device), DQN().to(self.device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=20_000)
        self.batch_size = batch_size
        self.updates = 0
        self.decisions = 0
        self.observations = 0
        self.last_td_loss = None
        self.last_tactical_loss = None
        self.last_q_min = self.last_q_max = None
        self.last_target_min = self.last_target_max = None
        self.rejected_updates = 0
        self.checkpoint_errors = 0
        self.checkpoint_error = None
        self.last_saved_at = None
        self.exploration_decay = 50_000
        self.exploration_start = 1.0
        self.evaluations = deque(maxlen=100)
        self.pretraining = None
        data = staged if staged is not None else (
            torch.load(path, map_location=self.device, weights_only=True) if path.exists() else None)
        incompatible = data is not None and data.get("architecture") != ARCHITECTURE
        if incompatible:
            log.warning("Discarding incompatible checkpoint %s; starting the tiny CNN from scratch", path)
        if data is not None and not incompatible:
            self.online.load_state_dict(data["online"])
            self.optimizer.load_state_dict(data["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = LEARNING_RATE
            self.updates, self.decisions = data["updates"], data["decisions"]
            self.exploration_start = data.get("exploration_start", 1.0)
            self.observations = data.get("observations", len(data["memory"]))
            self.evaluations.extend(data.get("evaluations", []))
            self.pretraining = data.get("pretraining")
            memory = data["memory"]
            if data.get("reward_version", 0) < REWARD_VERSION:
                # Replay next states are normalized to the acting player's sign.
                # Recompute from boards so both older reward schemes migrate once.
                memory = [(s, a, outcome_reward(ns, 1, done), ns, legal, done)
                          for s, a, reward, ns, legal, done in memory]
            self.memory.extend(memory)
            self.target.load_state_dict(data["target"])
        else:
            self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        if reset_requested or incompatible:
            self.save()
            reset_marker.unlink(missing_ok=True)

    @property
    def exploration_rate(self):
        return max(0.05, self.exploration_start - self.decisions / self.exploration_decay)

    def reset_exploration(self, rate=1.0):
        if not 0.05 <= rate <= 1.0:
            raise ValueError("Exploration rate must be between 0.05 and 1.0")
        self.exploration_start = rate
        self.decisions = 0

    def choose(self, board, player, explore):
        legal = legal_moves(board)
        epsilon = self.exploration_rate if explore else 0.0
        if explore:
            self.decisions += 1
        if explore and random.random() < epsilon:
            return random.choice(legal), epsilon
        with torch.no_grad():
            state = torch.tensor(encode(board, player), dtype=torch.float32, device=self.device)
            q = self.online(state)
            mask = torch.ones(7, dtype=torch.bool, device=self.device)
            mask[legal] = False
            return q.masked_fill(mask, -torch.inf).argmax().item(), epsilon

    def remember(self, state, action, reward, board, player, done, steps=1):
        self.observations += 1
        entry = (state, action, reward, encode(board, player), [] if done else legal_moves(board))
        self.memory.append(entry + ((done,) if steps == 1 else (GAMMA ** steps, done)))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(list(self.memory), self.batch_size)
        batch = [entry[:5] + (GAMMA, entry[-1]) if len(entry) == 6 else entry for entry in batch]
        states, actions, rewards, next_states, legal, discounts, done = zip(*batch)
        hints = [tactical_actions(tuple(tuple(col) for col in state)) for state in states]
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        mask = torch.zeros((self.batch_size, COLS), dtype=torch.bool, device=self.device)
        hint_mask = torch.zeros_like(mask)
        for i, columns in enumerate(legal):
            mask[i, columns] = True
        for i, columns in enumerate(hints):
            hint_mask[i, columns] = True
        # Horizontal reflection is a valid game symmetry; gravity is unchanged.
        reflected = torch.rand(self.batch_size, device=self.device) < 0.5
        states = torch.where(reflected[:, None, None], states.flip(1), states)
        next_states = torch.where(reflected[:, None, None], next_states.flip(1), next_states)
        actions = torch.where(reflected, COLS - 1 - actions, actions)
        mask = torch.where(reflected[:, None], mask.flip(1), mask)
        hint_mask = torch.where(reflected[:, None], hint_mask.flip(1), hint_mask)
        q = self.online(states)
        predictions = q.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: select with online, evaluate with target; mask full columns.
            actions_next = self.online(next_states).masked_fill(~mask, -torch.inf).argmax(1)
            values_next = self.target(next_states).gather(1, actions_next[:, None]).squeeze(1)
            values_next = torch.where(torch.tensor(done, device=self.device), 0.0, values_next)
            targets = torch.tensor(rewards, dtype=torch.float32, device=self.device) + torch.tensor(discounts, device=self.device) * values_next
        td_loss = nn.functional.smooth_l1_loss(predictions, targets)
        hint_loss = tactical_margin_loss(q, hint_mask, states[:, :, 0] == 0)
        loss = td_loss + 0.5 * hint_loss
        if not torch.isfinite(loss):
            self.rejected_updates += 1
            raise FloatingPointError("Non-finite training loss; update rejected")
        self.last_td_loss = td_loss.item()
        self.last_tactical_loss = hint_loss.item()
        legal_values = q.detach()[states[:, :, 0] == 0]
        self.last_q_min, self.last_q_max = legal_values.min().item(), legal_values.max().item()
        self.last_target_min, self.last_target_max = targets.min().item(), targets.max().item()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def diagnostics(self):
        return {"td_loss": self.last_td_loss, "tactical_loss": self.last_tactical_loss,
                "q_min": self.last_q_min, "q_max": self.last_q_max,
                "target_min": self.last_target_min, "target_max": self.last_target_max,
                "q_warning": self.last_q_min is not None and (self.last_q_min < -1.25 or self.last_q_max > 1.25),
                "rejected_updates": self.rejected_updates, "checkpoint_errors": self.checkpoint_errors,
                "checkpoint_error": self.checkpoint_error, "last_saved_at": self.last_saved_at,
                "learning_rate": self.optimizer.param_groups[0]["lr"], "training_version": TRAINING_VERSION}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=self.path.stem + "-", suffix=".tmp", dir=self.path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "architecture": ARCHITECTURE, "observations": self.observations,
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "decisions": self.decisions, "memory": list(self.memory),
                    "exploration_start": self.exploration_start,
                    "evaluations": list(self.evaluations),
                    "pretraining": self.pretraining,
                    "training_version": TRAINING_VERSION, "reward_version": REWARD_VERSION}, temporary)
            for attempt in range(5):
                try:
                    temporary.replace(self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.025 * 2 ** attempt)
            self.last_saved_at = time.time()
            self.checkpoint_error = None
        except OSError as exc:
            self.checkpoint_errors += 1
            self.checkpoint_error = str(exc)
            raise
        finally:
            temporary.unlink(missing_ok=True)
