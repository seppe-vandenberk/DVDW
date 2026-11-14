"""Discover existing checkpoints and serve them without modifying their files."""
import hashlib
import os
from pathlib import Path

import torch
from fastapi import HTTPException

from .az import Agent, ARCHITECTURE
from .dqn import ARCHITECTURE as LEGACY, DQN, Agent as LegacyAgent


class FrozenAgent(Agent):
    def __init__(self, path, device, simulations):
        # Initialize common telemetry without loading replay or optimizer state.
        super().__init__(path.with_suffix('.catalog-init'), device=device, simulations=simulations)
        self.path = path
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.architecture = data.get('architecture')
        if self.architecture == LEGACY:
            self.online = DQN().to(self.device)
        elif self.architecture != ARCHITECTURE:
            raise HTTPException(409, 'Unsupported checkpoint architecture')
        self.online.load_state_dict(data['online'])
        self.online.eval()
        for key in ('updates', 'observations', 'decisions', 'generation', 'published_updates', 'last_published_at'):
            setattr(self, key, data.get(key, getattr(self, key)))
        self.evaluations.extend(data.get('evaluations', []))

    def choose(self, board, player, explore):
        if self.architecture == LEGACY:
            action, _ = LegacyAgent.choose(self, board, player, False)
            self.last_policy = [float(c == action) for c in range(7)]
            return action, 0.0
        return super().choose(board, player, False)

    def diagnostics(self):
        return {**super().diagnostics(), 'architecture': self.architecture, 'read_only': True}

    def save(self):
        pass  # Catalog entries are immutable inference snapshots.


class Catalog:
    def __init__(self, model_dir):
        self.roots = [Path(model_dir), Path(os.getenv('REPO_MODEL_RUNTIME', 'runtime'))]
        self.cache, self.entries = {}, {}

    def scan(self):
        rows = []
        for root in self.roots:
            if not root.exists():
                continue
            paths = []
            for directory, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not (d.startswith(('.', 'hf-', 'realsense-', 'test_')) or 'tests' in d or '-qa-' in d or d == '__pycache__')]
                paths.extend(Path(directory) / f for f in files if f.endswith('.pt'))
            for path in paths:
                # Exclude third-party detector weights and package caches.
                if path.name in ('sam3.pt', 'sam3.1.pt'):
                    continue
                if not path.is_file():
                    continue
                stat = path.stat()
                signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
                key = 'saved-' + hashlib.sha256(repr(signature).encode()).hexdigest()[:24]
                if key not in self.cache:
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=True)
                        architecture = data.get('architecture')
                        history = data.get('evaluations', [])
                        # A candidate's latest check measures its weights; a serving file
                        # must use the check for the last published weights.
                        published = data.get('published_updates', data.get('updates', 0))
                        checks = [h for h in history if h.get('updates') == published and h.get('promoted', True)]
                        benchmark = history[-1] if '.learner.' in path.name and history else (checks[-1] if checks else None)
                        row = {'id': key, 'label': str(path.relative_to(root)), 'architecture': architecture,
                               'generation': data.get('generation'), 'updates': data.get('updates', 0),
                               'playable': architecture in (ARCHITECTURE, LEGACY), 'benchmark': benchmark,
                               'modified_at': stat.st_mtime}
                    except Exception as exc:
                        row = {'id':key, 'label':str(path.relative_to(root)), 'playable':False, 'error':str(exc)}
                    self.cache[key] = row
                    self.entries[key] = (path, signature)
                rows.append(self.cache[key])
        return sorted(rows, key=lambda r:r.get('modified_at',0), reverse=True)

    def load(self, name, device, simulations):
        if name not in self.entries:
            self.scan()
        if name not in self.entries:
            raise HTTPException(404, 'Unknown saved model; refresh the model list')
        path, signature = self.entries[name]
        stat = path.stat()
        if (str(path.resolve()), stat.st_mtime_ns, stat.st_size) != signature:
            raise HTTPException(409, 'Checkpoint changed; refresh and select it again')
        return FrozenAgent(path, device, simulations)
