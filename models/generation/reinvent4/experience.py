"""Experience replay for REINVENT augmented-likelihood training."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from rdkit import Chem

from models.generation.reinvent4.config import ReplayConfig

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.max_size = cfg.buffer_size
        self.smiles: list[str] = []
        self.rewards: list[float] = []
        self.priorities: list[float] = []
        self.metadata: list[dict[str, Any]] = []

    def push(
        self,
        smiles_list: list[str],
        rewards: list[float],
        metadata_by_smiles: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        for smi, r in zip(smiles_list, rewards):
            if Chem.MolFromSmiles(smi) is None:
                continue
            meta = (metadata_by_smiles or {}).get(smi, {})
            docking_success = bool(meta.get("docking_success", False))
            # Prioritize valid, docking-successful, high-reward samples.
            pr = abs(float(r)) + self.cfg.priority_epsilon
            if docking_success:
                pr *= 1.25
            self.smiles.append(smi)
            self.rewards.append(r)
            self.priorities.append(pr)
            self.metadata.append(dict(meta))
        self._truncate()

    def _truncate(self) -> None:
        if len(self.smiles) <= self.max_size:
            return
        paired = sorted(
            zip(self.rewards, self.smiles, self.priorities, self.metadata),
            reverse=True,
        )
        paired = paired[: self.max_size]
        self.rewards = [p[0] for p in paired]
        self.smiles = [p[1] for p in paired]
        self.priorities = [p[2] for p in paired]
        self.metadata = [p[3] for p in paired]

    def sample(self, n: int) -> tuple[list[str], list[float]]:
        if len(self.smiles) < n:
            return list(self.smiles), list(self.rewards)
        if not self.cfg.prioritized:
            idx = random.sample(range(len(self.smiles)), n)
            return [self.smiles[i] for i in idx], [self.rewards[i] for i in idx]

        z = sum(self.priorities) or 1.0
        probs = [p / z for p in self.priorities]
        idx = random.choices(range(len(self.smiles)), weights=probs, k=n)
        return [self.smiles[i] for i in idx], [self.rewards[i] for i in idx]

    def __len__(self) -> int:
        return len(self.smiles)

    def state_dict(self) -> dict[str, Any]:
        return {
            "smiles": list(self.smiles),
            "rewards": list(self.rewards),
            "priorities": list(self.priorities),
            "metadata": list(self.metadata),
            "max_size": self.max_size,
            "prioritized": self.cfg.prioritized,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.smiles = list(d.get("smiles", []))
        self.rewards = [float(x) for x in d.get("rewards", [])]
        self.priorities = [float(x) for x in d.get("priorities", [1.0] * len(self.smiles))]
        md = d.get("metadata")
        if isinstance(md, list):
            self.metadata = [dict(x) if isinstance(x, dict) else {} for x in md]
        else:
            self.metadata = [{} for _ in self.smiles]
        self.max_size = int(d.get("max_size", self.max_size))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict()), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
