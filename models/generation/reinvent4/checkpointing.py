"""Training checkpoints: agent, optimizer, replay, curriculum step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    agent_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    replay_state: dict[str, Any] | None,
    iteration: int,
    config_dict: dict[str, Any] | None = None,
    reward_state: dict[str, Any] | None = None,
    diversity_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_state_dict": agent_state,
        "optimizer_state_dict": optimizer_state,
        "replay": replay_state,
        "iteration": iteration,
        "config": config_dict,
        "reward_state": reward_state or {},
        "diversity_state": diversity_state or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def config_to_jsonable(cfg: Any) -> dict[str, Any]:
    """Shallow dump of Reinvent4Config for metadata (best-effort)."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return {"repr": repr(cfg)}
