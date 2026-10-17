from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from models.generation.reinvent4.config import resolve_prior_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedPriorCheckpoint:
    path: Path
    source: str
    is_bootstrap_fallback: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_best_prior_checkpoint() -> ResolvedPriorCheckpoint:
    root = _project_root()
    candidates: list[tuple[Path, str, bool]] = [
        (root / "checkpoints" / "prior_best_validity.pt", "best_validity", False),
        (root / "checkpoints" / "prior_latest.pt", "latest", False),
    ]
    for p, source, is_bootstrap in candidates:
        if p.exists() and p.is_file():
            logger.info("[PRIOR] selected checkpoint source=%s path=%s", source, p)
            return ResolvedPriorCheckpoint(path=p, source=source, is_bootstrap_fallback=is_bootstrap)

    bootstrap = resolve_prior_file()
    logger.warning("[PRIOR] warning: using bootstrap prior weights: %s", bootstrap)
    return ResolvedPriorCheckpoint(path=bootstrap, source="bootstrap", is_bootstrap_fallback=True)


def torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_model_state_dict(state: object) -> dict[str, torch.Tensor]:
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "agent_state_dict" in state:
        state = state["agent_state_dict"]
    if isinstance(state, dict) and "network" in state:
        state = state["network"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint state is not a dict")
    return state


def state_param_count(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state_dict.values():
        if hasattr(v, "numel"):
            total += int(v.numel())
    return total


def checkpoint_metadata(state: object) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    md: dict[str, Any] = {}
    for k in (
        "best_raw_validity",
        "best_raw_validity_step",
        "raw_valid_fraction",
        "uniqueness",
        "novelty",
        "timestamp_utc",
        "step",
        "iteration",
        "vocab_size",
    ):
        if k in state:
            md[k] = state[k]
    return md


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

