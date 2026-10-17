"""Structured logging for REINVENT RL."""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def log_rl_iteration(
    iteration: int,
    total_iters: int,
    valid_count: int,
    batch_size: int,
    mean_reward_raw: float,
    mean_reward_norm: float,
    loss: float,
    n_unique: int,
    docking_success_rate: float | None = None,
    best_vina_score: float | None = None,
    diversity_buffer_occupancy: int | None = None,
    diversity_rejected: int | None = None,
    diversity_mean_tanimoto: float | None = None,
    diversity_unique_scaffolds: int | None = None,
    diversity_scaffold_entropy: float | None = None,
    top_smiles: Sequence[str] | None = None,
) -> None:
    msg = (
        f"iter={iteration+1}/{total_iters} valid={valid_count}/{batch_size} "
        f"mean_R_raw={mean_reward_raw:.4f} mean_R_norm={mean_reward_norm:.4f} "
        f"loss={loss:.4f} unique={n_unique}"
    )
    if docking_success_rate is not None:
        msg += f" dock_succ_rate={docking_success_rate:.3f}"
    if best_vina_score is not None:
        msg += f" best_vina={best_vina_score:.3f}"
    if diversity_buffer_occupancy is not None:
        msg += f" div_buf={diversity_buffer_occupancy}"
    if diversity_rejected is not None:
        msg += f" div_rej={diversity_rejected}"
    if diversity_mean_tanimoto is not None:
        msg += f" div_mean_tani={diversity_mean_tanimoto:.3f}"
    if diversity_unique_scaffolds is not None:
        msg += f" div_scaf_unique={diversity_unique_scaffolds}"
    if diversity_scaffold_entropy is not None:
        msg += f" div_scaf_H={diversity_scaffold_entropy:.3f}"
    logger.debug(msg)
    if top_smiles:
        preview = ", ".join(s[:40] + ("..." if len(s) > 40 else "") for s in top_smiles[:3])
        logger.debug("top_smiles: %s", preview)


def log_grad_norm(model, tag: str = "agent") -> None:
    import torch

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
    total = total**0.5
    logger.debug("%s_grad_norm=%.4f", tag, total)
