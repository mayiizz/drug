"""Sampling utilities for REINVENT policy networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import cast

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.diversity import RollingDiversityMemory
from models.generation.reinvent4.vocab import SMILESVocabulary


def sample_smiles_batch(
    policy: nn.Module,
    vocab: SMILESVocabulary,
    batch_size: int,
    *,
    device,
    max_len: int,
    temperature: float,
    greedy: bool = False,
) -> tuple[list[str], torch.Tensor]:
    """Explicit temperature / greedy orchestration around the policy sampler."""
    return cast(SMILESLSTM, policy).sample(
        vocab,
        batch_size,
        max_len=max_len,
        temperature=temperature,
        device=device,
        greedy=greedy,
    )


def diversity_acceptance_filter(
    smiles: list[str],
    memory: RollingDiversityMemory,
    *,
    threshold: float = 0.80,
    max_keep: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Diversity-aware acceptance/rejection for sampled molecules.
    """
    kept: list[str] = []
    reasons: dict[str, str] = {}
    for smi in smiles:
        ok, reason = memory.should_accept(smi, threshold=threshold)
        if ok:
            kept.append(smi)
            reasons[smi] = "accepted"
            if max_keep is not None and len(kept) >= max_keep:
                break
        else:
            memory.reject()
            reasons[smi] = reason
    return kept, reasons
