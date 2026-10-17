"""
REINVENT-style SMILES-LSTM molecular generator — thin facade.

RL logic lives in ``models.generation.reinvent4``. This module exposes the
historic ``ChemGPTGenerator`` API and backward-compatible constants.
"""

from __future__ import annotations

from typing import Any

import torch

from models.generation.reinvent4.config import (
    Reinvent4Config,
    default_config,
    resolve_prior_paths,
)
from models.generation.reinvent4.checkpoint_utils import resolve_best_prior_checkpoint
from models.generation.reinvent4.scoring import CompositeScorer
from models.generation.reinvent4.trainer import ReinventTrainer


def _snapshot_constants(cfg: Reinvent4Config) -> dict[str, Any]:
    primary, legacy = resolve_prior_paths()
    resolved = resolve_best_prior_checkpoint()
    return {
        "WEIGHTS_DIR": primary,
        "PRIOR_PATH": resolved.path,
        "LEGACY_WEIGHTS_DIR": legacy,
        "WEIGHTS_URLS": list(cfg.weight_urls),
        "WEIGHTS_URL": cfg.weight_urls[0],
        "RL_ITERATIONS": cfg.rl_iterations,
        "BATCH_SIZE": cfg.batch_size,
        "LEARNING_RATE": cfg.learning_rate,
        "SIGMA": cfg.sigma,
        "REPLAY_BUFFER_SIZE": cfg.replay.buffer_size,
        "REPLAY_BATCH": cfg.replay.batch_size,
        "W_BINDING": cfg.w_binding,
        "W_QED": cfg.w_qed,
        "W_SA": cfg.w_sa,
        "W_NOVELTY": cfg.w_novelty,
        "W_VALIDITY": cfg.w_validity,
        "BINDING_SIGMOID_CENTRE": cfg.binding_sigmoid_centre_nm,
        "BINDING_SIGMOID_SLOPE": cfg.binding_sigmoid_slope,
    }


_cfg_default = default_config()
snap = _snapshot_constants(_cfg_default)
WEIGHTS_DIR = snap["WEIGHTS_DIR"]
PRIOR_PATH = snap["PRIOR_PATH"]
LEGACY_WEIGHTS_DIR = snap["LEGACY_WEIGHTS_DIR"]
WEIGHTS_URLS: list[str] = snap["WEIGHTS_URLS"]
WEIGHTS_URL: str = snap["WEIGHTS_URL"]

RL_ITERATIONS = snap["RL_ITERATIONS"]
BATCH_SIZE = snap["BATCH_SIZE"]
LEARNING_RATE = snap["LEARNING_RATE"]
SIGMA = snap["SIGMA"]
REPLAY_BUFFER_SIZE = snap["REPLAY_BUFFER_SIZE"]
REPLAY_BATCH = snap["REPLAY_BATCH"]

W_BINDING = snap["W_BINDING"]
W_QED = snap["W_QED"]
W_SA = snap["W_SA"]
W_NOVELTY = snap["W_NOVELTY"]
W_VALIDITY = snap["W_VALIDITY"]
BINDING_SIGMOID_CENTRE = snap["BINDING_SIGMOID_CENTRE"]
BINDING_SIGMOID_SLOPE = snap["BINDING_SIGMOID_SLOPE"]


def compute_rewards(
    smiles_list: list[str],
    seed_fp,
    binding_model,
    protein_sequence: str,
    weight_override: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Backward-compatible batch reward tensor (same ordering as legacy).
    """
    scorer = CompositeScorer(default_config())
    t, _ = scorer.reward_vector(
        smiles_list,
        seed_fp,
        binding_model,
        protein_sequence or "",
        weight_override=weight_override,
    )
    return t


class ChemGPTGenerator:
    """
    SMILES-LSTM policy with frozen ChEMBL prior and augmented-likelihood RL.

    Public API::
        generate_molecules(seed_smiles, num_molecules=20,
                           protein_sequence=\"\", binding_model=None) -> list[str]
    """

    def __init__(self, config: Reinvent4Config | None = None):
        self.config = config or Reinvent4Config()
        self._trainer = ReinventTrainer(self.config)

    def generate_molecules(
        self,
        seed_smiles: str,
        num_molecules: int = 20,
        protein_sequence: str = "",
        binding_model: Any = None,
    ) -> list[str]:
        return self._trainer.generate(
            seed_smiles,
            num_molecules=num_molecules,
            protein_sequence=protein_sequence,
            binding_model=binding_model,
        )


__all__ = [
    "ChemGPTGenerator",
    "compute_rewards",
    "WEIGHTS_DIR",
    "PRIOR_PATH",
    "LEGACY_WEIGHTS_DIR",
    "WEIGHTS_URLS",
    "WEIGHTS_URL",
    "RL_ITERATIONS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "SIGMA",
    "REPLAY_BUFFER_SIZE",
    "REPLAY_BATCH",
    "W_BINDING",
    "W_QED",
    "W_SA",
    "W_NOVELTY",
    "W_VALIDITY",
    "BINDING_SIGMOID_CENTRE",
    "BINDING_SIGMOID_SLOPE",
]
