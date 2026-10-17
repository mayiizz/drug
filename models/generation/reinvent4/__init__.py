"""
REINVENT 4-inspired modular SMILES-LSTM RL for CURA.

Prefer ``from models.generation.reinvent4_generator import ChemGPTGenerator`` for the
public facade. Submodules import without pulling the facade (no heavy side effects).
"""

from __future__ import annotations

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.config import (
    DockingRLConfig,
    LEGACY_WEIGHTS_DIR,
    PRIOR_FILENAME,
    PRIMARY_WEIGHTS_DIR,
    Reinvent4Config,
    default_config,
    resolve_prior_file,
    resolve_prior_paths,
)
from models.generation.reinvent4.experience import ReplayBuffer
from models.generation.reinvent4.trainer import ReinventTrainer
from models.generation.reinvent4.vocab import SMILESVocabulary

__all__ = [
    "ReinventTrainer",
    "Reinvent4Config",
    "DockingRLConfig",
    "default_config",
    "resolve_prior_file",
    "resolve_prior_paths",
    "PRIMARY_WEIGHTS_DIR",
    "LEGACY_WEIGHTS_DIR",
    "PRIOR_FILENAME",
    "SMILESVocabulary",
    "SMILESLSTM",
    "ReplayBuffer",
]


def __getattr__(name: str):
    if name == "ChemGPTGenerator":
        from models.generation.reinvent4_generator import ChemGPTGenerator
        return ChemGPTGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
