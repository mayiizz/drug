"""
Backward-compatible SAFE-GPT wrapper.

Core implementation now lives under ``models.generation.safe_gpt`` as a modular,
scaffold-aware, reward-guided molecular generation subsystem.
"""

from __future__ import annotations

from models.generation.safe_gpt import SAFEGPTConfig
from models.generation.safe_gpt.safe_gpt_generator import SAFEGPTFramework


class SAFEGPTGenerator:
    def __init__(self, cfg: SAFEGPTConfig | None = None) -> None:
        self.generator = SAFEGPTFramework(cfg=cfg)

    def generate_molecules(
        self,
        seed_smiles: str,
        num_molecules: int = 20,
        protein_sequence: str = "",
        binding_model: object = None,
    ) -> list[str]:
        return self.generator.generate(
            seed_smiles=seed_smiles,
            num_molecules=num_molecules,
            protein_sequence=protein_sequence,
            binding_model=binding_model,
        )


ChemGPTGenerator = SAFEGPTGenerator

