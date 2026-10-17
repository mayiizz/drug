# Backward-compatibility shim
# Original: models/reinvent_generator.py
# New location: models/generation/reinvent4_generator.py
from models.generation.safe_gpt_generator import ChemGPTGenerator

__all__ = ["ChemGPTGenerator"]
