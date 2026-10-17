# Backward-compatibility shim
# Original: models/conformer_filter.py
# New location: models/synthesis/conformer_filter.py
from models.synthesis.conformer_filter import ConformerFilter

__all__ = ["ConformerFilter"]
