# Backward-compatibility shim
# Original: models/esm_fold_client.py
# New location: models/protein/structure_engine.py
from models.protein.structure_engine import ESMFoldClient

__all__ = ["ESMFoldClient"]
