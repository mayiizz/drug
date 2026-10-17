# Backward-compatibility shim
# Original: models/fingerprint_engine.py
# New location: models/similarity/similarity_engine.py
from models.similarity.similarity_engine import FingerprintEngine

__all__ = ["FingerprintEngine"]