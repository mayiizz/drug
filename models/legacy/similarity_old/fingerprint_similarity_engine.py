# Backward-compatibility shim
# Original: models/fingerprint_similarity_engine.py
# New location: models/similarity/similarity_engine.py
from models.similarity.similarity_engine import FingerprintSimilarityEngine

__all__ = ["FingerprintSimilarityEngine"]