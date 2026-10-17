# Backward-compatibility shim
# Original: models/chemberta_encoder.py
# New location: models/embedding/embedding_engine.py
from models.embedding.embedding_engine import ChemBERTaEncoder

__all__ = ["ChemBERTaEncoder"]