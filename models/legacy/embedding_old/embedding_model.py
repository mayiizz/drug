# Backward-compatibility shim
# Original: models/embedding_model.py
# New location: models/embedding/embedding_engine.py
from models.embedding.embedding_engine import ChemEmbeddingModel

__all__ = ["ChemEmbeddingModel"]