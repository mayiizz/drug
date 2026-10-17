"""
embedding/embedding_engine.py
==============================
Merged module combining:
  - chemberta_encoder.py   (ChemBERTaEncoder — mean-pooled CLS embeddings)
  - embedding_model.py     (ChemEmbeddingModel — CLS token embeddings with device support)

Both encoders use seyonec/ChemBERTa-zinc-base-v1.
ChemBERTaEncoder is used by the pipelines; ChemEmbeddingModel is the extended version.
"""

import torch
from transformers import AutoTokenizer, AutoModel


# ── ChemBERTaEncoder (used by drug_discovery_pipeline + denovo_pipeline) ──────

class ChemBERTaEncoder:
    """
    Light-weight ChemBERTa encoder used throughout the pipelines.
    Returns mean-pooled embedding over token dimension.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, smiles: str):
        """
        Encode a SMILES string into a fixed-size embedding vector.

        Returns:
            numpy array of shape (hidden_size,)
        """
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.numpy()[0]


# ── ChemEmbeddingModel (extended version with device support) ─────────────────

class ChemEmbeddingModel:
    """
    Extended ChemBERTa embedding model with GPU support and CLS token pooling.
    Use this when device-specific embedding or fine-tuning is needed.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def embed(self, smiles: str):
        """
        Embed a SMILES string using the CLS token.

        Returns:
            numpy array of shape (1, hidden_size)
        """
        inputs = self.tokenizer(smiles, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.cpu().numpy()


# ── Convenience alias ─────────────────────────────────────────────────────────
# Provide both names so any code using either class continues to work.
MoleculeEncoder = ChemBERTaEncoder
