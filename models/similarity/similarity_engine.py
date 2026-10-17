"""
similarity/similarity_engine.py
================================
Merged module combining:
  - fingerprint_engine.py           (FingerprintEngine — Morgan FP generation)
  - fingerprint_similarity_engine.py (FingerprintSimilarityEngine — ChEMBL search)
  - similarity_engine.py            (was misnamed; contained SideEffectEngine)

This module provides:
  - FingerprintEngine           : SMILES → Morgan fingerprint (numpy array)
  - FingerprintSimilarityEngine : find top-K similar molecules in ChEMBL DB
  - tanimoto_similarity()       : standalone Tanimoto helper

Note: SideEffectEngine (formerly mislabeled in similarity_engine.py) has been
      moved to models/side_effect_engine.py to keep concerns separate.
"""

import numpy as np
import pandas as pd
from rdkit.DataStructs import ConvertToNumpyArray

from models.utils.fingerprint_utils import get_morgan_fingerprint


# ── FingerprintEngine ─────────────────────────────────────────────────────────

class FingerprintEngine:
    """Converts SMILES to Morgan fingerprint numpy arrays."""

    def smiles_to_fingerprint(self, smiles: str):
        """
        Convert SMILES to a Morgan-2 / 2048-bit fingerprint as a numpy array.

        Returns:
            numpy int array of shape (2048,), or None if SMILES is invalid.
        """
        fp = get_morgan_fingerprint(smiles)
        if fp is None:
            return None
        arr = np.zeros((2048,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        return arr


# ── Standalone helper ─────────────────────────────────────────────────────────

def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two binary numpy fingerprints."""
    intersection = np.logical_and(fp1, fp2).sum()
    union        = np.logical_or(fp1, fp2).sum()
    return float(intersection / union) if union > 0 else 0.0


# ── FingerprintSimilarityEngine ───────────────────────────────────────────────

class FingerprintSimilarityEngine:
    """
    Fast Tanimoto-based similar drug search over the ChEMBL-100K database.

    Requires:
      - data/chembl_100k.csv       : molecule database with 'smiles' column
      - data/chembl_fingerprints.npy : pre-computed fingerprint matrix
    """

    def __init__(self):
        self.db           = pd.read_csv("data/chembl_100k.csv")
        self.fingerprints = np.load("data/chembl_fingerprints.npy", mmap_mode="r")
        self.fp_engine    = FingerprintEngine()

    def find_similar(self, smiles: str, k: int = 5) -> pd.DataFrame:
        """
        Find the top-K molecules in the ChEMBL database most similar to `smiles`.

        Returns:
            DataFrame with original columns + 'similarity' column, sorted descending.
        """
        query_fp     = self.fp_engine.smiles_to_fingerprint(smiles)
        similarities = np.array([tanimoto_similarity(query_fp, fp) for fp in self.fingerprints])

        top_indices = np.argsort(similarities)[-k:][::-1]
        top_indices = top_indices[top_indices < len(self.db)]

        results               = self.db.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        return results
