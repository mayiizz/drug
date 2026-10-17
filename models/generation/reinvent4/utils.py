"""Shared helpers for REINVENT SMILES-LSTM RL."""

from __future__ import annotations

import logging
from typing import Any, cast

from rdkit.Chem import DataStructs

from models.utils.fingerprint_utils import get_morgan_fingerprint

logger = logging.getLogger(__name__)


def parse_binding(result: Any) -> float:
    """
    Extract a float nM value from DeepPurpose (or compatible) binding outputs.
    Mirrors robust handling used elsewhere in CURA (e.g. safe_gpt_generator).
    """
    if result is None:
        raise ValueError("Binding model returned None")

    if isinstance(result, (int, float)):
        return _normalize_binding_to_nm(float(result))

    try:
        import numpy as np
        if isinstance(result, np.floating):
            return float(result)
        if isinstance(result, np.integer):
            return float(result)
        if isinstance(result, np.ndarray):
            if result.size == 0:
                raise ValueError("Binding model returned empty ndarray")
            return _normalize_binding_to_nm(float(result.flat[0]))
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(result, pd.Series):
            if result.empty:
                raise ValueError("Binding model returned empty Series")
            return _normalize_binding_to_nm(float(result.iloc[0]))
        if isinstance(result, pd.DataFrame):
            if result.empty:
                raise ValueError("Binding model returned empty DataFrame")
            return _normalize_binding_to_nm(float(result.iloc[0, 0]))
    except ImportError:
        pass

    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("Binding model returned empty sequence")
        return _normalize_binding_to_nm(float(result[0]))

    import numbers
    if isinstance(result, numbers.Number):
        return _normalize_binding_to_nm(float(cast(Any, result)))

    return _normalize_binding_to_nm(float(cast(Any, result)))


def _normalize_binding_to_nm(raw_value: float) -> float:
    """Normalize DeepPurpose-like outputs to nM.

    Many model checkpoints output pAffinity/pKd-like values (e.g. 6.0 means
    ~1e-6 M = 1000 nM), while some deployments return nM directly.
    Heuristic:
      - 0 < x < 20 -> treat as pAffinity and convert with 10**(9 - x)
      - otherwise  -> treat as already in nM
    """
    x = float(raw_value)
    if 0.0 < x < 20.0:
        return float(10.0 ** (9.0 - x))
    return x


def morgan_fp(smiles: str):
    """RDKit Morgan fingerprint or None."""
    return get_morgan_fingerprint(smiles)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def safe_sa(mol) -> float:
    """SA Score 1 (easy) to 10 (hard)."""
    if mol is None:
        return 5.0
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-untyped]
        score = sascorer.calculateScore(mol)
        return float(score) if score is not None else 5.0
    except Exception:
        pass
    try:
        import os
        import sys
        rdkit_contribs = os.path.join(
            os.environ.get("CONDA_PREFIX", ""),
            "share",
            "RDKit",
            "Contrib",
            "SA_Score",
        )
        if rdkit_contribs not in sys.path:
            sys.path.append(rdkit_contribs)
        import sascorer  # type: ignore[import-untyped]
        score = sascorer.calculateScore(mol)
        return float(score) if score is not None else 5.0
    except Exception:
        pass
    try:
        from rdkit.Chem import Descriptors as _D
        n_rings = mol.GetRingInfo().NumRings()
        n_stereo = sum(
            1 for a in mol.GetAtoms() if a.GetChiralTag().name != "CHI_UNSPECIFIED"
        )
        n_heavy = mol.GetNumHeavyAtoms()
        score = 2.0 + (n_rings * 0.3) + (n_stereo * 0.5) + (n_heavy * 0.02)
        return min(float(score), 9.0)
    except Exception:
        return 5.0


def encode_batch_aligned(
    vocab,
    smiles_list: list[str],
    rewards: list[float],
    device,
    min_ids: int = 3,
):
    """Padded token tensor and rewards aligned to encoded rows."""
    encoded: list[list[int]] = []
    kept_rewards: list[float] = []
    for smi, r in zip(smiles_list, rewards):
        ids = vocab.encode(smi)
        if len(ids) > min_ids:
            encoded.append(ids)
            kept_rewards.append(r)
    if not encoded:
        return None, []
    max_len = max(len(e) for e in encoded)
    pad_idx = vocab.pad_idx
    padded = [e + [pad_idx] * (max_len - len(e)) for e in encoded]

    import torch

    tensor = torch.tensor(padded, dtype=torch.long, device=device)
    return tensor, kept_rewards
