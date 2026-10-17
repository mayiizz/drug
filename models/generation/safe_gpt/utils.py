from __future__ import annotations

import os
from typing import Any, cast

from rdkit.Chem import DataStructs

from models.utils.fingerprint_utils import get_morgan_fingerprint


def parse_binding(result: Any) -> float:
    if result is None:
        raise ValueError("Binding model returned None")
    if isinstance(result, (int, float)):
        return float(result)
    try:
        import numpy as np
        if isinstance(result, np.floating):
            return float(result)
        if isinstance(result, np.integer):
            return float(result)
        if isinstance(result, np.ndarray):
            if result.size == 0:
                raise ValueError("Binding model returned empty ndarray")
            return float(result.flat[0])
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(result, pd.Series):
            if result.empty:
                raise ValueError("Binding model returned empty Series")
            return float(result.iloc[0])
        if isinstance(result, pd.DataFrame):
            if result.empty:
                raise ValueError("Binding model returned empty DataFrame")
            return float(result.iloc[0, 0])
    except ImportError:
        pass
    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("Binding model returned empty sequence")
        return float(result[0])
    return float(cast(Any, result))


def morgan_fp(smiles: str):
    return get_morgan_fingerprint(smiles)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def safe_sa(mol) -> float:
    if mol is None:
        return 5.0
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-untyped]
        score = sascorer.calculateScore(mol)
        return float(score) if score is not None else 5.0
    except Exception:
        pass
    try:
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
        n_rings = mol.GetRingInfo().NumRings()
        n_stereo = sum(
            1 for a in mol.GetAtoms() if a.GetChiralTag().name != "CHI_UNSPECIFIED"
        )
        n_heavy = mol.GetNumHeavyAtoms()
        score = 2.0 + (n_rings * 0.3) + (n_stereo * 0.5) + (n_heavy * 0.02)
        return min(float(score), 9.0)
    except Exception:
        return 5.0

