from __future__ import annotations

from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.filtering import PharmacophoreResult


def compatibility_score(result: PharmacophoreResult, cfg: PharmacophoreConfig) -> float:
    """
    Soft compatibility score for logging/analysis.
    This is not intended to dominate docking-guided optimization.
    """
    if not cfg.allow_soft_scoring:
        return 0.0
    return float(max(0.0, min(1.0, result.score)))

