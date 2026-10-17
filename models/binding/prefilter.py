from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from models.utils.smiles_validation import validate_smiles_strict


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    smiles: str
    raw_score: float | None
    norm_score: float | None
    valid: bool
    reason: str | None = None


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _percentile(values: list[float], q: float) -> float:
    """
    Deterministic percentile (linear interpolation).
    q in [0,1].
    """
    if not values:
        raise ValueError("percentile on empty list")
    xs = sorted(values)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    t = pos - lo
    return xs[lo] * (1.0 - t) + xs[hi] * t


def robust_normalize_log10kd(raw_scores: list[float]) -> list[float]:
    """
    Normalize arbitrary DeepPurpose scores (log10(Kd)-like) to [0,1] where
    higher means better (stronger binding).

    We use a robust mapping:
      z = (median - x) / IQR
      norm = sigmoid(z)

    Rationale:
      - DeepPurpose output is a surrogate score; normalization is for reporting
        and consistent shortlisting. We do NOT interpret it as an absolute affinity.
    """
    if not raw_scores:
        return []

    med = _percentile(raw_scores, 0.5)
    q25 = _percentile(raw_scores, 0.25)
    q75 = _percentile(raw_scores, 0.75)
    iqr = max(1e-8, q75 - q25)

    norms: list[float] = []
    for x in raw_scores:
        z = (med - x) / iqr
        norms.append(float(_sigmoid(z)))
    return norms


class DeepPurposePrefilter:
    """
    DeepPurposePrefilter

    Scientific role:
      - Fast neural pre-screening / approximate affinity prioritization.
      - Produces a shortlist for docking (Vina), which is the authoritative
        final structure-based scoring stage when available.
    """

    def __init__(
        self,
        binding_model: Any,
        *,
        cache_dir: Path | None = None,
        cache_enabled: bool = False,
    ):
        self.binding_model = binding_model
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = cache_dir
        if self.cache_enabled and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def predict_batch(self, smiles_list: list[str], protein_sequence: str) -> list[PrefilterResult]:
        """
        Batch neural pre-screen using DeepPurpose (surrogate model).

        Returns one PrefilterResult per input SMILES in the same order.
        """
        if not smiles_list:
            return []

        # Validate SMILES early to avoid wasting model calls.
        valid_mask: list[bool] = []
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi.strip():
                valid_mask.append(False)
                continue
            valid_mask.append(validate_smiles_strict(smi) is not None)

        raw_scores: list[float | None] = [None] * len(smiles_list)
        reasons: list[str | None] = [None] * len(smiles_list)

        try:
            scores = self.binding_model.predict_batch(smiles_list, protein_sequence)
            # DeepPurposeBindingPredictor returns float-like scores, with invalids as 999.0
            for i, s in enumerate(scores):
                try:
                    raw_scores[i] = float(s)
                except Exception:
                    raw_scores[i] = None
                    reasons[i] = "score_parse_failed"
        except Exception as e:
            # If DeepPurpose fails, return empty scores; caller will decide fallback.
            logger.warning("DeepPurpose batch prefilter failed: %s", e)
            for i in range(len(smiles_list)):
                raw_scores[i] = None
                reasons[i] = "deep_purpose_failed"

        # Normalize only on finite “real” scores.
        finite_scores: list[float] = []
        finite_idx: list[int] = []
        for i, rs in enumerate(raw_scores):
            if rs is None:
                continue
            # Treat 999 as invalid sentinel (DeepPurposeBindingPredictor convention).
            if rs >= 900:
                continue
            finite_scores.append(rs)
            finite_idx.append(i)

        norms: list[float] = robust_normalize_log10kd(finite_scores) if finite_scores else []
        norm_scores: list[float | None] = [None] * len(smiles_list)
        for j, i in enumerate(finite_idx):
            norm_scores[i] = norms[j]

        results: list[PrefilterResult] = []
        for i, smi in enumerate(smiles_list):
            ok = bool(valid_mask[i]) and (raw_scores[i] is not None) and (raw_scores[i] < 900)
            if not valid_mask[i]:
                results.append(PrefilterResult(smiles=smi, raw_score=None, norm_score=None, valid=False, reason="invalid_smiles"))
            elif raw_scores[i] is None:
                results.append(PrefilterResult(smiles=smi, raw_score=None, norm_score=None, valid=False, reason=reasons[i] or "no_score"))
            elif raw_scores[i] >= 900:
                results.append(PrefilterResult(smiles=smi, raw_score=float(raw_scores[i]), norm_score=None, valid=False, reason="invalid_or_failed"))
            else:
                results.append(PrefilterResult(smiles=smi, raw_score=float(raw_scores[i]), norm_score=norm_scores[i], valid=True))

        return results

    def shortlist_candidates(
        self,
        results: list[PrefilterResult],
        *,
        top_k: int | None = None,
        top_percent: float | None = None,
        affinity_threshold: float | None = None,
    ) -> list[str]:
        """
        Create a shortlist from PrefilterResult entries.

        DeepPurpose outputs are treated as surrogate scores (log10(Kd)-like), so
        we shortlist primarily by **relative ordering**: lower raw_score is better.
        """
        valid = [r for r in results if r.valid and r.raw_score is not None]

        # Optional threshold (raw score scale): keep scores <= threshold.
        if affinity_threshold is not None:
            try:
                thr = float(affinity_threshold)
                valid = [r for r in valid if float(r.raw_score) <= thr]
            except Exception:
                pass

        # Sort by “better binding first”: lower raw score.
        valid.sort(key=lambda r: (float(r.raw_score), r.smiles))

        if not valid:
            return []

        if top_percent is not None:
            try:
                p = float(top_percent)
                if p > 1.0:
                    p = p / 100.0
                p = max(0.0, min(1.0, p))
                k = max(1, int(math.ceil(len(valid) * p)))
                valid = valid[:k]
            except Exception:
                pass

        if top_k is not None:
            try:
                k2 = int(top_k)
                if k2 > 0:
                    valid = valid[: min(k2, len(valid))]
            except Exception:
                pass

        return [r.smiles for r in valid]

