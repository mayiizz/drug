"""
Selectivity-oriented scoring: hERG / CYP penalties and optional anti-target binding.
Extensible — no hard-coded protein targets on anti-target list (from config).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from models.generation.reinvent4.config import SelectivityConfig
from models.generation.reinvent4.utils import parse_binding

logger = logging.getLogger(__name__)


class SelectivityScorer:
    """Returns a score in [0, 1]; higher is safer / more selective."""

    def __init__(self, config: SelectivityConfig):
        self.cfg = config

    def _herg_cyp_penalty(self, smiles: str) -> float:
        if not self.cfg.enable_selectivity:
            return 0.0
        try:
            from models.admet.admet_engine import get_admet_profile

            profile = get_admet_profile(smiles)
            penalty = 0.0
            admet = profile.get("admetlab") or {}

            herg = admet.get("hERG")
            if herg is not None:
                try:
                    h = float(herg)
                    if h > self.cfg.herg_threshold:
                        penalty += self.cfg.w_herg_penalty * (h - self.cfg.herg_threshold)
                except (TypeError, ValueError):
                    pass

            cyp = admet.get("CYP3A4_inhibition")
            if cyp is None:
                cyp = admet.get("cyp3a4")
            if cyp is not None:
                try:
                    c = float(cyp)
                    if c > self.cfg.cyp_threshold:
                        penalty += self.cfg.w_cyp_penalty * (c - self.cfg.cyp_threshold)
                except (TypeError, ValueError):
                    pass

            return min(1.0, max(0.0, penalty))
        except Exception as e:
            logger.debug("Selectivity ADMET lookup failed: %s", e)
            return 0.0

    def _antitarget_penalty(
        self,
        smiles: str,
        binding_predictor: Any | None,
    ) -> float:
        if not self.cfg.enable_selectivity or not self.cfg.anti_target_sequences:
            return 0.0
        if binding_predictor is None:
            return 0.0
        pen = 0.0
        for seq in self.cfg.anti_target_sequences:
            if not seq or len(seq) < 10:
                continue
            try:
                raw = binding_predictor.predict(smiles, seq)
                nm = parse_binding(raw)
                # Strong off-target bind → penalty (sigmoid rising with potency)
                sig = 1.0 / (1.0 + math.exp(-0.02 * (300.0 - nm)))
                pen += self.cfg.w_antitarget_binding * sig
            except Exception:
                continue
        return min(1.0, pen)

    def score(
        self,
        smiles: str,
        mol_ok: bool,
        binding_predictor: Any | None = None,
    ) -> float:
        if not mol_ok or not self.cfg.enable_selectivity:
            return 1.0
        p = self._herg_cyp_penalty(smiles) + self._antitarget_penalty(smiles, binding_predictor)
        return max(0.0, 1.0 - p)
