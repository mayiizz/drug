"""Staged reward weights: validity → QED → binding → selectivity → diversity."""

from __future__ import annotations

from models.generation.reinvent4.config import Reinvent4Config, CurriculumConfig


class CurriculumScheduler:
    """Maps iteration index → dict of reward component weights (sums to ~1)."""

    def __init__(self, config: Reinvent4Config):
        self.cfg = config
        self.cu = config.curriculum
        base = {
            "binding": config.w_binding,
            "qed": config.w_qed,
            "sa": config.w_sa,
            "novelty": config.w_novelty,
            "validity": config.w_validity,
            "selectivity": getattr(config, "w_selectivity", 0.0),
            "diversity": getattr(config, "w_diversity_bonus", 0.0),
        }
        self._base = self._normalize(base)

    def _normalize(self, w: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, v) for v in w.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in w.items()}

    def weights(self, iteration: int) -> dict[str, float]:
        cu: CurriculumConfig = self.cu
        w = dict(self._base)

        if iteration <= cu.stage_validity_end:
            for k in w:
                w[k] *= 0.1
            w["validity"] = 1.0 - 1e-6
        elif iteration <= cu.stage_qed_end:
            w["binding"] *= 0.3
            w["qed"] = max(w["qed"], 0.45)
            w["novelty"] *= 0.5
        elif iteration <= cu.stage_binding_end:
            w["binding"] = max(w["binding"], 0.5)
            w["novelty"] *= 0.85
        elif iteration <= cu.stage_selectivity_end:
            w["binding"] *= 0.9
            if w.get("selectivity", 0) > 0:
                w["selectivity"] = max(w["selectivity"], 0.15)
        else:
            if w.get("diversity", 0) >= 0:
                w["novelty"] = max(w["novelty"], 0.2)
                w["diversity"] = max(w.get("diversity", 0.0), 0.05)

        return self._normalize(w)
