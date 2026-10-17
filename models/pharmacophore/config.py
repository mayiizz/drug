from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PharmacophoreConfig:
    enabled: bool = True
    target_name: str = "EGFR"

    enforce_hard_filters: bool = False
    allow_soft_scoring: bool = True

    min_aromatic_rings: int = 1
    max_rotatable_bonds: int = 10

    fallback_to_generic: bool = True
    unknown_target_behavior: str = "generic"  # generic | skip
    strictness: str = "soft"  # soft | hard
    log_rejections: bool = True

    def use_hard_filters(self) -> bool:
        if self.strictness.lower() == "hard":
            return True
        return bool(self.enforce_hard_filters)

