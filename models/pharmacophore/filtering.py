from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from models.pharmacophore.base import PharmacophoreProfile
from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.cdk2 import build_cdk2_profile
from models.pharmacophore.egfr import build_egfr_profile
from models.pharmacophore.mpro import build_mpro_profile
from models.pharmacophore.utils import canonical_target_name


@dataclass(slots=True)
class PharmacophoreResult:
    passed: bool
    score: float
    failed_rules: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    target_profile: str = "GENERIC"


def _generic_profile() -> PharmacophoreProfile:
    from models.pharmacophore.base import PharmacophoreConstraint, PharmacophoreRule
    from models.pharmacophore import rules

    def _ring(m):
        n = rules.aromatic_ring_count(m)
        ok = n >= 1
        return ok, min(1.0, n / 2.0), None if ok else "needs_aromatic_ring"

    def _rot(m):
        n = rules.rotatable_bonds(m)
        ok = n <= 10
        return ok, 1.0 if ok else 0.2, None if ok else "too_flexible"

    return PharmacophoreProfile(
        target_name="GENERIC",
        constraints=[
            PharmacophoreConstraint(PharmacophoreRule("generic_aromatic_ring", _ring, hard=False, weight=1.0), threshold=0.4),
            PharmacophoreConstraint(PharmacophoreRule("generic_rotatable", _rot, hard=False, weight=0.8), threshold=0.3),
        ],
        description="Generic medicinal-chemistry profile fallback.",
    )


def select_profile(target_name: str, cfg: PharmacophoreConfig) -> PharmacophoreProfile | None:
    canon = canonical_target_name(target_name or cfg.target_name)
    if canon == "EGFR":
        return build_egfr_profile()
    if canon == "CDK2":
        return build_cdk2_profile()
    if canon == "Mpro":
        return build_mpro_profile()
    if cfg.fallback_to_generic or cfg.unknown_target_behavior == "generic":
        return _generic_profile()
    return None


def evaluate_smiles(
    smiles: str,
    profile: PharmacophoreProfile | None,
    cfg: PharmacophoreConfig,
) -> PharmacophoreResult:
    if not cfg.enabled:
        return PharmacophoreResult(True, 1.0, target_profile=(profile.target_name if profile else "DISABLED"))
    if profile is None:
        return PharmacophoreResult(True, 0.5, target_profile="SKIPPED")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PharmacophoreResult(False, 0.0, failed_rules=["invalid_smiles"], target_profile=profile.target_name)

    passed, score, failed, matched = profile.evaluate(
        mol,
        enforce_hard_filters=cfg.use_hard_filters(),
    )
    if not cfg.use_hard_filters():
        passed = True
    return PharmacophoreResult(
        passed=bool(passed),
        score=float(score),
        failed_rules=failed,
        matched_rules=matched,
        target_profile=profile.target_name,
    )


def filter_smiles_batch(
    smiles_list: list[str],
    *,
    target_name: str,
    cfg: PharmacophoreConfig,
) -> tuple[list[str], dict[str, PharmacophoreResult]]:
    profile = select_profile(target_name, cfg)
    out: list[str] = []
    details: dict[str, PharmacophoreResult] = {}
    for smi in smiles_list:
        r = evaluate_smiles(smi, profile, cfg)
        details[smi] = r
        if r.passed:
            out.append(smi)
    return out, details

