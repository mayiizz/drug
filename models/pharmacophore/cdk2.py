from __future__ import annotations

from rdkit import Chem

from models.pharmacophore.base import PharmacophoreConstraint, PharmacophoreProfile, PharmacophoreRule
from models.pharmacophore import rules


def _aromatic_heterocycle_pref(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    n = rules.hetero_aromatic_ring_count(mol)
    ok = n >= 1
    return ok, min(1.0, n / 2.0), None if ok else "missing_aromatic_heterocycle"


def _planar_aromatic_bias(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    n_ar = rules.aromatic_ring_count(mol)
    ok = n_ar >= 1
    return ok, min(1.0, n_ar / 3.0), None if ok else "insufficient_planar_aromaticity"


def _donor_acceptor_balance(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    hbd = rules.hbd_count(mol)
    hba = rules.hba_count(mol)
    ok = (hbd <= 3) and (hba >= 2)
    score = 1.0 if ok else 0.3
    return ok, score, None if ok else "poor_hbond_balance"


def build_cdk2_profile() -> PharmacophoreProfile:
    constraints = [
        PharmacophoreConstraint(
            PharmacophoreRule("cdk2_aromatic_heterocycle", _aromatic_heterocycle_pref, hard=True, weight=1.4),
            threshold=0.5,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("cdk2_planar_aromatic", _planar_aromatic_bias, hard=False, weight=1.0),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("cdk2_hbond_balance", _donor_acceptor_balance, hard=False, weight=1.1),
            threshold=0.5,
        ),
    ]
    return PharmacophoreProfile(
        target_name="CDK2",
        constraints=constraints,
        description="CDK2 hinge-compatible aromatic heterocycle and donor/acceptor balance.",
    )

