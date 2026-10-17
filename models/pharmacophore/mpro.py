from __future__ import annotations

from rdkit import Chem

from models.pharmacophore.base import PharmacophoreConstraint, PharmacophoreProfile, PharmacophoreRule
from models.pharmacophore import rules


def _peptidomimetic_like(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    motifs = [
        "NC(=O)",      # amide-like
        "NCC(=O)",     # peptidic linker-like
    ]
    matched = any(rules.has_substructure(mol, s) for s in motifs)
    return matched, (1.0 if matched else 0.2), None if matched else "missing_peptidomimetic_feature"


def _warhead_hook_soft(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    # Soft hooks: nitrile/acrylamide-like motifs as optional compatibility.
    motifs = [
        "C#N",
        "C=CC(=O)N",
        "C=CC(=O)",
    ]
    matched = any(rules.has_substructure(mol, s) for s in motifs)
    return True, (1.0 if matched else 0.4), None


def _hbond_balance(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    hbd = rules.hbd_count(mol)
    hba = rules.hba_count(mol)
    ok = (hbd >= 1) and (hba >= 2)
    score = min(1.0, (hbd / 2.0 + hba / 6.0))
    return ok, score, None if ok else "suboptimal_hbond_balance"


def _hydrophobic_motif(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    n_ar = rules.aromatic_ring_count(mol)
    ok = n_ar >= 1
    return ok, min(1.0, n_ar / 2.0), None if ok else "missing_hydrophobic_aromatic_motif"


def build_mpro_profile() -> PharmacophoreProfile:
    constraints = [
        PharmacophoreConstraint(
            PharmacophoreRule("mpro_peptidomimetic_like", _peptidomimetic_like, hard=False, weight=1.2),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("mpro_warhead_hook_soft", _warhead_hook_soft, hard=False, weight=0.8),
            threshold=0.3,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("mpro_hbond_balance", _hbond_balance, hard=False, weight=1.0),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("mpro_hydrophobic_motif", _hydrophobic_motif, hard=False, weight=0.9),
            threshold=0.4,
        ),
    ]
    return PharmacophoreProfile(
        target_name="Mpro",
        constraints=constraints,
        description="Mpro-compatible peptidomimetic and interaction motif constraints.",
    )

