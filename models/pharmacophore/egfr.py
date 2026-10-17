from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

from models.pharmacophore.base import PharmacophoreConstraint, PharmacophoreProfile, PharmacophoreRule
from models.pharmacophore import rules


HINGE_MOTIFS: list[str] = [
    "n1cnc(nc1)",     # pyrimidine-like
    "n1ccnc(n1)",     # aza-heteroaromatic hinge-like
    "n1nccnc1",       # triazine-like
    "c1nc2ncccc2n1",  # quinazoline-like fused bicyclic
    "c1ncc2cc[nH]c2n1",  # azaindole-like
]


def _hinge_binder_detection_score(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    matches = sum(1 for s in HINGE_MOTIFS if rules.has_substructure(mol, s))
    aromatic_n = rules.aromatic_n_count(mol)
    fused_like = _fused_or_bicyclic_heteroaromatic(mol)[0]
    raw = 0.4 * min(1.0, matches / 2.0) + 0.3 * min(1.0, aromatic_n / 2.0) + (0.3 if fused_like else 0.0)
    score = max(0.0, min(1.0, raw))
    ok = score >= 0.35
    return ok, score, None if ok else "weak_hinge_binder_motif"


def _fused_or_bicyclic_heteroaromatic(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    ri = mol.GetRingInfo()
    ring_count = ri.NumRings()
    fused_bonds = 0
    for ring in ri.BondRings():
        for b in ring:
            if sum(1 for br in ri.BondRings() if b in br) >= 2:
                fused_bonds += 1
                break
    hetero_aromatic_rings = rules.hetero_aromatic_ring_count(mol)
    score = min(1.0, 0.3 * min(ring_count, 4) / 4.0 + 0.4 * min(hetero_aromatic_rings, 3) / 3.0 + 0.3 * min(fused_bonds, 2) / 2.0)
    ok = ring_count >= 2 and hetero_aromatic_rings >= 1
    return ok, score, None if ok else "needs_fused_heteroaromatic_bicyclic_system"


def _hba_hbd_balance_rule(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    hba = rules.hba_count(mol)
    hbd = rules.hbd_count(mol)
    ratio = hba / max(1, hbd + 1)
    score = 1.0 - min(1.0, abs(ratio - 1.5) / 2.5)
    ok = 1 <= hbd <= 3 and 2 <= hba <= 8
    return ok, max(0.0, score), None if ok else "unbalanced_hbond_profile"


def _aromatic_ring_preference(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    n = rules.aromatic_ring_count(mol)
    score = 1.0 - min(1.0, abs(n - 3) / 3.0)  # prefer 2-4 with peak around 3
    ok = 2 <= n <= 4
    return ok, max(0.0, score), None if ok else "aromatic_ring_count_out_of_range"


def _flexibility_penalty_rule(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    rot = rules.rotatable_bonds(mol)
    heavy = max(1, mol.GetNumHeavyAtoms())
    flex_index = rot / heavy
    score = 1.0 - min(1.0, max(0.0, flex_index - 0.18) / 0.25)
    ok = rot <= 8 and flex_index <= 0.35
    return ok, max(0.0, score), None if ok else "excessive_flexibility_or_long_linkers"


def _lipophilic_flatness_penalty_rule(mol: Chem.Mol) -> tuple[bool, float, str | None]:
    logp = Descriptors.MolLogP(mol)
    aromatic = rules.aromatic_ring_count(mol)
    hetero_aromatic = rules.hetero_aromatic_ring_count(mol)
    aliphatic_heavy = sum(1 for a in mol.GetAtoms() if not a.GetIsAromatic() and a.GetAtomicNum() == 6)
    total_heavy = max(1, mol.GetNumHeavyAtoms())
    aliphatic_frac = aliphatic_heavy / total_heavy
    score = (
        0.35 * (1.0 - min(1.0, max(0.0, logp - 4.2) / 2.5))
        + 0.35 * min(1.0, hetero_aromatic / max(1, aromatic))
        + 0.30 * (1.0 - min(1.0, max(0.0, aliphatic_frac - 0.65) / 0.35))
    )
    ok = logp <= 5.5 and aliphatic_frac <= 0.75
    return ok, max(0.0, min(1.0, score)), None if ok else "flat_lipophilic_or_aliphatic_bias"


def build_egfr_profile() -> PharmacophoreProfile:
    constraints = [
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_fused_bicyclic_heteroaromatic", _fused_or_bicyclic_heteroaromatic, hard=False, weight=1.2),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_hinge_binding_score", _hinge_binder_detection_score, hard=False, weight=1.4),
            threshold=0.35,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_aromatic_ring_preference", _aromatic_ring_preference, hard=False, weight=1.0),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_hba_hbd_balance", _hba_hbd_balance_rule, hard=False, weight=1.0),
            threshold=0.4,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_moderate_rigidity", _flexibility_penalty_rule, hard=False, weight=1.0),
            threshold=0.35,
        ),
        PharmacophoreConstraint(
            PharmacophoreRule("egfr_lipophilicity_flatness_control", _lipophilic_flatness_penalty_rule, hard=False, weight=0.9),
            threshold=0.3,
        ),
    ]
    return PharmacophoreProfile(
        target_name="EGFR",
        constraints=constraints,
        description="EGFR kinase-focused soft pharmacophore profile with hinge-compatible heteroaromatic motif preference.",
    )

