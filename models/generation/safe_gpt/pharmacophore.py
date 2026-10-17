from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Lipinski

from models.generation.safe_gpt.config import PharmacophoreConfig


def pharmacophore_score(smiles: str, cfg: PharmacophoreConfig) -> float:
    if not cfg.enabled:
        return 1.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    score = 1.0
    rings = mol.GetRingInfo().NumRings()
    if rings < cfg.min_ring_count:
        score -= 0.25
    hba = Lipinski.NumHAcceptors(mol)  # type: ignore[attr-defined]
    hbd = Lipinski.NumHDonors(mol)  # type: ignore[attr-defined]
    if hba < cfg.min_hba:
        score -= 0.15
    if hbd < cfg.min_hbd:
        score -= 0.10
    if cfg.require_aromatic_heterocycle:
        aromatic_hetero = any(a.GetIsAromatic() and a.GetAtomicNum() in (7, 8, 16) for a in mol.GetAtoms())
        if not aromatic_hetero:
            score -= 0.2
    return max(0.0, min(1.0, score))

