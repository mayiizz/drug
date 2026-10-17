from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from models.generation.safe_gpt.config import MedicinalPriorsConfig


def medicinal_prior_score(smiles: str, cfg: MedicinalPriorsConfig) -> float:
    if not cfg.enabled:
        return 1.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    score = 1.0
    rings = mol.GetRingInfo().NumRings()
    aromatic_rings = sum(
        1 for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )
    hetero_aromatic_rings = sum(
        1 for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        and any(mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16) for i in ring)
    )
    rot = Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]

    kinase_motifs = [
        "n1cnc(nc1)",         # pyrimidine-like
        "c1nc2ncccc2n1",      # quinazoline-like
        "c1ncc2cc[nH]c2n1",   # azaindole-like
    ]
    if any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in kinase_motifs if Chem.MolFromSmarts(s) is not None):
        score += 0.12
    if rings > cfg.ring_complexity_penalty_start:
        score -= 0.2
    if cfg.prefer_heterocycles:
        hetero_ring = any(a.GetAtomicNum() in (7, 8, 16) and a.IsInRing() for a in mol.GetAtoms())
        if not hetero_ring:
            score -= 0.15
        else:
            score += min(0.10, 0.04 * hetero_aromatic_rings)
    if cfg.oral_like_soft_bias:
        mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        logp = Crippen.MolLogP(mol)  # type: ignore[attr-defined]
        if mw > 600 or logp > 5.8:
            score -= 0.2
        if 250 <= mw <= 550 and 1.0 <= logp <= 4.8:
            score += 0.08
    if cfg.cns_like_soft_bias:
        tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
        if tpsa > 90:
            score -= 0.1
    if aromatic_rings < 2:
        score -= 0.12
    if aromatic_rings > 4:
        score -= 0.08
    if rot > 9:
        score -= 0.12

    # Reactive/unstable chemistry and unrealistic medicinal patterns.
    bad_patterns = [
        "[N+](=O)[O-]",      # nitro
        "[NX2]=[O]",         # nitroso
        "[CX3H1](=O)[#6]",   # aldehyde
        "C=CC(=O)[#6]",      # michael acceptor (enone-like)
    ]
    for smarts in bad_patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            score -= 0.18

    halogen_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    if halogen_count >= 4:
        score -= 0.10

    # Symmetry proxy: repeated atom invariants can indicate unrealistic symmetry.
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol))
        if len(set(ranks)) < max(2, int(0.35 * len(ranks))):
            score -= 0.08
    except Exception:
        pass

    if rings >= 7:
        score -= 0.12
    return max(0.0, min(1.0, score))

