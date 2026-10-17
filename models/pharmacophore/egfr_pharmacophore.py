from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from models.utils.smiles_validation import validate_smiles_detailed

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EgfrMotif:
    name: str
    query: str
    weight: float
    kind: str = "smarts"  # smarts | smiles


EGFR_MOTIFS: tuple[EgfrMotif, ...] = (
    EgfrMotif("quinazoline", "c1nc2ncccc2cc1", 0.24, "smarts"),
    EgfrMotif("pyrimidine", "n1ccnc(n1)", 0.16, "smarts"),
    EgfrMotif("aminopyrimidine", "Nc1nccnc1", 0.18, "smarts"),
    EgfrMotif("azaindole_like", "c1cc2[nH]cnc2n1", 0.16, "smarts"),
    EgfrMotif("aminopyridine_like", "Nc1ccccn1", 0.12, "smarts"),
    EgfrMotif("fused_n_heteroaromatic", "c1ccc2ncccc2c1", 0.12, "smarts"),
    EgfrMotif("hinge_donor_acceptor", "[nH]-c:n", 0.08, "smarts"),
    EgfrMotif("aromatic_heterocycle", "n1cccc1", 0.08, "smarts"),
)

MEDCHEM_ALERTS: tuple[EgfrMotif, ...] = (
    EgfrMotif("azo_motif", "[N]=[N]", 0.35),
    EgfrMotif("triazene_motif", "[NX3]-[N]=[N]", 0.45),
    EgfrMotif("nitroso_motif", "[NX2]=[O]", 0.40),
    EgfrMotif("strained_exotic_small_ring", "[r3,r4][n,o,s][r3,r4]", 0.22),
)

KNOWN_EGFR_DRUGS: dict[str, str] = {
    "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib": "COc1cc2ncnc(Nc3ccc(C#C)c(Cl)c3)c2cc1OCCOC",
    "Afatinib": "CN(C)C/C=C/C(=O)Nc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Osimertinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C)CC1",
}


def _build_query(motif: EgfrMotif) -> Chem.Mol | None:
    if motif.kind == "smiles":
        return Chem.MolFromSmiles(motif.query)
    return Chem.MolFromSmarts(motif.query)


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))


def score_egfr_pharmacophore(smiles: str) -> dict[str, Any]:
    ok, mol, reason = validate_smiles_detailed(smiles)
    if not ok or mol is None:
        return {
            "pharmacophore_score": 0.0,
            "matched_motifs": [],
            "reason": reason,
        }

    matched: list[str] = []
    score = 0.0
    for motif in EGFR_MOTIFS:
        q = _build_query(motif)
        if q is not None and mol.HasSubstructMatch(q):
            matched.append(motif.name)
            score += motif.weight
            logger.info("[PHARMACOPHORE] matched %s motif", motif.name)

    aromatic_rings = Lipinski.RingCount(mol)
    hetero_aromatic = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() in (7, 8, 16)
    )
    score += 0.06 * min(3.0, float(aromatic_rings))
    score += 0.04 * min(3.0, float(hetero_aromatic))
    score = float(min(1.0, score))
    return {
        "pharmacophore_score": score,
        "matched_motifs": matched,
        "aromatic_ring_count": int(aromatic_rings),
        "hetero_aromatic_atoms": int(hetero_aromatic),
    }


def score_medchem_penalty(smiles: str) -> dict[str, Any]:
    ok, mol, reason = validate_smiles_detailed(smiles)
    if not ok or mol is None:
        return {"medchem_penalty": 1.0, "reasons": [reason]}

    penalty = 0.0
    reasons: list[str] = []
    for alert in MEDCHEM_ALERTS:
        q = _build_query(alert)
        if q is not None and mol.HasSubstructMatch(q):
            penalty += alert.weight
            reasons.append(alert.name)
            logger.info("[MEDCHEM] penalty applied: %s", alert.name)

    aromatic_ring_count = Lipinski.RingCount(mol)
    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_flatness = float(aromatic_atoms) / float(heavy_atoms)
    if aromatic_ring_count >= 5 and aromatic_flatness > 0.72:
        reasons.append("excessive_aromatic_flatness")
        penalty += 0.20
        logger.info("[MEDCHEM] penalty applied: excessive aromatic flatness")

    halogen_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    if halogen_count >= 4:
        reasons.append("over_halogenated")
        penalty += 0.18
        logger.info("[MEDCHEM] penalty applied: over-halogenated structure")

    return {
        "medchem_penalty": float(min(1.25, penalty)),
        "reasons": reasons,
        "aromatic_flatness": float(aromatic_flatness),
        "halogen_count": int(halogen_count),
    }


def score_kinase_descriptor_bias(smiles: str) -> dict[str, Any]:
    ok, mol, reason = validate_smiles_detailed(smiles)
    if not ok or mol is None:
        return {"kinase_bias_score": 0.0, "reason": reason}

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    hbd = int(Lipinski.NumHDonors(mol))

    mw_s = _normalize(mw, 250.0, 550.0)
    logp_s = 1.0 - abs(_normalize(logp, 1.5, 4.5) - 0.5) * 2.0
    tpsa_s = _normalize(tpsa, 60.0, 120.0)
    hba_s = _normalize(float(hba), 4.0, 10.0)
    hbd_s = _normalize(float(hbd), 1.0, 4.0)
    score = max(0.0, min(1.0, 0.30 * mw_s + 0.20 * logp_s + 0.25 * tpsa_s + 0.15 * hba_s + 0.10 * hbd_s))
    return {
        "kinase_bias_score": float(score),
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hba": hba,
        "hbd": hbd,
    }


def score_egfr_reference_overlap(smiles: str) -> dict[str, Any]:
    """Soft benchmark against known EGFR inhibitor motif/scaffold overlap.

    Intended only as analysis and weak guidance signal - not template copying.
    """
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold

    ok, mol, reason = validate_smiles_detailed(smiles)
    if not ok or mol is None:
        return {"reference_similarity": 0.0, "reason": reason}

    fp = Chem.RDKFingerprint(mol)
    sc = MurckoScaffold.GetScaffoldForMol(mol)
    sc_smi = Chem.MolToSmiles(sc) if sc is not None else ""

    sims: dict[str, float] = {}
    scaffold_hits: list[str] = []
    for name, ref_smi in KNOWN_EGFR_DRUGS.items():
        ref = Chem.MolFromSmiles(ref_smi)
        if ref is None:
            continue
        sims[name] = float(DataStructs.FingerprintSimilarity(fp, Chem.RDKFingerprint(ref)))
        ref_sc = MurckoScaffold.GetScaffoldForMol(ref)
        ref_sc_smi = Chem.MolToSmiles(ref_sc) if ref_sc is not None else ""
        if sc_smi and ref_sc_smi and sc_smi == ref_sc_smi:
            scaffold_hits.append(name)
    best = max(sims.values()) if sims else 0.0
    return {
        "reference_similarity": float(best),
        "reference_similarities": sims,
        "scaffold_overlap_drugs": scaffold_hits,
    }


__all__ = [
    "EGFR_MOTIFS",
    "MEDCHEM_ALERTS",
    "KNOWN_EGFR_DRUGS",
    "score_egfr_pharmacophore",
    "score_medchem_penalty",
    "score_kinase_descriptor_bias",
    "score_egfr_reference_overlap",
]

