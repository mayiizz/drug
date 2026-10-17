from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

from benchmarks.utils import morgan_fp


@dataclass(frozen=True)
class KnownInhibitor:
    name: str
    target: str
    smiles: str
    potency_nm: float | None = None


KNOWN_INHIBITORS: tuple[KnownInhibitor, ...] = (
    KnownInhibitor("Gefitinib", "EGFR", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", 33.0),
    KnownInhibitor("Erlotinib", "EGFR", "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1", 2.0),
    KnownInhibitor("Afatinib", "EGFR", "CN(C)C/C=C/C(=O)Nc1ccc(F)c(Cl)c1Oc1ncnc2cc(OCCCN3CCOCC3)ccc12", 0.5),
    KnownInhibitor("Osimertinib", "EGFR", "COc1cc(N(C)C)c(NC(=O)C=C)c2ncnc(Nc3cccc(c3)Cl)c12", 0.3),
    KnownInhibitor("Vemurafenib", "BRAF", "CCCS(=O)(=O)Nc1ccc(F)cc1F", 31.0),
    KnownInhibitor("Dabrafenib", "BRAF", "CC(C)(C)N1CCN(CC1)C(=O)c1ccc(F)c(Nc2ncnc3[nH]ccc23)c1", 0.7),
)


def inhibitor_subset(target: str) -> list[KnownInhibitor]:
    t = (target or "").upper()
    return [k for k in KNOWN_INHIBITORS if k.target.upper() == t]


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    fp_a = morgan_fp(smiles_a)
    fp_b = morgan_fp(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(TanimotoSimilarity(fp_a, fp_b))


def scaffold_similarity(smiles_a: str, smiles_b: str) -> float:
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return 0.0
    try:
        sa = MurckoScaffold.GetScaffoldForMol(ma)
        sb = MurckoScaffold.GetScaffoldForMol(mb)
        if sa is None or sb is None:
            return 0.0
        s1 = Chem.MolToSmiles(sa)
        s2 = Chem.MolToSmiles(sb)
        return tanimoto_similarity(s1, s2)
    except Exception:
        return 0.0


def nearest_known_inhibitor(smiles: str, target: str | None = None) -> dict[str, Any]:
    pool = inhibitor_subset(target) if target else list(KNOWN_INHIBITORS)
    if not pool:
        pool = list(KNOWN_INHIBITORS)
    best = None
    best_t = -1.0
    best_s = -1.0
    for inhibitor in pool:
        ts = tanimoto_similarity(smiles, inhibitor.smiles)
        ss = scaffold_similarity(smiles, inhibitor.smiles)
        if ts > best_t:
            best_t = ts
            best_s = ss
            best = inhibitor
    if best is None:
        return {"name": "", "target": "", "tanimoto": 0.0, "scaffold_similarity": 0.0}
    return {
        "name": best.name,
        "target": best.target,
        "potency_nm": best.potency_nm,
        "tanimoto": float(best_t),
        "scaffold_similarity": float(best_s),
    }
