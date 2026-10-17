"""
drug_pipeline_engine.py
=======================
Validation utilities for CURA molecule analysis.

DEPRECATED: orchestration has moved to ``models.runtime.cura_pipeline.CuraPipeline``.
This module now only carries pure validation helpers (DrugScorer,
CandidateSelector, validate_molecule, validate_batch) consumed by the
``/validate`` endpoint and a few exporters. The legacy ``DrugDesignEngine``
(MolGPT/ChemGPTGenerator + BatchScreeningEngine orchestrator) has been
removed; the unified ``CuraPipeline`` is the single canonical runtime.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED

from models.utils.fingerprint_utils import get_morgan_fingerprint

# SA Score — uses RDKit contrib (included in rdkit >= 2020)
try:
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    from rdkit.Contrib.SA_Score import sascorer
    _HAS_SA = True
except Exception:
    try:
        import sys, os
        for p in sys.path:
            sa_path = os.path.join(p, "rdkit", "Contrib", "SA_Score")
            if os.path.isdir(sa_path):
                sys.path.insert(0, sa_path)
                import sascorer as sascorer
                _HAS_SA = True
                break
        else:
            _HAS_SA = False
    except Exception:
        _HAS_SA = False


# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "mol_weight": {
        "pass_max":  500,
        "warn_max":  600,
        "unit":      "Da",
        "label":     "Molecular Weight",
    },
    "logp": {
        "pass_min": -0.4,
        "pass_max":  5.0,
        "warn_max":  6.0,
        "unit":      "",
        "label":     "LogP",
    },
    "tpsa": {
        "pass_max":  140,
        "warn_max":  180,
        "unit":      "Å²",
        "label":     "TPSA",
    },
    "rotatable_bonds": {
        "pass_max":  10,
        "warn_max":  15,
        "unit":      "",
        "label":     "Rotatable Bonds",
    },
    "qed": {
        "pass_min":  0.5,
        "warn_min":  0.3,
        "unit":      "",
        "label":     "QED Score",
    },
    "sa_score": {
        "pass_max":  4.0,
        "warn_max":  6.0,
        "unit":      "/10",
        "label":     "SA Score",
    },
    "novelty": {
        "pass_max":  0.85,
        "warn_max":  0.95,
        "unit":      "",
        "label":     "Novelty (Tanimoto vs seed)",
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tanimoto(mol_a, mol_b) -> float:
    fp_a = get_morgan_fingerprint(Chem.MolToSmiles(mol_a))
    fp_b = get_morgan_fingerprint(Chem.MolToSmiles(mol_b))
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def _verdict(value, param: str) -> str:
    t = THRESHOLDS.get(param, {})
    if param in ("mol_weight", "tpsa", "rotatable_bonds", "sa_score", "novelty"):
        pass_max = t.get("pass_max", float("inf"))
        warn_max = t.get("warn_max", float("inf"))
        if value <= pass_max:
            return "PASS"
        elif value <= warn_max:
            return "WARN"
        else:
            return "FAIL"
    if param == "qed":
        if value >= t.get("pass_min", 0):
            return "PASS"
        elif value >= t.get("warn_min", 0):
            return "WARN"
        else:
            return "FAIL"
    if param == "logp":
        pass_min = t.get("pass_min", -float("inf"))
        pass_max = t.get("pass_max", float("inf"))
        warn_max = t.get("warn_max", float("inf"))
        if pass_min <= value <= pass_max:
            return "PASS"
        elif value <= warn_max:
            return "WARN"
        else:
            return "FAIL"
    return "PASS"


def _lipinski(mol):
    """Returns (violations, is_drug_like, detail_dict)."""
    mw  = Descriptors.MolWt(mol)
    lp  = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, lp > 5, hbd > 5, hba > 10])
    detail = {
        "mol_weight_ro5":  {"value": round(mw, 2),  "limit": 500, "ok": mw <= 500},
        "logp_ro5":        {"value": round(lp, 2),  "limit": 5,   "ok": lp <= 5},
        "h_bond_donors":   {"value": hbd,            "limit": 5,   "ok": hbd <= 5},
        "h_bond_acceptors":{"value": hba,            "limit": 10,  "ok": hba <= 10},
    }
    return violations, violations <= 1, detail


# ── DrugScorer ────────────────────────────────────────────────────────────────

class DrugScorer:
    """Lipinski Rule of Five + MW-weighted drug score."""

    def lipinski_score(self, mol) -> int:
        """Returns count of Ro5 rules satisfied (0–4)."""
        mw          = Descriptors.MolWt(mol)
        logp        = Descriptors.MolLogP(mol)
        h_donors    = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        score = 0
        if mw <= 500:          score += 1
        if logp <= 5:          score += 1
        if h_donors <= 5:      score += 1
        if h_acceptors <= 10:  score += 1
        return score

    def drug_score(self, smiles: str) -> float:
        """Combined Lipinski + MW drug score."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        lipinski  = self.lipinski_score(mol)
        mw        = Descriptors.MolWt(mol)
        mw_score  = 1 - abs(mw - 350) / 350
        return lipinski + mw_score


# ── CandidateSelector ─────────────────────────────────────────────────────────

class CandidateSelector:
    """Selects top-K candidates from a scored results list."""

    def select_top_candidates(self, screening_results: list, top_k: int = 5) -> list:
        if not screening_results:
            return []
        sorted_results = sorted(
            screening_results,
            key=lambda x: x.get("final_score", 0),
            reverse=True,
        )
        return sorted_results[:top_k]


# ── validate_molecule / validate_batch (from drug_validator) ──────────────────

def validate_molecule(smiles: str, seed_smiles: str = None) -> dict:
    """
    Validate a single molecule against 8 drug parameters.

    Returns dict with keys:
        smiles, valid, overall_score (0–1), verdict (PASS/WARN/FAIL),
        parameters (per-param breakdown), lipinski, disqualified, disqualification_reason
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "smiles":                  smiles,
            "valid":                   False,
            "overall_score":           0.0,
            "verdict":                 "FAIL",
            "disqualified":            True,
            "disqualification_reason": "Invalid SMILES — RDKit could not parse.",
            "parameters":              {},
            "lipinski":                {},
        }

    mw          = round(Descriptors.MolWt(mol), 2)
    logp        = round(Descriptors.MolLogP(mol), 2)
    tpsa        = round(Descriptors.TPSA(mol), 2)
    rot_bonds   = Descriptors.NumRotatableBonds(mol)
    qed_score   = round(QED.qed(mol), 3)
    h_donors    = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    sa = None
    if _HAS_SA:
        try:
            sa = round(sascorer.calculateScore(mol), 2)
        except Exception:
            pass

    novelty_sim = None
    if seed_smiles:
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol:
            novelty_sim = round(_tanimoto(seed_mol, mol), 3)

    lip_violations, lip_pass, lip_detail = _lipinski(mol)

    parameters = {}
    parameters["mol_weight"]       = {"label": "Molecular Weight",   "value": mw,        "unit": "Da",      "verdict": _verdict(mw,        "mol_weight"),       "ideal": "≤ 500 Da"}
    parameters["logp"]             = {"label": "LogP",               "value": logp,      "unit": "",        "verdict": _verdict(logp,      "logp"),             "ideal": "-0.4 – 5.0"}
    parameters["tpsa"]             = {"label": "TPSA",               "value": tpsa,      "unit": "Å²",      "verdict": _verdict(tpsa,      "tpsa"),             "ideal": "≤ 140 Å²"}
    parameters["rotatable_bonds"]  = {"label": "Rotatable Bonds",    "value": rot_bonds, "unit": "",        "verdict": _verdict(rot_bonds, "rotatable_bonds"),  "ideal": "≤ 10"}
    parameters["qed"]              = {"label": "QED Score",          "value": qed_score, "unit": "",        "verdict": _verdict(qed_score, "qed"),              "ideal": "≥ 0.5 (max 1.0)"}
    parameters["h_donors"]         = {"label": "H-Bond Donors",      "value": h_donors,  "unit": "",        "verdict": "PASS" if h_donors <= 5  else "FAIL",    "ideal": "≤ 5"}
    parameters["h_acceptors"]      = {"label": "H-Bond Acceptors",   "value": h_acceptors,"unit": "",       "verdict": "PASS" if h_acceptors <= 10 else "FAIL", "ideal": "≤ 10"}

    if sa is not None:
        parameters["sa_score"] = {"label": "Synthetic Accessibility", "value": sa,          "unit": "/10",    "verdict": _verdict(sa,          "sa_score"),        "ideal": "≤ 4.0 (1=easy, 10=impossible)"}
    if novelty_sim is not None:
        parameters["novelty"]  = {"label": "Novelty vs Seed",         "value": novelty_sim, "unit": "Tanimoto","verdict": _verdict(novelty_sim, "novelty"),         "ideal": "< 0.85 (sufficiently different from seed)"}

    disqualified = False
    disqualification_reason = None
    if mw > 800:
        disqualified = True; disqualification_reason = f"Molecular weight too high ({mw} Da > 800 Da)"
    elif logp > 8:
        disqualified = True; disqualification_reason = f"LogP too high ({logp} > 8) — likely toxic/insoluble"
    elif lip_violations >= 3:
        disqualified = True; disqualification_reason = f"Too many Lipinski violations ({lip_violations}/4)"
    elif novelty_sim is not None and novelty_sim >= 0.99:
        disqualified = True; disqualification_reason = "Identical to seed compound (no novelty)"
    elif sa is not None and sa > 7:
        disqualified = True; disqualification_reason = f"Synthetically inaccessible (SA score {sa}/10 > 7)"

    score_components = [("qed", qed_score, 0.30), ("lipinski", max(0.0, 1.0 - lip_violations / 4), 0.25)]
    if sa is not None:
        score_components.append(("sa", max(0.0, (10 - sa) / 9), 0.20))
    score_components.append(("tpsa", max(0.0, 1.0 - tpsa / 180), 0.10))
    if novelty_sim is not None:
        score_components.append(("novelty", max(0.0, 1.0 - novelty_sim), 0.15))

    total_weight  = sum(w for _, _, w in score_components)
    overall_score = round(max(0.0, min(1.0, sum(s * w for _, s, w in score_components) / total_weight)), 3)
    if disqualified:
        overall_score = 0.0

    n_fail = sum(1 for p in parameters.values() if p["verdict"] == "FAIL")
    n_warn = sum(1 for p in parameters.values() if p["verdict"] == "WARN")
    if disqualified or n_fail >= 3:
        overall_verdict = "FAIL"
    elif n_fail >= 1 or n_warn >= 3:
        overall_verdict = "WARN"
    else:
        overall_verdict = "PASS"

    return {
        "smiles":                  smiles,
        "valid":                   True,
        "overall_score":           overall_score,
        "verdict":                 overall_verdict,
        "disqualified":            disqualified,
        "disqualification_reason": disqualification_reason,
        "parameters":              parameters,
        "lipinski": {
            "violations": lip_violations,
            "drug_like":  lip_pass,
            "detail":     lip_detail,
        },
    }


def validate_batch(smiles_list: list, seed_smiles: str = None) -> list:
    """Validate a list of SMILES. Returns sorted by overall_score descending."""
    results = [validate_molecule(s, seed_smiles) for s in smiles_list]
    results.sort(key=lambda r: (0 if r["disqualified"] else 1, r["overall_score"]), reverse=True)
    return results


# ── Backward-compat aliases ───────────────────────────────────────────────────
DrugValidator = validate_molecule   # legacy alias
