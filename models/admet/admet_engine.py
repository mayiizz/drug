import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from models.admet.local_admet_heuristics import LocalADMETHeuristics
from models.toxicity.toxicity_engine import predict_toxicity
from models.utils.smiles_validation import validate_smiles_strict

local_heuristics_engine = LocalADMETHeuristics()


# -------------------------------
# 1. Lipinski Rule (Drug-likeness) + Veber (oral bioavailability)
# -------------------------------


def veber_rule(mol):
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    rot = Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]
    return tpsa <= 140 and rot <= 10


def lipinski(smiles: str):
    mol = validate_smiles_strict(smiles)
    if mol is None:
        return {"valid": False, "error": "Invalid SMILES", "Veber_Pass": False}

    return {
        "valid": True,
        "MW": Descriptors.MolWt(mol),  # type: ignore[attr-defined]
        "LogP": Crippen.MolLogP(mol),  # type: ignore[attr-defined]
        "HBD": Lipinski.NumHDonors(mol),  # type: ignore[attr-defined]
        "HBA": Lipinski.NumHAcceptors(mol),  # type: ignore[attr-defined]
        "Lipinski_Pass": (
            Descriptors.MolWt(mol) <= 500 and  # type: ignore[attr-defined]
            Crippen.MolLogP(mol) <= 5 and  # type: ignore[attr-defined]
            Lipinski.NumHDonors(mol) <= 5 and  # type: ignore[attr-defined]
            Lipinski.NumHAcceptors(mol) <= 10  # type: ignore[attr-defined]
        ),
        "Veber_Pass": veber_rule(mol),
    }


# -------------------------------
# 2. ADMETlab API
# -------------------------------
def admetlab(smiles: str):
    try:
        url = "https://admetlab3.scbdd.com/api/prediction"

        response = requests.post(
            url,
            json={"smiles": smiles},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=10,
        )

        if response.status_code != 200:
            return {
                "status": "failed",
                "data": None,
                "reason": f"HTTP {response.status_code}",
            }

        data = response.json()

        if not data:
            return {
                "status": "failed",
                "data": None,
                "reason": "Empty response",
            }

        return {
            "status": "success",
            "data": data,
        }

    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "reason": str(e),
        }


def get_admetlab_data(profile: dict):
    """ADMETlab payload stored on profile as raw `data` or None (see get_admet_profile)."""
    return profile.get("admetlab")


def get_admet_profile(smiles: str):
    try:
        lip = lipinski(smiles)
    except Exception as e:
        lip = {"valid": False, "error": str(e)}

    admet_api = admetlab(smiles)
    tox = predict_toxicity(smiles)
    local_pred = local_heuristics_engine.predict(smiles)

    # Separate local descriptor heuristics from external predictive sources
    # for scientific clarity in reporting.
    external_predictive_sources = {
        "admetlab": admet_api.get("data") if admet_api.get("status") == "success" else None,
        "tox21": tox,
    }
    local_heuristics = local_pred

    return {
        "lipinski": lip,
        "local_heuristics": local_heuristics,
        "external_predictive_sources": external_predictive_sources,
        # Compatibility keys retained for existing callers.
        "admetlab": external_predictive_sources["admetlab"],
        "tox21": external_predictive_sources["tox21"],
        "local_admet": local_heuristics,
    }


def _float_gt(v, thresh: float = 0.5) -> bool:
    if v is None:
        return False
    try:
        return float(v) > thresh
    except (TypeError, ValueError):
        return False


def _float_lt(v, thresh: float) -> bool:
    if v is None:
        return False
    try:
        return float(v) < thresh
    except (TypeError, ValueError):
        return False


def _tox21_any_endpoint_high(tox21_block: dict, thresh: float = 0.5) -> bool:
    """Use numeric Tox21 task probs only (predict_toxicity returns status + tox21 dict)."""
    if not isinstance(tox21_block, dict):
        return False
    inner = tox21_block.get("tox21")
    if isinstance(inner, dict):
        for v in inner.values():
            if _float_gt(v, thresh):
                return True
        return False
    for k, v in tox21_block.items():
        if k in ("status", "reason"):
            continue
        if _float_gt(v, thresh):
            return True
    return False


def final_admet_decision(profile: dict) -> str:
    lip = profile.get("lipinski", {})
    ext = profile.get("external_predictive_sources", {}) or {}
    tox21 = ext.get("tox21", profile.get("tox21", {}))
    admet = ext.get("admetlab", profile.get("admetlab") or {})
    local_h = profile.get("local_heuristics", profile.get("local_admet") or {}) or {}

    # Lipinski
    if not lip.get("Lipinski_Pass"):
        return "REJECT (descriptor rules: Lipinski)"

    # Tox21 (numeric task probs only — not raw any(v > 0.5 on status strings))
    if isinstance(tox21, dict) and _tox21_any_endpoint_high(tox21):
        return "REJECT (external predictive source: Tox21 risk)"

    if bool(local_h.get("reactive_alert")):
        return "REJECT (local medicinal heuristic: reactive functional group)"
    if bool(local_h.get("lipophilicity_risk")):
        return "MODERATE (local medicinal heuristic: excessive lipophilicity/aromatic hydrophobicity)"

    if admet:
        # Cardiotoxicity
        if admet.get("hERG") and _float_gt(admet["hERG"], 0.5):
            return "REJECT (external predictive source: hERG risk)"

        # Liver toxicity
        if admet.get("Hepatotoxicity") and _float_gt(admet["Hepatotoxicity"], 0.5):
            return "REJECT (external predictive source: hepatotoxicity risk)"

        # BBB (optional)
        if admet.get("BBB") and _float_lt(admet["BBB"], 0.3):
            return "MODERATE (external predictive source: low BBB penetration)"

        # CYP metabolism
        if admet.get("CYP3A4_inhibition") and _float_gt(admet["CYP3A4_inhibition"], 0.7):
            return "MODERATE (external predictive source: metabolism risk)"

    return "STRONG CANDIDATE (no high-risk alerts from current filters)"
