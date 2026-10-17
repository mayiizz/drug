import logging
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, DataStructs

from models.utils.fingerprint_utils import get_morgan_fingerprint
from models.utils.smiles_validation import validate_smiles_strict
from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.filtering import evaluate_smiles, select_profile

logger = logging.getLogger(__name__)


def _parse_binding(result) -> float:
    """
    Safely extract a float nM value from whatever DeepPurpose returns.
    DeepPurpose predict() can return: float, list, np.ndarray, or pd.Series.
    """
    import numbers
    if result is None:
        raise ValueError("Binding model returned None")
    if isinstance(result, numbers.Number):
        return _normalize_binding_to_nm(float(result))
    try:
        import numpy as np
        if isinstance(result, np.generic):
            return _normalize_binding_to_nm(float(result))
        if isinstance(result, np.ndarray):
            if result.size == 0:
                raise ValueError("Binding model returned empty ndarray")
            return _normalize_binding_to_nm(float(result.flat[0]))
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 0:
                raise ValueError("Binding model returned empty pandas object")
            return _normalize_binding_to_nm(float(result.iloc[0]))
    except ImportError:
        pass
    if hasattr(result, '__iter__'):
        try:
            first = next(iter(result))
        except StopIteration:
            raise ValueError("Binding model returned empty iterable")
        return _normalize_binding_to_nm(float(first))
    return _normalize_binding_to_nm(float(result))


def _normalize_binding_to_nm(raw_value: float) -> float:
    """Normalize DeepPurpose-like outputs to nM.

    If output resembles pAffinity/pKd (e.g. 6), convert using:
      nM = 10 ** (9 - pAffinity)
    """
    x = float(raw_value)
    if 0.0 < x < 20.0:
        return float(10.0 ** (9.0 - x))
    return x


# ── SA Score loader — tried once at import, cached for the whole process ──────
_sascorer = None
_sascorer_loaded = False

def _load_sascorer():
    """
    Try every known location for sascorer and cache the module.
    Returns the sascorer module or None.
    """
    global _sascorer, _sascorer_loaded
    if _sascorer_loaded:
        return _sascorer

    _sascorer_loaded = True

    # Attempt 1: rdkit.Contrib (works in newer RDKit conda installs)
    try:
        from rdkit.Contrib.SA_Score import sascorer as _s
        _sascorer = _s
        logger.info("[SA] Loaded via rdkit.Contrib.SA_Score")
        return _sascorer
    except Exception:
        pass

    # Attempt 2: locate via rdkit package path
    try:
        import rdkit
        rdkit_root = os.path.dirname(rdkit.__file__)
        candidates = [
            os.path.join(rdkit_root, "Contrib", "SA_Score"),
            os.path.join(rdkit_root, "..", "Contrib", "SA_Score"),
        ]
        for path in candidates:
            path = os.path.normpath(path)
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        import sascorer as _s
        _sascorer = _s
        logger.info("[SA] Loaded via rdkit package path")
        return _sascorer
    except Exception:
        pass

    # Attempt 3: conda/mamba share directory
    try:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        share_path = os.path.join(conda_prefix, "share", "RDKit", "Contrib", "SA_Score")
        if os.path.isdir(share_path) and share_path not in sys.path:
            sys.path.insert(0, share_path)
        import sascorer as _s
        _sascorer = _s
        logger.info("[SA] Loaded via CONDA_PREFIX share path")
        return _sascorer
    except Exception:
        pass

    # Attempt 4: search common Windows/Linux install paths
    try:
        import glob
        patterns = [
            r"C:\ProgramData\miniconda3\**\SA_Score",
            r"C:\Users\*\miniconda3\**\SA_Score",
            r"C:\Users\*\anaconda3\**\SA_Score",
            "/opt/conda/**/SA_Score",
            os.path.expanduser("~/miniconda3/**/SA_Score"),
            os.path.expanduser("~/anaconda3/**/SA_Score"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                if os.path.isdir(match) and match not in sys.path:
                    sys.path.insert(0, match)
                    try:
                        import sascorer as _s
                        _sascorer = _s
                        logger.info(f"[SA] Loaded via glob path: {match}")
                        return _sascorer
                    except Exception:
                        continue
    except Exception:
        pass

    logger.warning(
        "[SA] sascorer not found in any location — using complexity heuristic.\n"
        "     To fix: ensure RDKit is installed with Contrib files, e.g.:\n"
        "       conda install -c conda-forge rdkit"
    )
    return None


def _safe_sa_score(mol) -> float:
    """
    Calculate SA Score (1=easy to synthesize, 10=hard).
    Uses real sascorer if available, otherwise a per-molecule complexity heuristic
    that actually varies meaningfully between molecules.
    """
    if mol is None:
        return 5.0

    scorer = _load_sascorer()

    if scorer is not None:
        try:
            score = scorer.calculateScore(mol)
            if score is not None:
                return float(score)
        except Exception as e:
            logger.debug(f"[SA] calculateScore failed: {e}")

    # ── Heuristic fallback — varies meaningfully per molecule ─────────────────
    # Based on: ring complexity, stereocenters, bridgehead atoms,
    # unusual valences, spiro atoms, and heavy atom count.
    try:
        ri = mol.GetRingInfo()
        n_rings        = ri.NumRings()
        n_ring_atoms   = len(set(a for ring in ri.AtomRings() for a in ring))
        n_stereo       = sum(
            1 for a in mol.GetAtoms()
            if a.GetChiralTag().name != "CHI_UNSPECIFIED"
        )
        n_heavy        = mol.GetNumHeavyAtoms()
        n_hetero       = sum(
            1 for a in mol.GetAtoms()
            if a.GetAtomicNum() not in (6, 1)
        )

        # Spiro atoms: in 2+ rings but not fused
        ring_memberships = {}
        for i, ring in enumerate(ri.AtomRings()):
            for atom in ring:
                ring_memberships.setdefault(atom, set()).add(i)
        n_spiro = sum(1 for rings in ring_memberships.values() if len(rings) >= 2)

        # Bridgehead: atoms shared between rings that share a bond
        bond_rings = ri.BondRings()
        bond_ring_map = {}
        for i, ring in enumerate(bond_rings):
            for b in ring:
                bond_ring_map.setdefault(b, set()).add(i)
        shared_bonds = sum(1 for rings in bond_ring_map.values() if len(rings) >= 2)

        score = (
            1.5                         # base
            + n_rings        * 0.35
            + n_stereo       * 0.60
            + n_spiro        * 0.70
            + shared_bonds   * 0.25
            + n_heavy        * 0.018
            - n_ring_atoms   * 0.04     # aromatic rings are easier
            + n_hetero       * 0.05
        )
        return float(min(max(score, 1.0), 9.0))
    except Exception:
        return 5.0


def _morgan_fp(smiles: str):
    return get_morgan_fingerprint(smiles)


def _tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _validate_molecule(smiles: str, seed_smiles: str | None = None) -> dict:
    """
    Returns a full validation dict with PASS/WARN/FAIL per parameter
    and a composite score 0→1.
    """
    mol = validate_smiles_strict(smiles)
    if mol is None:
        return {"verdict": "FAIL", "composite_score": 0.0, "parameters": {}}

    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    rot  = Descriptors.NumRotatableBonds(mol)
    qed  = QED.qed(mol)
    sa   = _safe_sa_score(mol)

    novelty = 0.5
    if seed_smiles:
        seed_fp = _morgan_fp(seed_smiles)
        fp      = _morgan_fp(smiles)
        novelty = 1.0 - _tanimoto(fp, seed_fp)

    def grade(val, ideal_lo, ideal_hi, warn_lo, warn_hi):
        if ideal_lo <= val <= ideal_hi:
            return "PASS"
        if warn_lo <= val <= warn_hi:
            return "WARN"
        return "FAIL"

    params = {
        "mol_weight":  {"value": round(mw,      1), "verdict": grade(mw,   200, 500, 150, 700)},
        "logp":        {"value": round(logp,     2), "verdict": grade(logp, -0.4, 5.0, -1, 7)},
        "tpsa":        {"value": round(tpsa,     1), "verdict": grade(tpsa, 0, 140, 0, 160)},
        "rot_bonds":   {"value": int(rot),           "verdict": "PASS" if rot <= 10 else "WARN" if rot <= 15 else "FAIL"},
        "h_donors":    {"value": int(hbd),           "verdict": "PASS" if hbd <= 5  else "FAIL"},
        "h_acceptors": {"value": int(hba),           "verdict": "PASS" if hba <= 10 else "FAIL"},
        "qed":         {"value": round(qed,      2), "verdict": "PASS" if qed >= 0.5 else "WARN" if qed >= 0.3 else "FAIL"},
        "sa_score":    {"value": round(sa,        1), "verdict": "PASS" if sa <= 4  else "WARN" if sa <= 6 else "FAIL"},
        "novelty":     {"value": round(novelty,   2), "verdict": "PASS" if novelty >= 0.15 else "FAIL"},
    }

    fails = sum(1 for p in params.values() if p["verdict"] == "FAIL")
    warns = sum(1 for p in params.values() if p["verdict"] == "WARN")

    mw_score   = max(0, 1 - abs(mw - 350) / 350)
    lpi_score  = max(0, 1 - abs(logp - 2.5) / 5)
    tpsa_score = max(0, 1 - tpsa / 160)
    composite  = (
        qed        * 0.30 +
        mw_score   * 0.20 +
        lpi_score  * 0.15 +
        tpsa_score * 0.10 +
        max(0, 1 - sa / 10) * 0.15 +
        novelty    * 0.10
    )

    if fails >= 3:
        verdict = "FAIL"
    elif fails >= 1 or warns >= 3:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict":         verdict,
        "composite_score": round(composite, 3),
        "parameters":      params,
    }


class MultiObjectiveRanker:

    def rank(self, smiles_list: list[str],
             seed_smiles: str | None = None,
             protein_sequence: str | None = None,
             binding_model=None,
             vina_scores: dict[str, float] | None = None,
             docking_successes: dict[str, bool] | None = None,
             admet_profiles: dict[str, dict] | None = None,
             dp_prefilter_scores: dict[str, float] | None = None) -> list[dict]:
        """
        Score, validate, and rank a list of SMILES.

        Binding:
          - DeepPurpose provides a neural *pre-screen* surrogate score (log10(Kd)-like; lower is better).
            This is used for shortlisting and traceability, not as final affinity truth.
          - If provided, Vina (`vina_scores`) provides an additional physics-informed
            pose-energy ranking signal (more negative is better).
        """
        # NOTE: We do not run DeepPurpose inside the ranker by default anymore.
        # Pipelines should pass dp_prefilter_scores if they want them included.
        binding_scores_map: dict = dp_prefilter_scores or {}

        results = []
        admet_profiles = admet_profiles or {}
        pharm_cfg = PharmacophoreConfig(enabled=True, strictness="soft", enforce_hard_filters=False, target_name="EGFR")
        pharm_profile = select_profile("EGFR", pharm_cfg)

        def _admet_score_from_profile(profile: dict) -> float:
            """
            Map final_admet_decision() string → numeric priority score.
              Strong candidate → 1.0
              Moderate warning  → 0.5
              Reject            → 0.0
            """
            try:
                from models.admet.admet_engine import final_admet_decision

                decision = final_admet_decision(profile)
            except Exception:
                return 0.0

            if decision.startswith("REJECT"):
                return 0.0
            if decision.startswith("MODERATE"):
                return 0.5
            return 1.0

        n_skipped_rank = 0
        for smi in smiles_list:
            mol = validate_smiles_strict(smi)
            if mol is None:
                n_skipped_rank += 1
                continue

            mw   = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd  = Lipinski.NumHDonors(mol)
            hba  = Lipinski.NumHAcceptors(mol)
            rot  = Descriptors.NumRotatableBonds(mol)
            qed  = QED.qed(mol)
            sa   = _safe_sa_score(mol)

            dp_score = binding_scores_map.get(smi)
            if dp_score is not None:
                try:
                    dp_score = _normalize_binding_to_nm(float(dp_score))
                except Exception:
                    dp_score = None

            vina_score = None
            if vina_scores is not None:
                vina_score = vina_scores.get(smi)

            docking_success = None
            if docking_successes is not None:
                docking_success = docking_successes.get(smi)

            admet_score: float | None = None
            if admet_profiles is not None and smi in admet_profiles:
                try:
                    admet_score = _admet_score_from_profile(admet_profiles[smi])
                except Exception:
                    admet_score = None
            if sa > 5.5:
                # Penalize synthetically difficult molecules in final rank priority.
                admet_score = min(0.25, float(admet_score)) if admet_score is not None else 0.25

            validation = _validate_molecule(smi, seed_smiles)
            drug_score = validation["composite_score"]
            pharm = evaluate_smiles(smi, pharm_profile, pharm_cfg)

            pose_stability_bonus = 0.0
            if docking_success and vina_score is not None:
                pose_stability_bonus = max(0.0, min(0.25, ((-float(vina_score)) - 5.0) / 20.0))

            results.append({
                "smiles":        smi,
                "admet": {
                    "molecular_weight": round(mw,   2),
                    "logp":             round(logp,  2),
                    "tpsa":             round(tpsa,  2),
                    "h_donors":         int(hbd),
                    "h_acceptors":      int(hba),
                    "rotatable_bonds":  int(rot),
                    "drug_likeness":    round(qed,   3),
                    "sa_score":         round(sa,    2),
                },
                "drug_score":    round(drug_score, 3),
                "dp_prefilter_score": None if dp_score is None else round(float(dp_score), 4),
                "binding_score": None if dp_score is None else round(float(dp_score), 4),
                "binding_nm": None if dp_score is None else round(float(dp_score), 4),
                # Vina score is approximate pose-energy ranking; more negative is better.
                "vina_score":    None if vina_score is None else round(float(vina_score), 4),
                "docking_score": None if vina_score is None else round(float(vina_score), 4),
                "pose_energy":   None if vina_score is None else round(float(vina_score), 4),
                "docking_success": docking_success,
                "admet_score":  admet_score,
                "pharmacophore_compatibility": round(float(pharm.score), 4),
                "hinge_binding_score": round(float(pharm.score), 4),
                "pose_stability_bonus": round(float(pose_stability_bonus), 4),
                "validation":    validation,
            })

        if n_skipped_rank:
            logger.info(
                "[FEATURIZER] ranker skipped %d/%d invalid SMILES before scoring",
                n_skipped_rank,
                len(smiles_list),
            )

        if not results:
            return []

        def _sort_key(r: dict) -> tuple:
            # Priority:
            #   1) docking (more negative vina_score is better)
            #   2) pharmacophore compatibility
            #   3) ADMET
            #   4) QED
            #   5) SA
            #   6) DeepPurpose prefilter (tie-break/fallback only)
            v = r.get("vina_score")
            a = r.get("admet_score")
            a_norm = 0.0 if a is None else float(a)
            p = float(r.get("pharmacophore_compatibility", 0.0))
            pose_bonus = float(r.get("pose_stability_bonus", 0.0))
            if v is not None:
                return (
                    0,
                    float(v) - pose_bonus,  # slightly prefer stable low-energy poses
                    -p,
                    -a_norm,
                    -float(r["admet"]["drug_likeness"]),
                    float(r["admet"]["sa_score"]),
                    float(r.get("dp_prefilter_score") if r.get("dp_prefilter_score") is not None else float("inf")),
                )

            # No docking: fall back to DP prefilter score if available (lower better), else neutral.
            dp = r.get("dp_prefilter_score")
            dp_val = float(dp) if dp is not None else float("inf")
            return (
                1,
                -p,
                -a_norm,
                -float(r["admet"]["drug_likeness"]),
                float(r["admet"]["sa_score"]),
                dp_val,
                -float(r["drug_score"]),
            )

        results.sort(key=_sort_key)
        return results