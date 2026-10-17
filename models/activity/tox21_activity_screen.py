"""
Tox21 multi-task activity / assay-hit style screen (12 endpoints).

Loads the official Tox21 CSV directly via pandas to bypass the DeepChem
``molnet.load_tox21`` shard loader, which raises
``setting an array element with a sequence ... inhomogeneous shape`` on
recent NumPy versions when one of the SMILES rows fails to featurize. Every
row is validated with :func:`models.utils.smiles_validation.validate_smiles_strict`
before fingerprinting so the X / y matrices stay aligned and no ``None``
ever reaches RDKit's canonicalization path.

Feature matrix X is built with RDKit
``GetMorganGenerator(radius=2, fpSize=1024)`` so training and inference
match (``models.utils.fingerprint_utils.mols_to_morgan_fingerprint_matrix``).

One RandomForestClassifier per task, trained only on samples with a non-NaN
label for that task. Predictions are P(positive class) per assay.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from models.utils.fingerprint_utils import mols_to_morgan_fingerprint_matrix
from models.utils.smiles_validation import validate_smiles_strict

logger = logging.getLogger(__name__)

FP_SIZE = 1024

# 12 Tox21 endpoints in the same column order as the official CSV.
TOX21_TASKS: list[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

TOX21_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

_bundle: dict[str, Any] | None = None


def _load_tox21_csv() -> "tuple[list[str], np.ndarray]":
    """Read tox21.csv.gz directly into (ids, y_matrix).

    Bypasses ``dc.molnet.load_tox21`` because its shard featurizer raises
    ``setting an array element with a sequence ... inhomogeneous shape``
    on recent numpy versions. We use DeepChem only for the download URL;
    the actual parsing is plain pandas.
    """
    import pandas as pd

    try:
        import deepchem as dc

        data_dir = dc.utils.data_utils.get_data_dir()
    except Exception:
        data_dir = os.path.expanduser("~/.deepchem/data")
    os.makedirs(data_dir, exist_ok=True)

    path = os.path.join(data_dir, "tox21.csv.gz")
    if not os.path.exists(path):
        try:
            import deepchem as dc

            dc.utils.data_utils.download_url(url=TOX21_CSV_URL, dest_dir=data_dir)
        except Exception:
            # Fallback to urllib if DeepChem isn't importable.
            import urllib.request

            urllib.request.urlretrieve(TOX21_CSV_URL, path)

    df = pd.read_csv(path, compression="gzip")

    ids: list[str] = []
    y_rows: list[list[float]] = []
    smi_col = "smiles" if "smiles" in df.columns else df.columns[-1]
    for _, row in df.iterrows():
        smi = row.get(smi_col)
        if not isinstance(smi, str):
            continue
        ids.append(smi)
        out: list[float] = []
        for t in TOX21_TASKS:
            v = row[t] if t in row.index else float("nan")
            if pd.isna(v):
                out.append(float("nan"))
            else:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(float("nan"))
        y_rows.append(out)

    y = np.asarray(y_rows, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return ids, y


def _ensure_model() -> dict[str, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle

    from sklearn.ensemble import RandomForestClassifier

    try:
        ids, y_full = _load_tox21_csv()
    except Exception as e:
        logger.warning(
            "[Tox21 screen] CSV load failed (%s); screen will be unavailable.",
            e,
        )
        _bundle = {"tasks": list(TOX21_TASKS), "clfs": [None] * len(TOX21_TASKS)}
        return _bundle

    if y_full.shape[0] != len(ids):
        logger.warning(
            "[Tox21 train] y rows (%d) and ids (%d) length mismatch; truncating to min.",
            y_full.shape[0],
            len(ids),
        )
        n_min = min(y_full.shape[0], len(ids))
        y_full = y_full[:n_min]
        ids = ids[:n_min]

    # Strict validation prevents ``CanonicalRankAtoms(NoneType)`` flooding and
    # keeps X / y aligned by index. This is what makes the inhomogeneous
    # shape error impossible regardless of upstream loader changes.
    valid_idx: list[int] = []
    mols_train: list[Any] = []
    for i, s in enumerate(ids):
        m = validate_smiles_strict(s)
        if m is not None:
            valid_idx.append(i)
            mols_train.append(m)

    n_total = len(ids)
    n_valid = len(valid_idx)
    if n_valid != n_total:
        logger.info(
            "[Tox21 train] kept %d / %d valid SMILES (%d rejected at sanitize).",
            n_valid,
            n_total,
            n_total - n_valid,
        )

    if not mols_train:
        logger.warning("[Tox21 train] No valid SMILES after strict validation; aborting model build.")
        _bundle = {"tasks": list(TOX21_TASKS), "clfs": [None] * len(TOX21_TASKS)}
        return _bundle

    y = y_full[valid_idx]
    X = mols_to_morgan_fingerprint_matrix(mols_train, fp_size=FP_SIZE)

    n_tasks = len(TOX21_TASKS)
    if X.shape[1] != FP_SIZE:
        logger.warning("Tox21 X second dim is %s, expected %s", X.shape[1], FP_SIZE)
    if X.shape[0] != y.shape[0]:
        logger.warning(
            "[Tox21 train] X/y row mismatch (X=%d, y=%d); truncating to min.",
            X.shape[0],
            y.shape[0],
        )
        n_min = min(X.shape[0], y.shape[0])
        X = X[:n_min]
        y = y[:n_min]

    clfs: list[RandomForestClassifier | None] = []
    for k in range(n_tasks):
        yk = y[:, k]
        mask = np.isfinite(yk) & ~np.isnan(yk)
        yk_clean = yk[mask].astype(int)
        if mask.sum() < 20 or len(np.unique(yk_clean)) < 2:
            clfs.append(None)
            continue
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X[mask], yk_clean)
        clfs.append(clf)

    n_trained = sum(1 for c in clfs if c is not None)
    logger.info(
        "[Tox21 screen] %s/%s task models trained (%s samples).",
        n_trained,
        n_tasks,
        len(X),
    )

    _bundle = {
        "tasks": list(TOX21_TASKS),
        "clfs": clfs,
    }
    return _bundle


def screen_generated_smiles(
    generated_smiles: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Returns:
        results: [{"smiles", "tox21": {task: prob or None}}, ...]
        active_molecules: rows where any finite tox21 value > threshold
        activity_score: max finite tox21 prob per row (convenience for sorting/API)
    """
    bundle = _ensure_model()
    tasks: list[str] = bundle["tasks"]
    clfs: list = bundle["clfs"]

    valid_smiles: list[str] = []
    mols = []
    for sm in generated_smiles:
        m = validate_smiles_strict(sm)
        if m is not None:
            valid_smiles.append(sm)
            mols.append(m)

    n_skipped = len(generated_smiles) - len(valid_smiles)
    if n_skipped:
        logger.info(
            "[FEATURIZER] Tox21 skipped %d/%d invalid SMILES before featurization",
            n_skipped,
            len(generated_smiles),
        )

    if not mols:
        return {
            "results": [],
            "active_molecules": [],
            "threshold": threshold,
            "n_input": len(generated_smiles),
            "n_valid": 0,
        }

    Xq = mols_to_morgan_fingerprint_matrix(mols, fp_size=FP_SIZE)

    results: list[dict[str, Any]] = []
    for i, sm in enumerate(valid_smiles):
        tox: dict[str, float | None] = {}
        for k, task in enumerate(tasks):
            clf = clfs[k] if k < len(clfs) else None
            if clf is None:
                tox[task] = None
                continue
            p = clf.predict_proba(Xq[i : i + 1])
            if p.shape[1] >= 2:
                tox[task] = float(p[0, 1])
            else:
                tox[task] = float(p[0, 0])

        finite = [v for v in tox.values() if v is not None and np.isfinite(v)]
        max_p = max(finite) if finite else None

        row = {"smiles": sm, "tox21": tox, "activity_score": max_p}
        results.append(row)

    def _any_high(r: dict) -> bool:
        for v in r["tox21"].values():
            if v is not None and float(v) > threshold:
                return True
        return False

    active_molecules = [r for r in results if _any_high(r)]

    return {
        "results": results,
        "active_molecules": active_molecules,
        "threshold": threshold,
        "n_input": len(generated_smiles),
        "n_valid": len(valid_smiles),
    }
