"""
Tox21 multi-task toxicity (lazy-loaded, one-time init).

Loads the official Tox21 CSV (same source as DeepChem molnet), featurizes with
DeepChem CircularFingerprint, and fits a scikit-learn MultiOutputClassifier.
DeepChem Keras models (GraphConv / MultitaskClassifier) are skipped because
deepchem 2.5 + current TensorFlow/Keras often break at runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from models.utils.smiles_validation import validate_smiles_strict

logger = logging.getLogger(__name__)

TOX21_TASKS = [
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

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

FP_SIZE = 2048

_bundle: dict[str, Any] | None = None


def _row_y(row, tasks: list[str]) -> np.ndarray:
    import pandas as pd

    out: list[float] = []
    for t in tasks:
        v = row[t]
        if pd.isna(v):
            out.append(0.0)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
    return np.asarray(out, dtype=np.float32)


def _ensure_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle

    import deepchem as dc
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    data_dir = dc.utils.data_utils.get_data_dir()
    path = os.path.join(data_dir, "tox21.csv.gz")
    if not os.path.exists(path):
        dc.utils.data_utils.download_url(url=TOX21_URL, dest_dir=data_dir)

    df = pd.read_csv(path, compression="gzip")
    featurizer = dc.feat.CircularFingerprint(size=FP_SIZE, radius=2)

    X_rows: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        try:
            feat = featurizer.featurize([smi])[0]
        except Exception:
            continue
        if feat is None:
            continue
        fv = np.asarray(feat, dtype=np.float32).ravel()
        if fv.size != FP_SIZE:
            continue
        X_rows.append(fv)
        y_list.append(_row_y(row, TOX21_TASKS))

    if not X_rows:
        raise RuntimeError("Tox21 featurization produced no valid molecules.")

    X = np.vstack(X_rows)
    y = np.vstack(y_list)

    base = RandomForestClassifier(
        n_estimators=64,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base)
    model.fit(X, y)

    _bundle = {
        "tasks": TOX21_TASKS,
        "model": model,
        "featurizer": featurizer,
    }
    logger.info("[Tox21] sklearn MultiOutputClassifier ready (n=%s).", len(X_rows))
    return _bundle


def predict_toxicity(smiles: str) -> dict[str, Any]:
    try:
        if validate_smiles_strict(smiles) is None:
            return {"status": "error", "reason": "Invalid or unfeaturizable SMILES"}

        bundle = _ensure_bundle()
        model = bundle["model"]
        featurizer = bundle["featurizer"]
        tasks: list[str] = bundle["tasks"]

        features = featurizer.featurize([smiles])
        if features is None or len(features) < 1:
            return {"status": "error", "reason": "Invalid or unfeaturizable SMILES"}
        f0 = features[0]
        if f0 is None:
            return {"status": "error", "reason": "Invalid or unfeaturizable SMILES"}

        fv = np.asarray(f0, dtype=np.float32).reshape(1, -1)
        if fv.shape[1] != FP_SIZE:
            return {
                "status": "error",
                "reason": f"Fingerprint dim {fv.shape[1]} != {FP_SIZE}",
            }

        probs_list = model.predict_proba(fv)
        result: dict[str, float] = {}
        for i, task in enumerate(tasks):
            arr = np.asarray(probs_list[i])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                result[task] = float(arr[0, 1])
            elif arr.ndim == 2 and arr.shape[1] == 1:
                result[task] = float(arr[0, 0])
            else:
                result[task] = float(np.ravel(arr)[0])

        return {"status": "success", "tox21": result}
    except Exception as e:
        logger.warning("predict_toxicity failed: %s", e)
        return {"status": "error", "reason": str(e)}
