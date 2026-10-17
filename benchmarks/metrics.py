from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from benchmarks.utils import canonicalize_smiles
from models.generation.reinvent4.utils import safe_sa

_SMILES_COLUMNS = ("smiles", "canonical_smiles", "SMILES", "molecule_smiles", "standard_smiles")


def locate_chembl_corpus(root: Path) -> Path | None:
    search_roots = [
        root / "data",
        root / "results",
        root / "benchmarks",
        root / "scripts",
    ]
    candidates: list[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob("**/chembl*.csv")))
        candidates.extend(sorted(base.glob("**/chembl*.smi")))
        candidates.extend(sorted(base.glob("**/chembl*.txt")))
    for path in candidates:
        text = str(path).lower()
        if "node_modules" in text or ".next" in text or "__pycache__" in text:
            continue
        return path
    return None


def load_corpus_smiles(path: Path, *, max_rows: int = 2000) -> list[str]:
    suffix = path.suffix.lower()
    smiles: list[str] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            col = _detect_smiles_col(fieldnames)
            for row in reader:
                s = str(row.get(col, "")).strip() if col else ""
                if s:
                    smiles.append(s)
                if len(smiles) >= max_rows:
                    break
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    smiles.append(s)
                if len(smiles) >= max_rows:
                    break
    out: list[str] = []
    seen: set[str] = set()
    for smi in smiles:
        c = canonicalize_smiles(smi)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _detect_smiles_col(columns: list[str]) -> str | None:
    low_map = {c.lower(): c for c in columns}
    for col in _SMILES_COLUMNS:
        if col.lower() in low_map:
            return low_map[col.lower()]
    return columns[0] if columns else None


def aggregate_distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "top10_mean": 0.0}
    vals = sorted(float(v) for v in values)
    n = len(vals)
    k = max(1, int(round(n * 0.1)))
    top = vals[-k:]
    return {
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "std": float(statistics.pstdev(vals)) if n > 1 else 0.0,
        "top10_mean": float(statistics.fmean(top)),
    }


def summarize_smiles_properties(smiles_list: list[str]) -> dict[str, Any]:
    mw_vals: list[float] = []
    logp_vals: list[float] = []
    qed_vals: list[float] = []
    sa_vals: list[float] = []
    scaffolds: set[str] = set()
    valid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid += 1
        mw_vals.append(float(Descriptors.MolWt(mol)))
        logp_vals.append(float(Crippen.MolLogP(mol)))
        qed_vals.append(float(QED.qed(mol)))
        sa_vals.append(float(safe_sa(mol)))
        try:
            scf = MurckoScaffold.GetScaffoldForMol(mol)
            scf_smi = Chem.MolToSmiles(scf) if scf is not None else ""
            if scf_smi:
                scaffolds.add(scf_smi)
        except Exception:
            pass
    return {
        "n_total": len(smiles_list),
        "n_valid": valid,
        "validity": float(valid / max(1, len(smiles_list))),
        "avg_qed": float(statistics.fmean(qed_vals)) if qed_vals else 0.0,
        "avg_sa": float(statistics.fmean(sa_vals)) if sa_vals else 0.0,
        "mw_stats": aggregate_distribution_stats(mw_vals),
        "logp_stats": aggregate_distribution_stats(logp_vals),
        "qed_stats": aggregate_distribution_stats(qed_vals),
        "sa_stats": aggregate_distribution_stats(sa_vals),
        "scaffold_count": len(scaffolds),
        "scaffold_diversity": float(len(scaffolds) / max(1, valid)),
    }
