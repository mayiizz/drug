from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generation.reinvent4.vocab import SMILESVocabulary

logger = logging.getLogger("evaluate_dataset")

_SMILES_COLUMNS = ("smiles", "canonical_smiles", "SMILES", "molecule_smiles", "standard_smiles")
_ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def _read_smiles_from_csv(path: Path) -> tuple[list[str], str]:
    import pandas as pd

    df = pd.read_csv(path)
    col = next((c for c in _SMILES_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No SMILES-like column found in {path}")
    return [str(x) for x in df[col].dropna().tolist()], col


def _canonicalize_and_filter(
    smiles: str,
    chooser: rdMolStandardize.LargestFragmentChooser,
) -> tuple[str | None, dict[str, int]]:
    stats = {
        "invalid_parse": 0,
        "invalid_sanitize": 0,
        "organometallic": 0,
        "wildcard_polymer": 0,
        "radical_reject": 0,
        "salts_stripped": 0,
    }
    raw = (smiles or "").strip()
    if not raw:
        stats["invalid_parse"] += 1
        return None, stats

    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        stats["invalid_parse"] += 1
        return None, stats

    frag = chooser.choose(mol)
    if frag is None:
        stats["invalid_parse"] += 1
        return None, stats
    if frag.GetNumAtoms() < mol.GetNumAtoms():
        stats["salts_stripped"] += 1
    mol = frag

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        stats["invalid_sanitize"] += 1
        return None, stats

    if "*" in raw:
        stats["wildcard_polymer"] += 1
        return None, stats

    for atom in mol.GetAtoms():
        anum = int(atom.GetAtomicNum())
        if anum not in _ALLOWED_ATOMIC_NUMS:
            stats["organometallic"] += 1
            return None, stats
        if atom.GetNumRadicalElectrons() > 0:
            stats["radical_reject"] += 1
            return None, stats

    can = Chem.MolToSmiles(mol, canonical=True)
    if not can:
        stats["invalid_parse"] += 1
        return None, stats
    return can, stats


def evaluate_dataset(csv_path: Path, out_path: Path) -> dict[str, Any]:
    vocab = SMILESVocabulary()
    chooser = rdMolStandardize.LargestFragmentChooser()
    raw_smiles, smiles_col = _read_smiles_from_csv(csv_path)
    logger.info("[DATASET] path=%s", csv_path)
    logger.info("[DATASET] total_rows=%d", len(raw_smiles))
    logger.info("[DATASET] smiles_column=%s", smiles_col)

    canonical_smiles: list[str] = []
    counters = Counter()
    unsupported_tokens = Counter()
    total_tokens = 0
    unsupported_token_count = 0
    mw_values: list[float] = []
    logp_values: list[float] = []

    for s in raw_smiles:
        can, stat = _canonicalize_and_filter(s, chooser)
        counters.update(stat)
        if can is None:
            continue

        toks = vocab.tokenize(can)
        total_tokens += len(toks)
        unk_here = 0
        for tok in toks:
            if tok not in vocab.tok2idx:
                unsupported_tokens[tok] += 1
                unk_here += 1
        unsupported_token_count += unk_here
        if unk_here > 0:
            continue

        mol = Chem.MolFromSmiles(can)
        if mol is None:
            counters["invalid_parse"] += 1
            continue
        canonical_smiles.append(can)
        mw_values.append(float(Descriptors.MolWt(mol)))
        logp_values.append(float(Descriptors.MolLogP(mol)))

    unique_smiles = sorted(set(canonical_smiles))
    total = len(raw_smiles)
    valid = len(canonical_smiles)
    unique = len(unique_smiles)
    report = {
        "dataset_path": str(csv_path),
        "detected_smiles_column": smiles_col,
        "total_molecules": total,
        "valid_molecules": valid,
        "invalid_fraction": float((total - valid) / max(1, total)),
        "unique_molecules": unique,
        "unique_fraction": float(unique / max(1, valid)),
        "avg_mw": float(sum(mw_values) / max(1, len(mw_values))),
        "avg_logp": float(sum(logp_values) / max(1, len(logp_values))),
        "token_coverage": {
            "total_tokens": int(total_tokens),
            "unsupported_token_count": int(unsupported_token_count),
            "unsupported_token_fraction": float(unsupported_token_count / max(1, total_tokens)),
            "top_unsupported_tokens": unsupported_tokens.most_common(10),
        },
        "filter_stats": dict(counters),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ChEMBL dataset quality for prior training")
    p.add_argument("--csv", type=Path, default=Path("data/chembl_100k.csv"))
    p.add_argument("--out", type=Path, default=Path("results/dataset_report.json"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = _parse_args()
    report = evaluate_dataset(args.csv, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

