from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem


KINASE_KEYWORDS = (
    "kinase",
    "egfr",
    "erk",
    "jak",
    "braf",
    "raf",
    "alk",
    "cdk",
    "mapk",
    "vegfr",
    "fgfr",
)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _read_smiles_from_csv(path: Path, smiles_col: str, target_col: str | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(smiles_col, "") or ""
            target_text = (row.get(target_col, "") or "") if target_col else ""
            rows.append((smi, target_text))
    return rows


def _read_smiles_from_smi(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            smi = line.strip().split()[0] if line.strip() else ""
            rows.append((smi, ""))
    return rows


def _looks_kinase(target_text: str) -> bool:
    t = (target_text or "").lower()
    return any(k in t for k in KINASE_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare kinase-focused fine-tuning dataset.")
    parser.add_argument("--input", required=True, help="Input file path (.csv or .smi).")
    parser.add_argument("--output", required=True, help="Output .smi path for canonicalized set.")
    parser.add_argument("--smiles-col", default="canonical_smiles", help="CSV column containing SMILES.")
    parser.add_argument("--target-col", default="target_chembl_id", help="CSV target annotation column.")
    parser.add_argument(
        "--kinase-only",
        action="store_true",
        help="Keep rows where target annotations appear kinase-related.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".csv":
        raw_rows = _read_smiles_from_csv(in_path, args.smiles_col, args.target_col)
    else:
        raw_rows = _read_smiles_from_smi(in_path)

    canonical: list[str] = []
    for smi, target_text in raw_rows:
        if args.kinase_only and not _looks_kinase(target_text):
            continue
        c = canonicalize_smiles(smi)
        if c is None:
            continue
        canonical.append(c)

    unique_sorted = sorted(set(canonical))
    with out_path.open("w", encoding="utf-8") as f:
        for smi in unique_sorted:
            f.write(f"{smi}\n")

    print(f"input_rows={len(raw_rows)}")
    print(f"canonical_rows={len(canonical)}")
    print(f"unique_rows={len(unique_sorted)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
