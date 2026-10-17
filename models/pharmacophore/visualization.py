from __future__ import annotations

from rdkit import Chem


def summarize_rule_debug(smiles: str, failed_rules: list[str], matched_rules: list[str]) -> dict:
    return {
        "smiles": smiles,
        "failed_rules": list(failed_rules),
        "matched_rules": list(matched_rules),
    }


def highlight_scaffold_atoms(smiles: str) -> list[int]:
    """
    Lightweight helper for downstream figure/debug tooling.
    Returns aromatic atom indices as a simple scaffold proxy.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]

