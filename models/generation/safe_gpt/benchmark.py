from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED

from models.generation.safe_gpt.scaffold import murcko_scaffold_smiles
from models.generation.safe_gpt.utils import safe_sa


def benchmark_stats(candidates: list[str], elapsed_s: float = 0.0) -> dict[str, float]:
    total = len(candidates)
    if total == 0:
        return {
            "count": 0.0,
            "valid_pct": 0.0,
            "unique_pct": 0.0,
            "scaffold_diversity": 0.0,
            "mean_qed": 0.0,
            "mean_sa": 0.0,
            "throughput_smiles_per_s": 0.0,
        }
    mols = [Chem.MolFromSmiles(s) for s in candidates]
    valid = [m for m in mols if m is not None]
    unique = len(set(candidates))
    scafs = {murcko_scaffold_smiles(s) for s in candidates if s}
    scafs.discard("")
    mean_qed = sum(float(QED.qed(m)) for m in valid) / max(1, len(valid))
    mean_sa = sum(safe_sa(m) for m in valid) / max(1, len(valid))
    return {
        "count": float(total),
        "valid_pct": 100.0 * len(valid) / total,
        "unique_pct": 100.0 * unique / total,
        "scaffold_diversity": len(scafs) / max(1, total),
        "mean_qed": mean_qed,
        "mean_sa": mean_sa,
        "throughput_smiles_per_s": (total / elapsed_s) if elapsed_s > 0 else 0.0,
    }

