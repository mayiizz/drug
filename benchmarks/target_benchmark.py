from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.known_inhibitors import nearest_known_inhibitor
from benchmarks.metrics import aggregate_distribution_stats
from models.pharmacophore.egfr_pharmacophore import score_egfr_pharmacophore


def _extract_numeric(items: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in items:
        v = row.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def evaluate_target(
    candidates: list[dict[str, Any]],
    *,
    target: str,
    docking_hit_threshold: float = -6.0,
    kd_hit_threshold_nm: float = 1000.0,
    out_path: Path = Path("results/benchmarks/target_metrics.json"),
) -> dict[str, Any]:
    dock_vals = _extract_numeric(candidates, "docking_score")
    kd_vals = _extract_numeric(candidates, "kd_nm")
    if not kd_vals:
        kd_vals = _extract_numeric(candidates, "binding_nm")

    top_docking = sorted(
        [c for c in candidates if isinstance(c.get("docking_score"), (int, float))],
        key=lambda x: float(x["docking_score"]),
    )[:10]

    nearest = []
    scaffold_overlaps = 0
    pharm_hits = 0
    for row in candidates:
        smi = str(row.get("smiles", "")).strip()
        if not smi:
            continue
        info = nearest_known_inhibitor(smi, target=target)
        nearest.append(info)
        if float(info.get("scaffold_similarity", 0.0)) > 0.5:
            scaffold_overlaps += 1
        if (target or "").upper() == "EGFR":
            try:
                p = score_egfr_pharmacophore(smi)
                if float(p.get("pharmacophore_score", 0.0)) > 0:
                    pharm_hits += 1
            except Exception:
                continue

    similarity_vals = [float(n.get("tanimoto", 0.0)) for n in nearest]
    scaffold_similarity_vals = [float(n.get("scaffold_similarity", 0.0)) for n in nearest]

    out: dict[str, Any] = {
        "target": target,
        "n_candidates": len(candidates),
        "docking_score_distribution": aggregate_distribution_stats(dock_vals),
        "kd_distribution_nm": aggregate_distribution_stats(kd_vals),
        "top10_docking_hits": [
            {
                "smiles": c.get("smiles", ""),
                "docking_score": c.get("docking_score"),
                "kd_nm": c.get("kd_nm", c.get("binding_nm")),
                "score": c.get("score"),
            }
            for c in top_docking
        ],
        "docking_hit_rate": float(sum(1 for d in dock_vals if d <= docking_hit_threshold) / max(1, len(dock_vals))),
        "kd_hit_rate": float(sum(1 for k in kd_vals if k <= kd_hit_threshold_nm) / max(1, len(kd_vals))),
        "nearest_known_similarity": aggregate_distribution_stats(similarity_vals),
        "nearest_known_scaffold_similarity": aggregate_distribution_stats(scaffold_similarity_vals),
        "scaffold_overlap_rate": float(scaffold_overlaps / max(1, len(nearest))),
        "pharmacophore_overlap_rate": float(pharm_hits / max(1, len(candidates))),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
