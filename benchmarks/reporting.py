from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.statistics import summarize
from benchmarks.utils import ensure_dir, stable_hash, stable_json_dumps


def write_metrics_csv(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return p


def write_summary_table(seed_metrics: dict[str, list[float]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, vals in seed_metrics.items():
        s = summarize(vals)
        out[k] = f"{s.mean:.4f} ± {s.std:.4f}"
    return out


def write_run_ledger(
    out_dir: str | Path,
    *,
    config_snapshot: dict[str, Any],
    seed: int,
    checkpoint_meta: dict[str, Any] | None,
    metrics: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> Path:
    d = ensure_dir(out_dir)
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "seed": seed,
        "config_snapshot": config_snapshot,
        "checkpoint_meta": checkpoint_meta or {},
        "metrics": metrics,
        "dataset_manifest": dataset_manifest,
    }
    run_id = stable_hash(payload)[:16]
    p = d / f"run_{run_id}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p

