from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def evaluate_runtime(
    *,
    sample_size: int,
    returned_candidates: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    elapsed_sec: float,
    out_path: Path = Path("results/benchmarks/runtime_metrics.json"),
) -> dict[str, Any]:
    stages = dict(diagnostics.get("stage_counts", {}) or {})
    timings_ms = dict(diagnostics.get("timings_ms", {}) or {})
    generated = int(stages.get("generated", sample_size))
    valid = int(stages.get("valid", 0))
    docked = int(stages.get("docked", 0))
    ranked = int(stages.get("ranked", 0))
    returned = int(stages.get("returned", len(returned_candidates)))

    docking_time_sec = float(timings_ms.get("dock", 0.0)) / 1000.0
    out: dict[str, Any] = {
        "generated_per_sec": float(generated / max(elapsed_sec, 1e-9)),
        "docking_throughput_per_sec": float(docked / max(docking_time_sec, 1e-9)) if docked else 0.0,
        "end_to_end_latency_sec": float(elapsed_sec),
        "survival_funnel": {
            "generated": generated,
            "valid": valid,
            "docked": docked,
            "ranked": ranked,
            "returned": returned,
        },
        "timings_ms": timings_ms,
        "cpu_utilization": None,
        "gpu_utilization": None,
        "timestamp_epoch": time.time(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
