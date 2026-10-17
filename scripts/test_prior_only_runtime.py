from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.runtime.cura_pipeline import CuraPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    gene = "EGFR"
    # EGFR kinase domain snippet is enough for smoke runtime checks.
    protein_sequence = (
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTTRTQEYQLVVK"
        "WMALESILHRRLPQANQINN"
    )

    pipe = CuraPipeline(profile="production")
    ranked = pipe.run(
        protein_sequence=protein_sequence,
        target=gene,
        num_molecules=8,
        sample_size=120,
    )

    di = pipe.last_diagnostics or {}
    stage = di.get("stage_counts", {}) or {}
    avg_kd = None
    avg_dock = None
    if ranked:
        kd_vals = [float(r.get("binding_nm")) for r in ranked if r.get("binding_nm") is not None]
        dock_vals = [float(r.get("vina_score")) for r in ranked if r.get("vina_score") is not None]
        if kd_vals:
            avg_kd = sum(kd_vals) / len(kd_vals)
        if dock_vals:
            avg_dock = sum(dock_vals) / len(dock_vals)

    report = {
        "runtime_mode": "PRIOR_ONLY",
        "generated": int(stage.get("generated", 0)),
        "valid": int(stage.get("druglike_pass", stage.get("valid", 0))),
        "docked": int(stage.get("docked", 0)),
        "ranked": int(stage.get("ranked", 0)),
        "returned": int(stage.get("returned", len(ranked))),
        "avg_kd_nm": avg_kd,
        "avg_docking_score": avg_dock,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

