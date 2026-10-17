from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.metrics import summarize_smiles_properties
from models.generation.reinvent4.config import Reinvent4Config
from models.generation.safe_gpt.config import FilteringConfig
from models.runtime.cura_pipeline import CuraPipeline


@dataclass(frozen=True)
class AblationScenario:
    name: str
    enable_runtime_rl: bool
    enable_docking_reward: bool
    safe_filter_enabled: bool
    strictness: str
    weak_bootstrap_prior: bool = False


DEFAULT_SCENARIOS: tuple[AblationScenario, ...] = (
    AblationScenario("healthy_prior_only", False, True, True, "default"),
    AblationScenario("rl_enabled", True, True, True, "default"),
    AblationScenario("no_docking_reward", False, False, True, "default"),
    AblationScenario("no_safe_filtering", False, True, False, "default"),
    AblationScenario("weak_bootstrap_prior", False, True, True, "default", weak_bootstrap_prior=True),
    AblationScenario("relaxed_filtering", False, True, True, "relaxed"),
    AblationScenario("strict_filtering", False, True, True, "strict"),
)


class AblationRunner:
    def __init__(self, protein_sequence: str, target: str, samples: int = 100):
        self.protein_sequence = protein_sequence
        self.target = target
        self.samples = max(10, int(samples))

    def run(
        self,
        *,
        scenarios: tuple[AblationScenario, ...] = DEFAULT_SCENARIOS,
        out_csv: Path = Path("results/benchmarks/ablation_results.csv"),
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sc in scenarios:
            rows.append(self._run_one(sc))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, out_csv)
        return rows

    def _run_one(self, sc: AblationScenario) -> dict[str, Any]:
        rcfg = Reinvent4Config()
        rcfg.enable_runtime_rl = bool(sc.enable_runtime_rl)
        rcfg.docking_rl.enable_docking_rl = bool(sc.enable_docking_reward)

        fcfg = FilteringConfig()
        if not sc.safe_filter_enabled:
            fcfg.pains_hook_enabled = False
            fcfg.reactive_group_hook_enabled = False
            fcfg.reject_formal_charge = False
        elif sc.strictness == "strict":
            fcfg.min_ring_count = 1
            fcfg.require_aromatic_if_multiring = True
            fcfg.max_ring_count = 8
        elif sc.strictness == "relaxed":
            fcfg.min_ring_count = 0
            fcfg.max_ring_count = 12

        t0 = time.perf_counter()
        pipeline = CuraPipeline(profile="benchmark", reinvent_config=rcfg, filtering_cfg=fcfg)
        ranked = pipeline.run(
            protein_sequence=self.protein_sequence,
            target=self.target,
            num_molecules=self.samples,
            sample_size=max(self.samples * 5, 200),
            seed_smiles=None,
        )
        elapsed = time.perf_counter() - t0
        smiles = [r.get("smiles", "") for r in ranked if r.get("smiles")]
        props = summarize_smiles_properties(smiles)

        stages = dict((pipeline.last_diagnostics or {}).get("stage_counts", {}) or {})
        kd = [float(r.get("kd_nm", r.get("binding_nm"))) for r in ranked if isinstance(r.get("kd_nm", r.get("binding_nm")), (int, float))]
        docking_success = int(stages.get("docking_success", 0))
        docked = int(stages.get("docked", 0))
        return {
            "scenario": sc.name,
            "validity": props.get("validity", 0.0),
            "uniqueness": float(len(set(smiles)) / max(1, len(smiles))),
            "docking_success": float(docking_success / max(1, docked)),
            "returned_candidates": len(ranked),
            "kd_mean_nm": float(sum(kd) / len(kd)) if kd else 0.0,
            "runtime_sec": float(elapsed),
            "diversity": props.get("scaffold_diversity", 0.0),
        }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
