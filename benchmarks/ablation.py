from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.docking.config import VinaConfig
from models.generation.reinvent4.config import Reinvent4Config
from models.pharmacophore.config import PharmacophoreConfig


@dataclass(slots=True)
class AblationConfig:
    name: str
    use_safe_gpt_only: bool
    enable_docking_reward: bool
    enable_diversity_memory: bool
    enable_pharmacophore_filter: bool
    enable_deeppurpose_prefilter: bool
    enable_vina_refinement: bool


ABLATIONS: dict[str, AblationConfig] = {
    "A": AblationConfig("SAFE-GPT only", True, False, False, False, False, False),
    "B": AblationConfig("REINVENT + DeepPurpose reward only", False, False, True, True, True, False),
    "C": AblationConfig("REINVENT + Vina reward (no diversity)", False, True, False, True, True, True),
    "D": AblationConfig("Full CURA pipeline", False, True, True, True, True, True),
}


def build_toggled_configs(tag: str) -> dict[str, Any]:
    ab = ABLATIONS[tag]
    r = Reinvent4Config()
    d = VinaConfig()
    p = PharmacophoreConfig()
    r.docking_rl.enable_docking_rl = ab.enable_docking_reward
    if not ab.enable_diversity_memory:
        r.diversity.memory_buffer_size = 1
        r.diversity.rejection_tanimoto_threshold = 1.0
    p.enabled = ab.enable_pharmacophore_filter
    d.prefilter_enabled = ab.enable_deeppurpose_prefilter
    if not ab.enable_vina_refinement:
        d.mode = "deeppurpose_only"
    return {"ablation": ab, "reinvent_cfg": r, "vina_cfg": d, "pharmacophore_cfg": p}


def run_ablation_matrix(
    runner,
    *,
    seeds: list[int] | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute ablations using an injected callable:
      runner(tag, seed, toggled_configs) -> metrics dict
    """
    from benchmarks.utils import set_global_seed

    seeds = seeds or [13, 37, 71]
    tags = tags or ["A", "B", "C", "D"]
    all_rows: list[dict[str, Any]] = []
    for tag in tags:
        cfgs = build_toggled_configs(tag)
        for seed in seeds:
            set_global_seed(seed)
            metrics = runner(tag, seed, cfgs)
            row = {"config_tag": tag, "seed": seed}
            row.update(metrics or {})
            all_rows.append(row)
    return all_rows

