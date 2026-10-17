from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkTargetConfig:
    name: str
    aliases: tuple[str, ...]
    chembl_target_id: str
    recommended_protein: str
    activity_threshold_nm: float


TARGETS: dict[str, BenchmarkTargetConfig] = {
    "egfr": BenchmarkTargetConfig(
        name="EGFR",
        aliases=("egfr", "erbb1", "epidermal growth factor receptor"),
        chembl_target_id="CHEMBL203",
        recommended_protein="1M17",
        activity_threshold_nm=1000.0,
    ),
    "cdk2": BenchmarkTargetConfig(
        name="CDK2",
        aliases=("cdk2", "cyclin dependent kinase 2"),
        chembl_target_id="CHEMBL301",
        recommended_protein="1HCK",
        activity_threshold_nm=1000.0,
    ),
    "mpro": BenchmarkTargetConfig(
        name="SARS-CoV-2 Mpro",
        aliases=("mpro", "main protease", "3clpro", "sars-cov-2 mpro"),
        chembl_target_id="CHEMBL4523582",
        recommended_protein="7BQY",
        activity_threshold_nm=5000.0,
    ),
}


def resolve_target(name_or_alias: str) -> BenchmarkTargetConfig:
    q = (name_or_alias or "").strip().lower()
    for cfg in TARGETS.values():
        if q == cfg.name.lower() or q in cfg.aliases:
            return cfg
    raise KeyError(f"Unknown benchmark target: {name_or_alias}")

