from __future__ import annotations

from typing import Optional


TARGET_ALIASES: dict[str, str] = {
    "egfr": "EGFR",
    "erbb1": "EGFR",
    "epidermal growth factor receptor": "EGFR",
    "cdk2": "CDK2",
    "cyclin-dependent kinase 2": "CDK2",
    "mpro": "Mpro",
    "3clpro": "Mpro",
    "main protease": "Mpro",
    "sars-cov-2 mpro": "Mpro",
    "sars cov 2 main protease": "Mpro",
}


def canonical_target_name(target_name: str) -> Optional[str]:
    if not target_name:
        return None
    key = target_name.strip().lower()
    if key in TARGET_ALIASES:
        return TARGET_ALIASES[key]
    # heuristic contains-matching for flexible pipeline names
    if "egfr" in key or "erbb1" in key:
        return "EGFR"
    if "cdk2" in key:
        return "CDK2"
    if "mpro" in key or "3cl" in key or "main protease" in key:
        return "Mpro"
    return None

