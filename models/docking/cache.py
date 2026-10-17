from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models.docking.config import VinaConfig


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def make_docking_cache_key(
    protein_id: str,
    smiles: str,
    box: Dict[str, Any],
    cfg: VinaConfig,
) -> str:
    """
    Deterministic hash for (protein, ligand, docking box, config).

    Used for cache filenames across runs.
    """
    payload = {
        "protein_id": protein_id,
        "smiles": smiles,
        "box": box,
        "cfg": asdict(cfg),
    }
    digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest


def make_ligand_prep_cache_key(smiles: str, cfg: VinaConfig) -> str:
    """
    Deterministic cache key for ligand PDBQT prep.
    Includes VinaConfig so prep results stay consistent across config changes.
    """
    payload = {"type": "ligand_prep", "smiles": smiles, "cfg": asdict(cfg)}
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def make_receptor_prep_cache_key(protein_id: str, cfg: VinaConfig) -> str:
    """
    Deterministic cache key for receptor PDBQT prep.
    Includes VinaConfig so prep results stay consistent across config changes.
    """
    payload = {"type": "receptor_prep", "protein_id": protein_id, "cfg": asdict(cfg)}
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


class DockingCache:
    """
    File-based cache for:
      - receptor PDBQT
      - ligand PDBQT
      - docking result JSON (Vina scores/poses)
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._receptor_dir = self.cache_dir / "receptors"
        self._ligand_dir = self.cache_dir / "ligands"
        self._docking_dir = self.cache_dir / "docking_results"
        for d in (self._receptor_dir, self._ligand_dir, self._docking_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._prefilter_dir = self.cache_dir / "prefilter"
        self._prefilter_dir.mkdir(parents=True, exist_ok=True)

    def receptor_path(self, receptor_key: str) -> Path:
        return self._receptor_dir / f"{receptor_key}.pdbqt"

    def ligand_path(self, ligand_key: str) -> Path:
        return self._ligand_dir / f"{ligand_key}.pdbqt"

    def docking_result_path(self, docking_key: str) -> Path:
        return self._docking_dir / f"{docking_key}.json"

    def ligand_conformer_marker_path(self, ligand_key: str) -> Path:
        # A lightweight marker file so we can tell “ligand prep succeeded”.
        return self._ligand_dir / f"{ligand_key}.prep_ok"

    @staticmethod
    def protein_id_hash(protein_id: str) -> str:
        return hashlib.sha256(protein_id.encode("utf-8")).hexdigest()

    def has_receptor(self, receptor_key: str) -> bool:
        return self.receptor_path(receptor_key).exists()

    def has_ligand(self, ligand_key: str) -> bool:
        return self.ligand_path(ligand_key).exists()

    def has_docking_result(self, docking_key: str) -> bool:
        return self.docking_result_path(docking_key).exists()

    # ── Prefilter cache helpers ──────────────────────────────────────────────

    def prefilter_scores_path(self, key: str) -> Path:
        return self._prefilter_dir / f"{key}.scores.json"

    def prefilter_shortlist_path(self, key: str) -> Path:
        return self._prefilter_dir / f"{key}.shortlist.json"


def make_prefilter_scores_key(protein_id: str, smiles_list: list[str], cfg: VinaConfig) -> str:
    """
    Deterministic cache key for DeepPurpose prefilter scores.
    Keyed on (protein_id, ordered smiles list, prefilter-related config).
    """
    payload = {
        "type": "prefilter_scores",
        "protein_id": protein_id,
        "smiles": smiles_list,
        "prefilter": {
            "enabled": cfg.prefilter_enabled,
            "top_k": cfg.prefilter_top_k,
            "top_percent": cfg.resolved_prefilter_top_percent(),
            "threshold": cfg.prefilter_affinity_threshold,
        },
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def make_prefilter_shortlist_key(protein_id: str, smiles_list: list[str], cfg: VinaConfig) -> str:
    """
    Deterministic cache key for shortlist selection.
    """
    payload = {
        "type": "prefilter_shortlist",
        "protein_id": protein_id,
        "smiles": smiles_list,
        "prefilter": {
            "top_k": cfg.prefilter_top_k,
            "top_percent": cfg.resolved_prefilter_top_percent(),
            "threshold": cfg.prefilter_affinity_threshold,
        },
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()

