from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import requests

from models.protein.structure_engine import ESMFoldClient


class StructureSource(str, Enum):
    PDB = "PDB"
    COLABFOLD = "COLABFOLD"
    ESMFOLD = "ESMFOLD"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class ProteinStructureResult:
    pdb_string: str
    structure_source: StructureSource
    is_experimental: bool
    mean_plddt: float | None
    confidence_label: str
    provenance: dict[str, Any] = field(default_factory=dict)


class ProteinStructureResolver:
    def __init__(
        self,
        *,
        cache_dir: str | Path = "result/structure_cache",
        colabfold_dir: str | Path = "result/colabfold_cache",
        allow_remote_colabfold: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.colabfold_dir = Path(colabfold_dir)
        self.allow_remote_colabfold = allow_remote_colabfold
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.colabfold_dir.mkdir(parents=True, exist_ok=True)
        self.esm = ESMFoldClient()

    def _confidence_label(self, plddt: float | None, experimental: bool) -> str:
        if experimental:
            return "Experimental"
        if plddt is None:
            return "Unknown"
        if plddt >= 80:
            return "High confidence"
        if plddt >= 60:
            return "Medium confidence"
        return "Low confidence"

    def _cache_key(self, sequence: str) -> str:
        import hashlib

        return hashlib.sha256(sequence.encode("utf-8")).hexdigest()

    def _pdb_from_rcsb(self, sequence: str) -> ProteinStructureResult | None:
        """
        Best-effort RCSB retrieval. If unavailable/offline/no match, return None.
        """
        # Lightweight sequence-search endpoint payload.
        query = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {"value": sequence[:200], "identity_cutoff": 0.95, "evalue_cutoff": 1.0},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 1}},
        }
        try:
            r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            rows = data.get("result_set") or []
            if not rows:
                return None
            pdb_id = rows[0].get("identifier")
            if not pdb_id:
                return None
            pdb_url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
            rpdb = requests.get(pdb_url, timeout=10)
            if rpdb.status_code != 200 or "ATOM" not in rpdb.text:
                return None
            # metadata best-effort
            meta_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            rm = requests.get(meta_url, timeout=10)
            meta = rm.json() if rm.status_code == 200 else {}
            resolution = None
            try:
                resolution = (meta.get("rcsb_entry_info") or {}).get("resolution_combined", [None])[0]
            except Exception:
                resolution = None
            method = None
            try:
                ex = meta.get("exptl") or []
                method = ex[0].get("method") if ex else None
            except Exception:
                method = None
            return ProteinStructureResult(
                pdb_string=rpdb.text,
                structure_source=StructureSource.PDB,
                is_experimental=True,
                mean_plddt=None,
                confidence_label="Experimental",
                provenance={
                    "pdb_id": pdb_id,
                    "resolution": resolution,
                    "experimental_method": method,
                },
            )
        except Exception:
            return None

    def _from_local_colabfold(self, sequence: str) -> ProteinStructureResult | None:
        key = self._cache_key(sequence)
        pdb_path = self.colabfold_dir / f"{key}.pdb"
        json_path = self.colabfold_dir / f"{key}.json"
        if not pdb_path.exists():
            return None
        try:
            pdb = pdb_path.read_text(encoding="utf-8", errors="ignore")
            conf = None
            if json_path.exists():
                meta = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
                conf = meta.get("mean_plddt")
            label = self._confidence_label(conf, experimental=False)
            return ProteinStructureResult(
                pdb_string=pdb,
                structure_source=StructureSource.COLABFOLD,
                is_experimental=False,
                mean_plddt=conf,
                confidence_label=label,
                provenance={"local_colabfold": True, "cache_key": key},
            )
        except Exception:
            return None

    def _from_remote_colabfold(self, sequence: str) -> ProteinStructureResult | None:
        # Placeholder for optional remote retrieval endpoint(s).
        # Keep fully optional and failure-safe.
        if not self.allow_remote_colabfold:
            return None
        return None

    def _from_esmfold(self, sequence: str) -> ProteinStructureResult | None:
        try:
            pdb = self.esm.fold(sequence)
            return ProteinStructureResult(
                pdb_string=pdb,
                structure_source=StructureSource.ESMFOLD,
                is_experimental=False,
                mean_plddt=None,
                confidence_label="Unknown",
                provenance={"fallback": "esmfold"},
            )
        except Exception:
            return None

    def resolve(self, sequence: str) -> ProteinStructureResult:
        # PDB first
        r = self._pdb_from_rcsb(sequence)
        if r is not None:
            return r
        # ColabFold local-first
        r = self._from_local_colabfold(sequence)
        if r is not None:
            return r
        # Optional remote ColabFold
        r = self._from_remote_colabfold(sequence)
        if r is not None:
            return r
        # ESMFold fallback
        r = self._from_esmfold(sequence)
        if r is not None:
            return r
        # Last-resort empty structure placeholder
        return ProteinStructureResult(
            pdb_string="",
            structure_source=StructureSource.UNKNOWN,
            is_experimental=False,
            mean_plddt=None,
            confidence_label="Unknown",
            provenance={"error": "all_structure_sources_failed"},
        )

