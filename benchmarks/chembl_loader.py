from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from benchmarks.targets import BenchmarkTargetConfig, resolve_target
from benchmarks.utils import ensure_dir, morgan_fp, preprocess_smiles


@dataclass(slots=True)
class ChEMBLRecord:
    chembl_id: str
    smiles: str
    activity_value: float | None
    activity_unit: str | None
    target_name: str
    target_chembl_id: str


class ChEMBLBenchmarkLoader:
    def __init__(self, cache_dir: str | Path = "result/benchmarks/chembl"):
        self.cache_dir = ensure_dir(cache_dir)

    def _raw_path(self, target: BenchmarkTargetConfig) -> Path:
        return self.cache_dir / f"{target.chembl_target_id}.raw.json"

    def _processed_path(self, target: BenchmarkTargetConfig) -> Path:
        return self.cache_dir / f"{target.chembl_target_id}.processed.csv"

    def fetch_or_load(self, target_name: str) -> list[ChEMBLRecord]:
        target = resolve_target(target_name)
        raw_path = self._raw_path(target)
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            payload = self._download_target_payload(target)
            raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        records = self._parse_payload(payload, target)
        self._write_processed(records, self._processed_path(target))
        return records

    def _download_target_payload(self, target: BenchmarkTargetConfig) -> dict[str, Any]:
        # ChEMBL REST endpoint (activity with canonical smiles).
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "target_chembl_id": target.chembl_target_id,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _parse_payload(self, payload: dict[str, Any], target: BenchmarkTargetConfig) -> list[ChEMBLRecord]:
        rows = payload.get("activities", []) or []
        smiles: list[str] = []
        by_smi: dict[str, ChEMBLRecord] = {}
        for row in rows:
            mol = row.get("molecule_structures") or {}
            smi = mol.get("canonical_smiles")
            if not smi:
                continue
            smiles.append(smi)
        canonical = set(preprocess_smiles(smiles))
        for row in rows:
            mol = row.get("molecule_structures") or {}
            smi = mol.get("canonical_smiles")
            if not smi:
                continue
            from benchmarks.utils import canonicalize_smiles, strip_salts

            cs = canonicalize_smiles(strip_salts(smi))
            if cs not in canonical:
                continue
            rec = ChEMBLRecord(
                chembl_id=str(row.get("molecule_chembl_id") or ""),
                smiles=cs,
                activity_value=(float(row["standard_value"]) if row.get("standard_value") not in (None, "") else None),
                activity_unit=(str(row.get("standard_units")) if row.get("standard_units") else None),
                target_name=target.name,
                target_chembl_id=target.chembl_target_id,
            )
            # dedup by canonical smiles; keep lower activity when available.
            prev = by_smi.get(cs)
            if prev is None:
                by_smi[cs] = rec
            else:
                pv = prev.activity_value
                nv = rec.activity_value
                if nv is not None and (pv is None or nv < pv):
                    by_smi[cs] = rec
        return list(by_smi.values())

    def _write_processed(self, rows: list[ChEMBLRecord], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["chembl_id", "smiles", "activity_value", "activity_unit", "target_name", "target_chembl_id"])
            for r in rows:
                w.writerow([r.chembl_id, r.smiles, r.activity_value, r.activity_unit, r.target_name, r.target_chembl_id])

    def build_fingerprint_db(self, records: list[ChEMBLRecord]) -> dict[str, Any]:
        fp_db = {}
        for r in records:
            fp = morgan_fp(r.smiles)
            if fp is not None:
                fp_db[r.smiles] = fp
        return fp_db

