"""Fetch and cache ChEMBL EGFR (CHEMBL203) actives.

Writes ``data/benchmarks/chembl_egfr_actives.csv`` with columns:

    chembl_id, smiles, pchembl_value

The script is intentionally dependency-light:

* uses ``chembl_webresource_client`` when available (cleaner pagination)
* otherwise falls back to plain ``requests`` against the ChEMBL REST API

Output is **idempotent** - if the CSV already exists and contains rows the
script is a no-op, so it is safe to call from CI smoke runs.

Usage::

    python scripts/download_chembl_egfr.py
    python scripts/download_chembl_egfr.py --target CHEMBL203 --limit 4000
    python scripts/download_chembl_egfr.py --force  # ignore the cache
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmarks" / "chembl_egfr_actives.csv"
CHEMBL_REST_BASE = "https://www.ebi.ac.uk/chembl/api/data"

DEFAULT_TARGET = "CHEMBL203"  # EGFR
DEFAULT_LIMIT = 5000
DEFAULT_PCHEMBL_FLOOR = 5.0  # only keep pChEMBL >= 5 (Ki/Kd <= 10 uM)

logger = logging.getLogger("download_chembl_egfr")


def _to_pchembl(activity: dict[str, Any]) -> float | None:
    """Best-effort coercion to a pChEMBL value.

    Prefers the ChEMBL-supplied ``pchembl_value`` field. Falls back to a
    classic ``pX = 9 - log10(value_nM)`` conversion when the value is in
    units we recognize. Returns ``None`` when no usable number is available.
    """
    raw = activity.get("pchembl_value")
    if raw not in (None, ""):
        try:
            return float(raw)
        except Exception:
            pass
    val = activity.get("standard_value")
    units = (activity.get("standard_units") or "").strip().lower()
    if val in (None, "") or not units:
        return None
    try:
        x = float(val)
    except Exception:
        return None
    if x <= 0:
        return None
    if units == "nm":
        nm = x
    elif units in ("um", "µm", "μm"):
        nm = x * 1e3
    elif units in ("mm",):
        nm = x * 1e6
    elif units in ("m",):
        nm = x * 1e9
    elif units in ("pm",):
        nm = x * 1e-3
    else:
        return None
    return float(9.0 - math.log10(nm))


def _yield_via_webresource_client(target: str, limit: int) -> Iterable[dict[str, Any]]:
    try:
        from chembl_webresource_client.new_client import new_client  # type: ignore
    except Exception:
        return  # signal: no client; let caller fall back
    activities = new_client.activity.filter(  # type: ignore[attr-defined]
        target_chembl_id=target,
        standard_type__in=["IC50", "Ki", "Kd"],
        pchembl_value__isnull=False,
    ).only(  # type: ignore[attr-defined]
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_value",
        "standard_units",
        "standard_type",
    )
    yielded = 0
    for record in activities:
        if yielded >= limit:
            return
        yield record  # type: ignore[misc]
        yielded += 1


def _yield_via_rest(target: str, limit: int) -> Iterable[dict[str, Any]]:
    page_size = 200
    offset = 0
    yielded = 0
    while yielded < limit:
        params = {
            "target_chembl_id": target,
            "standard_type__in": "IC50,Ki,Kd",
            "pchembl_value__isnull": "false",
            "limit": page_size,
            "offset": offset,
        }
        url = f"{CHEMBL_REST_BASE}/activity.json"
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
        except Exception as e:
            logger.warning("ChEMBL REST request failed at offset=%d: %s", offset, e)
            return
        payload = r.json() or {}
        rows = payload.get("activities") or []
        if not rows:
            return
        for row in rows:
            if yielded >= limit:
                return
            yield row
            yielded += 1
        offset += page_size
        page_meta = payload.get("page_meta") or {}
        if not page_meta.get("next"):
            return


def _record_for(row: dict[str, Any]) -> tuple[str, str, float] | None:
    chembl_id = (
        row.get("molecule_chembl_id")
        or (row.get("molecule_structures") or {}).get("molecule_chembl_id")
        or ""
    )
    smiles = row.get("canonical_smiles") or (row.get("molecule_structures") or {}).get(
        "canonical_smiles"
    )
    if not smiles or not chembl_id:
        return None
    pchembl = _to_pchembl(row)
    if pchembl is None:
        return None
    return str(chembl_id), str(smiles), float(pchembl)


def fetch_actives(
    target: str = DEFAULT_TARGET,
    limit: int = DEFAULT_LIMIT,
    pchembl_floor: float = DEFAULT_PCHEMBL_FLOOR,
) -> list[tuple[str, str, float]]:
    """Fetch deduplicated EGFR actives from ChEMBL.

    Tries ``chembl_webresource_client`` first; falls back to the REST API.
    Filters to ``pchembl_value >= pchembl_floor`` and dedupes by SMILES.
    """
    seen: dict[str, tuple[str, str, float]] = {}

    iter_fn = _yield_via_webresource_client(target, limit)
    rows = list(iter_fn) if iter_fn is not None else []
    if not rows:
        rows = list(_yield_via_rest(target, limit))

    for row in rows:
        rec = _record_for(row)
        if rec is None:
            continue
        chembl_id, smiles, pchembl = rec
        if pchembl < pchembl_floor:
            continue
        prev = seen.get(smiles)
        if prev is None or pchembl > prev[2]:
            seen[smiles] = (chembl_id, smiles, pchembl)
    return list(seen.values())


def write_csv(records: list[tuple[str, str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "smiles", "pchembl_value"])
        for chembl_id, smiles, pchembl in records:
            w.writerow([chembl_id, smiles, f"{pchembl:.3f}"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download ChEMBL EGFR actives.")
    p.add_argument("--target", default=DEFAULT_TARGET, help="ChEMBL target id (default CHEMBL203 / EGFR).")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum activities to fetch.")
    p.add_argument(
        "--pchembl-floor",
        type=float,
        default=DEFAULT_PCHEMBL_FLOOR,
        help="Minimum pChEMBL value to keep (default 5.0).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the cache exists.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.output.exists() and not args.force:
        try:
            with args.output.open("r", encoding="utf-8") as f:
                first = f.readline()
                second = f.readline()
            if first and second:
                logger.info(
                    "[chembl] cache exists at %s - reuse (use --force to refresh)",
                    args.output,
                )
                return 0
        except Exception:
            pass

    logger.info(
        "[chembl] fetching target=%s limit=%d pchembl_floor=%.2f",
        args.target,
        args.limit,
        args.pchembl_floor,
    )
    records = fetch_actives(
        target=args.target,
        limit=args.limit,
        pchembl_floor=args.pchembl_floor,
    )
    if not records:
        logger.error("[chembl] no actives downloaded - aborting")
        return 1
    write_csv(records, args.output)
    logger.info("[chembl] wrote %d actives -> %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
