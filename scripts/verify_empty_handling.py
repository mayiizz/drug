"""TestClient verification for the stabilised empty-result handling.

Runs three scenarios against the FastAPI app without spinning up the full
ESMFold / REINVENT / Vina stack by injecting a mock ``denovo_pipeline`` and
stubbing the UniProt fetcher.

Cases (per stabilisation plan, Step 9):
    A. Protein-only with rich pipeline output  -> 200, candidates non-empty,
       diagnostics present.
    B. Protein-only with mocked empty ranked    -> 200, candidates=[],
       diagnostics present, no 422.
    C. seed_smiles="INVALID"                    -> 422 from Pydantic field
       validator (seed_smiles validator).

Run:
    python scripts/verify_empty_handling.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Make the project root importable when run as ``python scripts/verify_empty_handling.py``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify")


def _fake_protein_record(gene: str) -> dict[str, Any]:
    return {
        "gene": gene,
        "name": gene,
        "uniprot_id": "P00533",
        "sequence": ("M" + "A" * 200),
    }


def _build_mock_pipeline(ranked_payload: list[dict], diagnostics: dict) -> MagicMock:
    cura = MagicMock()
    cura.last_diagnostics = diagnostics
    pipe = MagicMock()
    pipe._cura = cura
    pipe.run = MagicMock(return_value=ranked_payload)
    return pipe


def main() -> int:
    # Import after path setup; this triggers FastAPI app construction but the
    # lifespan startup only runs inside a TestClient context manager. We skip
    # that on purpose to avoid loading the real models.
    try:
        from fastapi.testclient import TestClient  # noqa: WPS433
        import backend.main as backend_main  # noqa: WPS433
    except Exception as e:
        log.error("Failed to import backend app: %s", e)
        return 2

    failures: list[str] = []

    # ── Case A: rich payload ─────────────────────────────────────────────
    rich_ranked = [
        {
            "smiles": "CCO",
            "qed": 0.62,
            "binding_nm": 250.0,
            "sa_score": 3.1,
            "score": 0.78,
            "mol_weight": 320.0,
            "logp": 2.3,
            "tpsa": 55.0,
            "rot_bonds": 4,
            "h_donors": 1,
            "h_acceptors": 4,
            "admet": {
                "drug_likeness": 0.62,
                "sa_score": 3.1,
                "logp": 2.3,
                "tpsa": 55.0,
                "molecular_weight": 320.0,
            },
            "validation": {"composite_score": 0.7},
            "vina_score": -7.4,
            "pose_energy": -7.4,
            "docking_success": True,
        }
    ]
    rich_diag = {
        "stage_counts": {"generated": 256, "valid": 12, "returned": 1},
        "rejection_counts": {},
        "timings_ms": {"total_run": 12345.0},
        "conformer": {},
        "profile": "production",
        "thresholds": {"MIN_MW": 220.0, "MAX_SA": 5.5},
        "drop_stage": None,
    }
    backend_main.denovo_pipeline = _build_mock_pipeline(rich_ranked, rich_diag)

    with patch.object(backend_main, "fetch_protein_sequence", side_effect=_fake_protein_record):
        client = TestClient(backend_main.app)
        rA = client.post(
            "/denovo",
            json={"gene_name": "EGFR", "num_molecules": 1, "sample_size": 64},
        )

    log.info("[A] status=%s", rA.status_code)
    if rA.status_code != 200:
        failures.append(f"A: expected 200, got {rA.status_code}: {rA.text[:200]}")
    else:
        body = rA.json()
        if not body.get("candidates"):
            failures.append("A: expected non-empty candidates")
        if "diagnostics" not in body:
            failures.append("A: missing diagnostics field")
        else:
            log.info(
                "[A] candidates=%d diagnostics.profile=%s",
                len(body["candidates"]),
                body["diagnostics"].get("profile"),
            )

    # ── Case B: empty ranked ─────────────────────────────────────────────
    empty_diag = {
        "stage_counts": {
            "generated": 256,
            "safe_pass": 200,
            "conformer_pass": 80,
            "druglike_pass": 0,
            "ranked": 0,
            "returned": 0,
        },
        "rejection_counts": {"rejected_druglike": 80},
        "timings_ms": {"total_run": 9876.0},
        "conformer": {"mean_strain_total": 14.2},
        "profile": "production",
        "thresholds": {"MIN_MW": 220.0, "MAX_SA": 5.5},
        "drop_stage": "druglike_filter",
    }
    backend_main.denovo_pipeline = _build_mock_pipeline([], empty_diag)

    with patch.object(backend_main, "fetch_protein_sequence", side_effect=_fake_protein_record):
        client = TestClient(backend_main.app)
        rB = client.post(
            "/denovo",
            json={"gene_name": "EGFR", "num_molecules": 1, "sample_size": 64},
        )

    log.info("[B] status=%s", rB.status_code)
    if rB.status_code != 200:
        failures.append(f"B: expected 200, got {rB.status_code}: {rB.text[:200]}")
    else:
        body = rB.json()
        if body.get("candidates") != []:
            failures.append(f"B: expected candidates=[], got {body.get('candidates')!r}")
        if body.get("total_returned", -1) != 0:
            failures.append(f"B: expected total_returned=0, got {body.get('total_returned')!r}")
        if "diagnostics" not in body:
            failures.append("B: missing diagnostics field")
        else:
            diag = body["diagnostics"]
            if diag.get("drop_stage") != "druglike_filter":
                failures.append(
                    f"B: expected drop_stage=druglike_filter, got {diag.get('drop_stage')!r}"
                )
            if diag.get("stage_counts", {}).get("returned") != 0:
                failures.append("B: expected stage_counts.returned == 0")
            log.info("[B] candidates=[] diagnostics.drop_stage=%s", diag.get("drop_stage"))

    # ── Case C: invalid seed_smiles ──────────────────────────────────────
    backend_main.denovo_pipeline = _build_mock_pipeline([], empty_diag)
    with patch.object(backend_main, "fetch_protein_sequence", side_effect=_fake_protein_record):
        client = TestClient(backend_main.app)
        rC = client.post(
            "/denovo",
            json={
                "gene_name": "EGFR",
                "num_molecules": 1,
                "sample_size": 64,
                "seed_smiles": "INVALID",
            },
        )

    log.info("[C] status=%s", rC.status_code)
    if rC.status_code != 422:
        failures.append(f"C: expected 422 for invalid seed, got {rC.status_code}: {rC.text[:200]}")
    else:
        body = rC.json()
        log.info("[C] 422 detail snippet: %s", str(body)[:200])

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n========================================")
    if failures:
        print(f"VERIFY FAILED ({len(failures)} issues):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFY PASSED: cases A, B, C all behave as expected.")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
