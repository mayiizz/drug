#!/usr/bin/env python3
"""Seed Neo4j Aura with CURA drug–protein–disease knowledge graph.

Usage (from repo root, with .env configured):

    python scripts/seed_neo4j.py

Idempotent: uses MERGE + unique constraints so re-runs are safe.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from models.graph.neo4j_client import (  # noqa: E402
    Neo4jNotConfiguredError,
    get_driver,
    is_configured,
    run_cypher,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("seed_neo4j")

DATA = ROOT / "data"
DPD_CSV = DATA / "drug_protein_disease.csv"
SIDER_CSV = DATA / "sider_side_effects.csv"
SMILES_JSON = DATA / "sider_smiles.json"

# Curated similar pairs for demo SIMILAR_TO edges (drug names as in DPD CSV).
SIMILAR_PAIRS: list[tuple[str, str, float]] = [
    ("Erlotinib", "Gefitinib", 0.72),
    ("Erlotinib", "Osimertinib", 0.65),
    ("Gefitinib", "Afatinib", 0.61),
    ("Afatinib", "Dacomitinib", 0.64),
    ("Imatinib", "Dasatinib", 0.68),
    ("Imatinib", "Nilotinib", 0.70),
    ("Dasatinib", "Bosutinib", 0.62),
    ("Crizotinib", "Alectinib", 0.58),
    ("Alectinib", "Brigatinib", 0.60),
    ("Vemurafenib", "Dabrafenib", 0.74),
    ("Dabrafenib", "Encorafenib", 0.69),
    ("Trametinib", "Cobimetinib", 0.63),
    ("Palbociclib", "Ribociclib", 0.80),
    ("Ribociclib", "Abemaciclib", 0.72),
    ("Ibrutinib", "Acalabrutinib", 0.70),
    ("Acalabrutinib", "Zanubrutinib", 0.68),
    ("Aspirin", "Ibuprofen", 0.55),
    ("Ibuprofen", "Naproxen", 0.58),
    ("Metformin", "Sitagliptin", 0.40),
    ("Empagliflozin", "Dapagliflozin", 0.75),
    ("Atorvastatin", "Rosuvastatin", 0.77),
    ("Rosuvastatin", "Simvastatin", 0.60),
    ("Sertraline", "Fluoxetine", 0.66),
    ("Fluoxetine", "Escitalopram", 0.55),
    ("Sotorasib", "Adagrasib", 0.71),
    ("Olaparib", "Rucaparib", 0.68),
    ("Apixaban", "Rivaroxaban", 0.55),
    ("Omeprazole", "Pantoprazole", 0.70),
    ("Pembrolizumab", "Nivolumab", 0.50),
]

# RDKit Morgan/Tanimoto auto-similarity thresholds.
SIM_TANIMOTO_MIN = 0.35
SIM_MAX_NEIGHBORS = 6


CONSTRAINTS = [
    "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT protein_name IF NOT EXISTS FOR (p:Protein) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT sideeffect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
]


def _title_drug(name: str) -> str:
    """Normalize SIDER lowercase names to title case for MERGE with DPD drugs."""
    return name.strip().title() if name else name


def ensure_constraints() -> None:
    for stmt in CONSTRAINTS:
        run_cypher(stmt, write=True)
    logger.info("Constraints ensured.")


def load_smiles_map() -> dict[str, str]:
    if not SMILES_JSON.exists():
        return {}
    try:
        raw = json.loads(SMILES_JSON.read_text())
        return {str(k).strip().lower(): str(v) for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Could not load SMILES map: %s", exc)
        return {}


def seed_drug_protein_disease(smiles_map: dict[str, str]) -> int:
    if not DPD_CSV.exists():
        raise FileNotFoundError(f"Missing {DPD_CSV}")
    df = pd.read_csv(DPD_CSV)
    rows = 0
    for _, row in df.iterrows():
        drug = str(row["drug"]).strip()
        protein = str(row["protein"]).strip()
        disease = str(row["disease"]).strip()
        smiles = smiles_map.get(drug.lower())
        run_cypher(
            """
            MERGE (d:Drug {name: $drug})
            ON CREATE SET d.smiles = $smiles
            ON MATCH SET d.smiles = coalesce(d.smiles, $smiles)
            MERGE (p:Protein {name: $protein})
            MERGE (dis:Disease {name: $disease})
            MERGE (d)-[:BINDS]->(p)
            MERGE (p)-[:ASSOCIATED_WITH]->(dis)
            MERGE (d)-[:TREATS]->(dis)
            """,
            {"drug": drug, "protein": protein, "disease": disease, "smiles": smiles},
            write=True,
        )
        rows += 1
    logger.info("Seeded %d drug–protein–disease rows.", rows)
    return rows


def seed_side_effects() -> int:
    if not SIDER_CSV.exists():
        logger.warning("No SIDER CSV at %s — skipping side effects.", SIDER_CSV)
        return 0
    df = pd.read_csv(SIDER_CSV)
    count = 0
    for _, row in df.iterrows():
        drug = _title_drug(str(row["drugname"]))
        effect = str(row["pt"]).strip()
        if not drug or not effect:
            continue
        run_cypher(
            """
            MERGE (d:Drug {name: $drug})
            MERGE (s:SideEffect {name: $effect})
            MERGE (d)-[:HAS_SIDE_EFFECT]->(s)
            """,
            {"drug": drug, "effect": effect},
            write=True,
        )
        count += 1
    logger.info("Seeded %d drug–side-effect links.", count)
    return count


def _merge_similar(a: str, b: str, score: float) -> None:
    run_cypher(
        """
        MATCH (a:Drug {name: $a}), (b:Drug {name: $b})
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.score = $score
        MERGE (b)-[r2:SIMILAR_TO]->(a)
        SET r2.score = $score
        """,
        {"a": a, "b": b, "score": float(score)},
        write=True,
    )


def seed_similar() -> int:
    count = 0
    for a, b, score in SIMILAR_PAIRS:
        _merge_similar(a, b, score)
        count += 1
    logger.info("Seeded %d curated SIMILAR_TO pairs.", count)
    return count


def seed_similar_from_smiles(smiles_map: dict[str, str]) -> int:
    """Add SIMILAR_TO edges from Morgan Tanimoto among drugs with SMILES."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("RDKit not available — skipping fingerprint SIMILAR_TO seeding.")
        return 0

    # Canonical Title-Case drug names that exist in the graph and have SMILES.
    graph_drugs = {
        row["name"]
        for row in run_cypher("MATCH (d:Drug) RETURN d.name AS name")
        if row.get("name")
    }
    named: list[tuple[str, object]] = []
    for raw_name, smiles in smiles_map.items():
        # Prefer exact Title-Case match against seeded Drug nodes.
        title = raw_name.strip().title()
        candidates = [n for n in graph_drugs if n.lower() == raw_name.lower()]
        drug_name = candidates[0] if candidates else title
        if drug_name not in graph_drugs:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

            fp = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
        except Exception:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        named.append((drug_name, fp))

    if len(named) < 2:
        logger.info("Not enough SMILES drugs for fingerprint similarity.")
        return 0

    # Top-k neighbors per drug above threshold (undirected via _merge_similar).
    neighbors: dict[str, list[tuple[str, float]]] = {n: [] for n, _ in named}
    for i in range(len(named)):
        a_name, a_fp = named[i]
        for j in range(i + 1, len(named)):
            b_name, b_fp = named[j]
            score = float(DataStructs.TanimotoSimilarity(a_fp, b_fp))
            if score < SIM_TANIMOTO_MIN:
                continue
            neighbors[a_name].append((b_name, score))
            neighbors[b_name].append((a_name, score))

    seen: set[tuple[str, str]] = set()
    count = 0
    for drug, pairs in neighbors.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        for other, score in pairs[:SIM_MAX_NEIGHBORS]:
            key = tuple(sorted((drug, other)))
            if key in seen:
                continue
            seen.add(key)
            _merge_similar(drug, other, round(score, 3))
            count += 1

    logger.info(
        "Seeded %d fingerprint SIMILAR_TO pairs (Tanimoto>=%.2f, top-%d).",
        count,
        SIM_TANIMOTO_MIN,
        SIM_MAX_NEIGHBORS,
    )
    return count


def print_counts() -> None:
    rows = run_cypher(
        """
        CALL {
          MATCH (n) RETURN count(n) AS nodes
        }
        CALL {
          MATCH ()-[r]->() RETURN count(r) AS rels
        }
        RETURN nodes, rels
        """
    )
    if rows:
        logger.info("Graph totals: nodes=%s relationships=%s", rows[0]["nodes"], rows[0]["rels"])
    labels = run_cypher(
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS c
        ORDER BY c DESC
        """
    )
    for row in labels:
        logger.info("  %s: %s", row["label"], row["c"])


def main() -> int:
    if not is_configured():
        logger.error(
            "Neo4j is not configured. Copy .env.example to .env and set "
            "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )
        return 1
    try:
        get_driver().verify_connectivity()
    except Neo4jNotConfiguredError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("Cannot connect to Neo4j: %s", exc)
        return 1

    ensure_constraints()
    smiles_map = load_smiles_map()
    seed_drug_protein_disease(smiles_map)
    seed_side_effects()
    seed_similar()
    seed_similar_from_smiles(smiles_map)
    print_counts()
    logger.info("Done. Open Aura Explore or Neo4j Browser to inspect the graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
