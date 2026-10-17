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
    "CREATE CONSTRAINT paper_pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT limitation_name IF NOT EXISTS FOR (l:Limitation) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
]

NOTES_DIR = DATA / "research_notes"

# Curated real PubMed papers + research-knowledge edges (demo provenance).
# DISCUSSES/PROPOSES/RELATED_TO mean curated links, not clinical proof or auto-inferred SUPPORTS.
CURATED_PAPERS: list[dict] = [
    {
        "pmid": "15118073",
        "title": "Erlotinib in previously treated non-small-cell lung cancer",
        "year": 2005,
        "authors": ["Shepherd FA", "Rodrigues Pereira J", "Ciuleanu T"],
        "concepts": ["EGFR inhibition"],
        "methods": ["EGFR TKI therapy"],
        "related_pmids": ["16014882", "27959700"],
        "discusses": [("Drug", "Erlotinib"), ("Protein", "EGFR"), ("Disease", "Lung cancer")],
    },
    {
        "pmid": "16014882",
        "title": "EGFR mutations in lung cancer: correlation with clinical response to gefitinib therapy",
        "year": 2004,
        "authors": ["Lynch TJ", "Bell DW", "Sordella R"],
        "concepts": ["EGFR inhibition", "EGFR mutation selection"],
        "methods": ["EGFR TKI therapy"],
        "related_pmids": ["15118073", "27959700"],
        "discusses": [("Drug", "Gefitinib"), ("Protein", "EGFR"), ("Disease", "Lung cancer")],
    },
    {
        "pmid": "27959700",
        "title": "Osimertinib in Untreated EGFR-Mutated Advanced Non–Small-Cell Lung Cancer",
        "year": 2018,
        "authors": ["Soria JC", "Ohe Y", "Vansteenkiste J"],
        "concepts": ["EGFR inhibition", "Acquired resistance management"],
        "methods": ["Third-generation EGFR TKI therapy"],
        "related_pmids": ["15118073", "16014882"],
        "discusses": [("Drug", "Osimertinib"), ("Protein", "EGFR"), ("Disease", "Lung cancer")],
    },
    {
        "pmid": "11287972",
        "title": "Efficacy and safety of a specific inhibitor of the BCR-ABL tyrosine kinase in chronic myeloid leukemia",
        "year": 2001,
        "authors": ["Druker BJ", "Talpaz M", "Resta DJ"],
        "concepts": ["BCR-ABL inhibition"],
        "methods": ["BCR-ABL TKI therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Imatinib"), ("Protein", "BCR-ABL"), ("Disease", "Leukemia")],
    },
    {
        "pmid": "21639808",
        "title": "Improved survival with vemurafenib in melanoma with BRAF V600E mutation",
        "year": 2011,
        "authors": ["Chapman PB", "Hauschild A", "Robert C"],
        "concepts": ["BRAF V600 targeting"],
        "methods": ["BRAF V600 inhibitor therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Vemurafenib"), ("Protein", "BRAF"), ("Disease", "Melanoma")],
    },
    {
        "pmid": "27959613",
        "title": "Palbociclib and Letrozole in Advanced Breast Cancer",
        "year": 2016,
        "authors": ["Finn RS", "Martin M", "Rugo HS"],
        "concepts": ["CDK4/6 cell-cycle inhibition"],
        "methods": ["CDK4/6 inhibitor therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Palbociclib"), ("Protein", "CDK4"), ("Disease", "Breast cancer")],
    },
    {
        "pmid": "22909280",
        "title": "Metformin and cancer risk in diabetic patients: a systematic review and meta-analysis",
        "year": 2012,
        "authors": ["Noto H", "Goto A", "Tsujimoto T"],
        "concepts": ["Metabolic oncology"],
        "methods": ["Biguanide metabolic modulation"],
        "related_pmids": [],
        "discusses": [("Drug", "Metformin"), ("Disease", "Type 2 diabetes")],
    },
    {
        "pmid": "34096690",
        "title": "Sotorasib for Lung Cancers with KRAS p.G12C Mutation",
        "year": 2021,
        "authors": ["Skoulidis F", "Li BT", "Dy GK"],
        "concepts": ["KRAS G12C inhibition"],
        "methods": ["KRAS G12C inhibitor therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Sotorasib"), ("Protein", "KRAS"), ("Disease", "Lung cancer")],
    },
    {
        "pmid": "25285672",
        "title": "Crizotinib versus chemotherapy in advanced ALK-positive lung cancer",
        "year": 2013,
        "authors": ["Shaw AT", "Kim DW", "Nakagawa K"],
        "concepts": ["ALK rearrangement targeting"],
        "methods": ["ALK TKI therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Crizotinib"), ("Protein", "ALK"), ("Disease", "Lung cancer")],
    },
    {
        "pmid": "28581356",
        "title": "Olaparib for Metastatic Breast Cancer in Patients with a Germline BRCA Mutation",
        "year": 2017,
        "authors": ["Robson M", "Im SA", "Senkus E"],
        "concepts": ["PARP synthetic lethality"],
        "methods": ["PARP inhibitor therapy"],
        "related_pmids": [],
        "discusses": [("Drug", "Olaparib"), ("Protein", "PARP1"), ("Disease", "Breast cancer")],
    },
]

# Method → protein targets + limitations (+ optional SideEffect manifestation).
RESEARCH_METHODS: list[dict] = [
    {
        "name": "EGFR TKI therapy",
        "targets": ["EGFR"],
        "limitations": [
            ("Skin toxicity / rash", "Rash"),
            ("GI toxicity / diarrhea", "Diarrhea"),
            ("Acquired resistance", None),
        ],
    },
    {
        "name": "Third-generation EGFR TKI therapy",
        "targets": ["EGFR"],
        "limitations": [
            ("Skin toxicity / rash", "Rash"),
            ("GI toxicity / diarrhea", "Diarrhea"),
            ("Acquired resistance", None),
        ],
    },
    {
        "name": "BCR-ABL TKI therapy",
        "targets": ["BCR-ABL"],
        "limitations": [
            ("Edema / fluid retention", "Edema"),
            ("Resistance mutations", None),
        ],
    },
    {
        "name": "BRAF V600 inhibitor therapy",
        "targets": ["BRAF"],
        "limitations": [
            ("Cutaneous adverse events", "Rash"),
            ("Rapid monotherapy resistance", None),
        ],
    },
    {
        "name": "CDK4/6 inhibitor therapy",
        "targets": ["CDK4"],
        "limitations": [
            ("Neutropenia / myelosuppression", "Neutropenia"),
            ("Fatigue", "Fatigue"),
        ],
    },
    {
        "name": "KRAS G12C inhibitor therapy",
        "targets": ["KRAS"],
        "limitations": [
            ("GI toxicity / diarrhea", "Diarrhea"),
            ("Nausea", "Nausea"),
        ],
    },
    {
        "name": "ALK TKI therapy",
        "targets": ["ALK"],
        "limitations": [
            ("Vision disorder", "Vision disorder"),
            ("Nausea", "Nausea"),
        ],
    },
    {
        "name": "PARP inhibitor therapy",
        "targets": ["PARP1"],
        "limitations": [
            ("Hematologic toxicity / anemia", "Anemia"),
            ("Nausea", "Nausea"),
        ],
    },
    {
        "name": "Biguanide metabolic modulation",
        "targets": ["AMPK"],
        "limitations": [
            ("GI intolerance", "Diarrhea"),
            ("Nausea", "Nausea"),
        ],
    },
]

# Document id → EXTRACTS_TO Concept/Method names.
RESEARCH_DOCUMENTS: list[dict] = [
    {
        "id": "note-egfr-tki",
        "title": "EGFR TKI approaches overview",
        "path": "egfr_tki_overview.md",
        "extracts_to": [
            ("Concept", "EGFR inhibition"),
            ("Method", "EGFR TKI therapy"),
            ("Method", "Third-generation EGFR TKI therapy"),
        ],
    },
    {
        "id": "note-bcr-abl",
        "title": "BCR-ABL TKI overview",
        "path": "bcr_abl_tki_overview.md",
        "extracts_to": [
            ("Concept", "BCR-ABL inhibition"),
            ("Method", "BCR-ABL TKI therapy"),
        ],
    },
    {
        "id": "note-braf",
        "title": "BRAF-targeted melanoma overview",
        "path": "braf_mek_overview.md",
        "extracts_to": [
            ("Concept", "BRAF V600 targeting"),
            ("Method", "BRAF V600 inhibitor therapy"),
        ],
    },
    {
        "id": "note-parp",
        "title": "PARP inhibition overview",
        "path": "parp_inhibition_overview.md",
        "extracts_to": [
            ("Concept", "PARP synthetic lethality"),
            ("Method", "PARP inhibitor therapy"),
        ],
    },
]


def _resolve_drug_name(raw: str, graph_drugs_lower: dict[str, str] | None = None) -> str:
    """Map SIDER/CSV drug names onto existing Drug nodes when possible."""
    key = raw.strip().lower()
    if not key:
        return raw.strip()
    if graph_drugs_lower and key in graph_drugs_lower:
        return graph_drugs_lower[key]
    return raw.strip().title()


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
    graph_drugs_lower = {
        str(r["name"]).lower(): str(r["name"])
        for r in run_cypher("MATCH (d:Drug) RETURN d.name AS name")
        if r.get("name")
    }
    df = pd.read_csv(SIDER_CSV)
    count = 0
    for _, row in df.iterrows():
        drug = _resolve_drug_name(str(row["drugname"]), graph_drugs_lower)
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


def seed_research_methods() -> int:
    """Seed Method/Limitation nodes and bridges to Protein / SideEffect."""
    count = 0
    for method in RESEARCH_METHODS:
        name = method["name"]
        run_cypher(
            "MERGE (m:Method {name: $name}) SET m.source = 'curated_demo'",
            {"name": name},
            write=True,
        )
        for protein in method.get("targets") or []:
            run_cypher(
                """
                MATCH (m:Method {name: $method})
                MERGE (p:Protein {name: $protein})
                MERGE (m)-[:TARGETS]->(p)
                """,
                {"method": name, "protein": protein},
                write=True,
            )
        for lim_name, side_effect in method.get("limitations") or []:
            run_cypher(
                """
                MATCH (m:Method {name: $method})
                MERGE (l:Limitation {name: $lim})
                SET l.source = 'curated_demo'
                MERGE (m)-[:HAS_LIMITATION]->(l)
                """,
                {"method": name, "lim": lim_name},
                write=True,
            )
            if side_effect:
                run_cypher(
                    """
                    MATCH (l:Limitation {name: $lim})
                    MERGE (s:SideEffect {name: $se})
                    MERGE (l)-[:MANIFESTS_AS]->(s)
                    """,
                    {"lim": lim_name, "se": side_effect},
                    write=True,
                )
        count += 1
    logger.info("Seeded %d Method nodes with limitations/targets.", count)
    return count


def seed_papers() -> int:
    """Seed Paper + Author/Concept/Method research edges (demo provenance)."""
    count = 0
    for paper in CURATED_PAPERS:
        pmid = str(paper["pmid"])
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        run_cypher(
            """
            MERGE (p:Paper {pmid: $pmid})
            SET p.title = $title, p.url = $url, p.year = $year,
                p.source = 'curated_demo'
            """,
            {
                "pmid": pmid,
                "title": paper["title"],
                "url": url,
                "year": int(paper["year"]),
            },
            write=True,
        )
        for author in paper.get("authors") or []:
            run_cypher(
                """
                MATCH (p:Paper {pmid: $pmid})
                MERGE (a:Author {name: $author})
                MERGE (p)-[:AUTHORED_BY]->(a)
                """,
                {"pmid": pmid, "author": author},
                write=True,
            )
        for concept in paper.get("concepts") or []:
            run_cypher(
                """
                MATCH (p:Paper {pmid: $pmid})
                MERGE (c:Concept {name: $concept})
                SET c.source = 'curated_demo'
                MERGE (p)-[:DISCUSSES]->(c)
                """,
                {"pmid": pmid, "concept": concept},
                write=True,
            )
        for method in paper.get("methods") or []:
            run_cypher(
                """
                MATCH (p:Paper {pmid: $pmid})
                MERGE (m:Method {name: $method})
                MERGE (p)-[:PROPOSES]->(m)
                """,
                {"pmid": pmid, "method": method},
                write=True,
            )
        for label, name in paper.get("discusses") or []:
            if label not in ("Drug", "Protein", "Disease", "SideEffect"):
                continue
            run_cypher(
                f"""
                MATCH (p:Paper {{pmid: $pmid}})
                MATCH (n:{label} {{name: $name}})
                MERGE (p)-[:DISCUSSES]->(n)
                """,
                {"pmid": pmid, "name": name},
                write=True,
            )
        count += 1

    # RELATED_TO between papers (second pass so both nodes exist)
    rel_count = 0
    for paper in CURATED_PAPERS:
        pmid = str(paper["pmid"])
        for other in paper.get("related_pmids") or []:
            run_cypher(
                """
                MATCH (a:Paper {pmid: $a}), (b:Paper {pmid: $b})
                MERGE (a)-[:RELATED_TO]->(b)
                MERGE (b)-[:RELATED_TO]->(a)
                """,
                {"a": pmid, "b": str(other)},
                write=True,
            )
            rel_count += 1
    logger.info(
        "Seeded %d Paper nodes (+ authors/concepts/methods); %d RELATED_TO pairs.",
        count,
        rel_count,
    )
    return count


def seed_research_documents() -> int:
    """Seed markdown research notes as Document nodes with EXTRACTS_TO edges."""
    if not NOTES_DIR.exists():
        logger.warning("No research notes dir at %s — skipping Documents.", NOTES_DIR)
        return 0
    count = 0
    for doc in RESEARCH_DOCUMENTS:
        path = NOTES_DIR / doc["path"]
        body = path.read_text(encoding="utf-8") if path.exists() else ""
        run_cypher(
            """
            MERGE (d:Document {id: $id})
            SET d.title = $title, d.path = $path, d.body = $body,
                d.source = 'curated_demo'
            """,
            {
                "id": doc["id"],
                "title": doc["title"],
                "path": str(doc["path"]),
                "body": body[:4000],
            },
            write=True,
        )
        for label, name in doc.get("extracts_to") or []:
            if label not in ("Concept", "Method"):
                continue
            run_cypher(
                f"""
                MATCH (d:Document {{id: $id}})
                MERGE (n:{label} {{name: $name}})
                MERGE (d)-[:EXTRACTS_TO]->(n)
                """,
                {"id": doc["id"], "name": name},
                write=True,
            )
        count += 1
    logger.info("Seeded %d Document research notes.", count)
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
    seed_research_methods()
    seed_papers()
    seed_research_documents()
    print_counts()
    logger.info("Done. Open Aura Explore or Neo4j Browser to inspect the graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
