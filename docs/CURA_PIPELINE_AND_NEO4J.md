# CURA — Platform Guide (How It Works & How Neo4j Is Used)

This document is the in-depth conceptual guide to CURA: what the product does, how each layer works end-to-end, how the molecule-generation pipeline runs, and how Neo4j powers the knowledge graph (Network, GraphRAG, and structured search). It is written for readers who want the full picture without reading the source line by line.

---

## Table of contents

1. [What CURA is and why it exists](#1-what-cura-is-and-why-it-exists)
2. [System architecture](#2-system-architecture)
3. [Product surface — every feature explained](#3-product-surface--every-feature-explained)
4. [The core drug-generation pipeline (detailed)](#4-the-core-drug-generation-pipeline-detailed)
5. [Neo4j in CURA — purpose, schema, seeding, and runtime](#5-neo4j-in-cura--purpose-schema-seeding-and-runtime)
6. [Example journeys (how pieces fit together)](#6-example-journeys-how-pieces-fit-together)
7. [Dependencies and degraded modes](#7-dependencies-and-degraded-modes)
8. [Glossary](#8-glossary)
9. [Summary](#9-summary)

---

## 1. What CURA Is and Why It Exists

### 1.1 Problem CURA addresses

Early-stage drug discovery is a search problem:

- A **protein target** (for example EGFR) is implicated in a disease.
- Scientists need **small molecules** that might bind that target, look drug-like, and not be obviously toxic.
- They also need **context**: which known drugs already hit related targets, what diseases they treat, what side effects appear, and what the literature says.

Those tasks usually live in separate tools (docking software, ChEMBL browsers, PubMed, graph databases). CURA puts several of them behind one web UI and one backend so a researcher can move from **target → candidates → explanation** in one place.

### 1.2 What CURA is (and is not)

**CURA is:**

- An AI-assisted **computational drug discovery demo platform**.
- A **Next.js frontend** plus a **FastAPI/Python backend**.
- A place to **generate and rank** candidate molecules for a protein (de novo / seed-biased discovery).
- A place to **analyze** a SMILES string (properties, similar drugs, predicted side effects).
- A place to **explore a biomedical knowledge graph** (Neo4j) visually and with natural-language GraphRAG.
- A place to **ask literature questions** (PubMed document RAG) and **view AlphaFold** structures in 3D.

**CURA is not:**

- A clinical decision system or a substitute for wet-lab validation.
- A full live mirror of ChEMBL / DrugBank / SIDER (the Neo4j graph is **demo-scale**, seeded from project files).
- A single model that “does everything” — different features use different engines (fingerprints, REINVENT, Neo4j, PubMed, AlphaFold).

### 1.3 Two layers

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Web application | Next.js (App Router) | Forms, pages, charts, 3D viewer, calling APIs |
| Backend service | FastAPI + Python science stack | Cheminformatics, ML models, UniProt/ESMFold, Groq, Neo4j, PubMed |

Most heavy science runs on the backend. The frontend mainly collects inputs and displays results. One exception: the **3D Structure** page talks to AlphaFold / UniProt through **Next.js API proxies** and renders in the browser.

### 1.4 Mental model of the whole product

```text
                    ┌─────────────────────────────────────┐
                    │           CURA Frontend             │
                    │  Analysis · Discovery · Network ·   │
                    │  Research · Structure · Exports     │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     FastAPI science API      Next.js proxies          Optional browser Groq
     (generate, analyze,      (AlphaFold / UniProt)    (discovery blurbs only)
      GraphRAG, research)
              │
              ├─ Local ML / RDKit / REINVENT / Tox21 / DeepPurpose
              ├─ External: UniProt, ESMFold, ADMETlab, PubMed, Groq
              └─ Neo4j Aura: Drug Network + GraphRAG + /graph/search
```

---

## 2. System Architecture

### 2.1 Runtime topology

```text
  User (browser)
       │
       ▼
  Next.js frontend  (typically :3000)
       │
       ├── most features  →  FastAPI backend  (typically :8000 or :8002)
       ├── /structure     →  AlphaFold DB + UniProt (via Next.js)
       └── /discovery AI blurbs → Groq (optional public browser key)

  FastAPI backend
       │
       ├── CuraPipeline          → /discover, /denovo
       ├── Local ADMET heuristics → /analyze
       ├── Fingerprint engines    → /similar-drugs, /side-effects
       ├── KnowledgeGraphEngine   → /network  (Neo4j or CSV)
       ├── GraphRAG               → /graphrag, /graph/search  (Neo4j required)
       ├── Research agent         → /research  (PubMed + embeddings + Groq)
       └── UniProt helper         → /protein/{name}
```

Port numbers are configurable. The frontend uses `NEXT_PUBLIC_API_BASE` (for example `http://127.0.0.1:8002`).

### 2.2 Environment configuration

| Variable | Used by | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | Backend GraphRAG, Research Agent | LLM synthesis of answers |
| `NEO4J_URI` | Network, GraphRAG, `/graph/search` | Aura / Bolt connection string |
| `NEO4J_USERNAME` (or `NEO4J_USER`) | Same | Database user |
| `NEO4J_PASSWORD` | Same | Database password |
| `NEXT_PUBLIC_API_BASE` | Frontend | Where the browser sends API calls |
| `NEXT_PUBLIC_GROQ_API_KEY` | Discovery page only (optional) | Short “AI reasoning” blurbs on candidate cards |

If Neo4j is missing: **Drug Network** can fall back to a local CSV graph; **GraphRAG** and **`/graph/search`** require Neo4j.

### 2.3 Key repositories of logic (conceptual map)

| Concern | Where it lives (conceptually) |
|---------|-------------------------------|
| UI pages | Frontend App Router routes (`/analysis`, `/network`, `/protein`, …) |
| HTTP API | FastAPI `backend` entrypoint |
| Generation funnel | Shared runtime pipeline used by both Discovery and De Novo shims |
| Neo4j driver helpers | Graph client (lazy driver, connectivity check, Cypher runner) |
| GraphRAG | Intent router + Cypher templates + Groq synthesis |
| Network visualization | Knowledge-graph engine → NetworkX → PyVis HTML |
| Seed data | Files under `data/` loaded by `scripts/seed_neo4j.py` |

---

## 3. Product Surface — Every Feature Explained

### 3.1 Dashboard (`/`)

Landing overview of tools and discovery entry points. Status indicators are largely **presentational** — they are not a live health check of every model, Neo4j, or Vina binary.

### 3.2 Molecule Analysis (`/analysis`)

**Input:** SMILES.  
**API:** `POST /analyze`.

**What happens:**

1. Validate the SMILES.
2. Compute physicochemical descriptors (molecular weight, LogP, TPSA, H-bond donors/acceptors, QED-style drug-likeness, etc.).
3. Apply **local ADMET-style heuristics** (rough solubility / toxicity / drug-likeness style scores — not a full regulatory ADMET suite).
4. Render a 2D molecule image.

**What this is not:** the multi-stage generation pipeline. It is a fast single-molecule profile.  
**Neo4j:** not used.

### 3.3 Similar Drugs (`/similar-drugs`)

**Input:** SMILES.  
**API:** `POST /similar-drugs`.

**What happens:**

1. Fingerprint the query molecule (Morgan fingerprints).
2. Compare to a precomputed **ChEMBL-style library** on disk (`chembl_100k.csv` + `chembl_fingerprints.npy`).
3. Rank by Tanimoto similarity and return top neighbors.

**Neo4j:** not used. (Neo4j’s `SIMILAR_TO` edges are a separate, smaller curated/fingerprint graph for Network/GraphRAG.)

### 3.4 Side Effects (`/side-effects`)

**Input:** SMILES.  
**API:** `POST /side-effects`.

**What happens:**

1. Fingerprint the molecule.
2. Find nearest neighbors in a **SIDER-derived** drug set that has known adverse-effect terms.
3. Transfer those terms as **predicted** side effects (similarity-based transfer, not a causal model).

**Neo4j:** not used at prediction time. The same SIDER-style table is also **seeded into Neo4j** as `HAS_SIDE_EFFECT` edges for graph exploration and GraphRAG — a different runtime path.

### 3.5 Drug Network + GraphRAG (`/network`)

**Inputs:**

- Optional drug name and/or disease name → interactive network.
- Free-text question → GraphRAG.

**APIs:**

- `POST /network` → HTML graph visualization.
- `POST /graphrag` → natural-language answer + evidence paths.
- (Backend also exposes `POST /graph/search` for structured retrieval without an LLM.)

**What happens (Network):**

1. Backend queries Neo4j for a neighborhood (drugs, proteins, diseases, side effects, similar drugs).
2. Builds a NetworkX graph and exports interactive PyVis HTML.
3. Frontend embeds that HTML so you can pan/zoom the relationships.

If Neo4j is down or unconfigured, Network can rebuild a simpler graph from `drug_protein_disease.csv`.

**What happens (GraphRAG):** see [Section 5.6](#56-how-graphrag-uses-neo4j-step-by-step).

This page is the **primary Neo4j surface** in the UI.

### 3.6 Protein Target Drug Generation / De Novo (`/protein`)

**Input:** gene name or UniProt accession (e.g. EGFR, P00533), how many candidates to return, how many molecules to sample; optional seed SMILES.  
**API:** `POST /denovo`.

**What happens:** UniProt lookup → full **CuraPipeline** (structure → pocket → REINVENT → filters → ranking).  
An optional seed SMILES only **biases ranking** toward chemical similarity; it does **not** condition the generative model the way a classic “grow-from-seed” approach would.

**Neo4j:** not used.

### 3.7 Seed Candidate Drug Generation / Discovery (`/discovery`)

**Input:** target name, protein sequence (often fetched via UniProt first), and a **required seed SMILES**.  
**API:** `POST /discover`.

**What happens:** same core pipeline as de novo, with a seed required for the discovery API shape and typically a larger fixed candidate budget. The UI may also call Groq from the browser for short “reasoning” blurbs on cards (optional public key).

**Neo4j:** not used.

### 3.8 Research Agent (`/research`)

**Input:** research question and/or molecule + target context.  
**API:** `POST /research` (streamed).

**What happens:** a research agent searches literature (PubMed), chunks and embeds text, retrieves relevant passages (vector search), and synthesizes an answer with Groq. Responses stream to the UI.

**Neo4j is not the literature store.** This is **document RAG**, not graph RAG. Use GraphRAG for facts that exist as edges in the seeded demo graph; use Research Agent for broader scientific narrative.

### 3.9 3D Structure Viewer (`/structure`)

**Input:** UniProt accession or gene name.  
**Path:** Next.js proxies → public **AlphaFold DB** + UniProt; browser 3Dmol viewer (ribbon, surface, confidence views).

**Important distinction — two structure stories:**

| Path | Used for | Source |
|------|----------|--------|
| Structure page | Teaching / exploration of a published fold | AlphaFold DB |
| Generation pipeline | Pocket finding and optional docking | PDB → ColabFold → ESMFold (tiered) |

They are intentionally separate.

### 3.10 Exports (`/exports`)

After a successful generation run, candidates can be written to CSV/SDF on the server and downloaded via `GET /export/csv` and `GET /export/sdf`.

### 3.11 Feature ↔ Neo4j cheat sheet

| Feature | Uses Neo4j? |
|---------|-------------|
| Molecule analysis | No |
| Similar drugs (ChEMBL fingerprints) | No |
| Side-effect predictor (fingerprint transfer) | No |
| De novo / discovery generation | No |
| Research Agent (PubMed RAG) | No |
| AlphaFold 3D viewer | No |
| Drug Network visualization | Yes (preferred; CSV fallback) |
| GraphRAG Q&A | Yes (required) |
| Structured `/graph/search` | Yes (required) |

---

## 4. The Core Drug-Generation Pipeline (Detailed)

Both **Discovery** and **De Novo** call the same orchestration engine (the CURA runtime pipeline). Think of it as a **funnel**: many molecules are proposed; most are filtered away; a few ranked survivors are returned.

### Stage A — Understand the protein

1. **Fetch / accept sequence**  
   Gene or UniProt accession → protein record (name, UniProt ID, amino-acid sequence).

2. **Resolve a 3D structure** (best available first)  
   - Prefer experimental PDB when possible.  
   - Else ColabFold-style folding if available.  
   - Else **ESMFold** prediction.  
   Structures are cached for repeated runs.

3. **Extract a binding pocket**  
   Geometric analysis estimates where a ligand might bind (including kinase-oriented heuristics when relevant). The pocket guides fragment seeds and docking box placement.

### Stage B — Propose molecules

4. **Pocket-aware fragment seeds**  
   Small chemical seeds from pocket context (with safety filters). Fallback: known kinase-friendly scaffolds so generation can still proceed.

5. **Generate with REINVENT**  
   A pretrained **REINVENT4 SMILES LSTM prior** samples many SMILES strings. Pocket seeds can be merged when sampling alone is thin.  
   SAFE-GPT-style components in the repo are used mainly as **filters**, not as the primary generator in the orchestrated path.

### Stage C — Filter and enrich

6. **Chemical safety / structural filters** — remove PAINS-like / reactive / undesirable patterns.  
7. **3D geometry** — conformers + force-field optimization (e.g. MMFF94); drop failures.  
8. **Drug-likeness rules** — Lipinski-style / QED thresholds (strictness depends on runtime profile).  
9. **Optional embedding deduplication** — ChemBERTa-style collapse of near-duplicates.  
10. **Pharmacophore / target rules** — target-aware chemical rules further prune or score.

### Stage D — Binding, ADMET, docking

11. **DeepPurpose binding prefilter** — protein–ligand ML shortlist; affinity-like scores in the UI.  
12. **ADMET and Tox21** — local heuristics ± external ADMETlab; Tox21-style multi-task fingerprint model.  
13. **Optional AutoDock Vina docking** — if binaries/tools exist; otherwise continue degraded without docking scores.

### Stage E — Rank and return

14. **Multi-objective ranking** — binding, drug-likeness, ADMET/Tox, docking (if present), optional **seed similarity bias** → composite score and PASS/WARN/FAIL-style validation.  
15. **Return top candidates + diagnostics** — ranked SMILES, property fields, stage drop reasons; optional CSV/SDF export.

### Funnel diagram

```text
Protein target
    → 3D structure + pocket
    → hundreds/thousands of generated SMILES
    → safety / 3D / drug-likeness / pharmacophore filters
    → ML binding + ADMET/Tox (± docking)
    → ranked shortlist (few to tens of candidates)
```

**Seed molecule role:** primarily a **similarity bias in ranking**, not a forced generative condition. Protein structure and pocket context drive what “de novo” means here.

---

## 5. Neo4j in CURA — Purpose, Schema, Seeding, and Runtime

This section is the deep dive on the knowledge graph.

### 5.1 Why Neo4j is here

Drug discovery knowledge is naturally a **graph**, not a single table:

| Fact type | Graph shape |
|-----------|-------------|
| Drug hits a target | `Drug -BINDS-> Protein` |
| Target linked to disease biology | `Protein -ASSOCIATED_WITH-> Disease` |
| Drug used in a disease context | `Drug -TREATS-> Disease` |
| Adverse effects | `Drug -HAS_SIDE_EFFECT-> SideEffect` |
| Chemical / therapeutic neighbors | `Drug -SIMILAR_TO-> Drug` (with score) |

A spreadsheet can store rows, but multi-hop questions are awkward:

> “Which drugs treat lung cancer, bind EGFR, and are similar to Erlotinib?”

Neo4j stores nodes and relationships so CURA can:

1. **Visualize** neighborhoods on the Drug Network page.  
2. **Retrieve evidence paths** for GraphRAG answers.  
3. **Serve structured filters** via `/graph/search` without an LLM.

This deployment is intentionally a **demo-scale knowledge graph** seeded from project CSV/JSON files — not a live enterprise biomedical warehouse.

### 5.2 What problem GraphRAG solves vs other CURA features

| Need | Right tool in CURA |
|------|---------------------|
| “Generate new molecules for EGFR” | De novo / Discovery pipeline |
| “Find ChEMBL neighbors of this SMILES” | Similar Drugs (fingerprints) |
| “Predict side effects from structure” | Side Effects page (fingerprint transfer) |
| “What does the literature say?” | Research Agent (PubMed RAG) |
| “What does our knowledge graph say about Erlotinib?” | GraphRAG / Network / `/graph/search` |

GraphRAG answers are **grounded only in retrieved Neo4j paths**. If a fact is not in the seeded graph, GraphRAG should say so — even if PubMed would know more.

### 5.3 Configuration and lifecycle

1. Create a Neo4j Aura (or local) instance.  
2. Put URI, username, and password in the project `.env`.  
3. Seed (idempotent — safe to re-run):

   ```bash
   python scripts/seed_neo4j.py
   ```

   Re-run whenever you expand files under `data/` so Aura picks up new nodes and edges.  
4. On backend startup, CURA checks whether Neo4j is configured and reachable.  
5. At runtime:
   - **Network** prefers Neo4j; CSV fallback if unconfigured.  
   - **GraphRAG** (`POST /graphrag`) requires Neo4j.  
   - **Structured search** (`POST /graph/search`) requires Neo4j; returns paths/entities with no LLM.

**Connectivity check (from repo root, with `.env` loaded):**

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
from models.graph.neo4j_client import is_configured, verify_connectivity
print('configured:', is_configured())
print('connected:', verify_connectivity())
"
```

### 5.4 What gets seeded into the graph

| Source file | Role in seed |
|-------------|--------------|
| `data/drug_protein_disease.csv` | Creates `Drug`, `Protein`, `Disease` nodes and `BINDS`, `ASSOCIATED_WITH`, `TREATS` relationships |
| `data/sider_side_effects.csv` | Creates `SideEffect` nodes and `HAS_SIDE_EFFECT` edges |
| `data/sider_smiles.json` | Attaches SMILES strings onto Drug nodes when names match |
| Curated similar pairs in the seed script | Explicit `SIMILAR_TO` edges with scores |
| RDKit Morgan/Tanimoto over drugs that have SMILES | Extra `SIMILAR_TO` edges (threshold ≈ 0.35, capped neighbors per drug) |

**Seeding semantics:**

- Uses `MERGE` so re-runs do not duplicate Erlotinib.  
- Uniqueness constraints on entity `name` fields.  
- Demo-scale by design: expand the CSVs and re-seed to grow the graph.

**After a successful seed**, typical counts are on the order of hundreds of nodes and relationships (drugs, proteins, diseases, side effects, plus edges). Exact numbers change as you expand the CSVs.

### 5.5 Graph schema (conceptual)

```text
  (Drug)-[:BINDS]->(Protein)-[:ASSOCIATED_WITH]->(Disease)
     │                  │
     │                  └──── biology link
     ├─[:TREATS]->(Disease)          treatment / indication context
     ├─[:HAS_SIDE_EFFECT]->(SideEffect)
     └─[:SIMILAR_TO {score}]->(Drug)
```

**Node types**

| Label | Meaning | Typical properties |
|-------|---------|--------------------|
| Drug | Therapeutic / chemical entity | `name`; sometimes `smiles` |
| Protein | Molecular target | `name` |
| Disease | Indication / disease context | `name` |
| SideEffect | Adverse effect term | `name` |

**Relationship types**

| Relationship | Direction | Meaning |
|--------------|-----------|---------|
| `BINDS` | Drug → Protein | Drug interacts with / targets the protein |
| `ASSOCIATED_WITH` | Protein → Disease | Protein linked to disease biology |
| `TREATS` | Drug → Disease | Drug used in that disease context |
| `HAS_SIDE_EFFECT` | Drug → SideEffect | Catalogued adverse effect |
| `SIMILAR_TO` | Drug ↔ Drug | Similarity with a numeric `score` |

### 5.6 How the Drug Network page uses Neo4j

When you submit a drug and/or disease:

1. The knowledge-graph engine runs Cypher to pull a **local neighborhood**:
   - proteins via `BINDS`
   - diseases via `TREATS` (and protein–disease `ASSOCIATED_WITH` where relevant)
   - side effects via `HAS_SIDE_EFFECT`
   - neighbors via `SIMILAR_TO`
2. Nodes/edges are loaded into NetworkX with typed colors (drug / protein / disease / side effect).
3. PyVis writes an interactive HTML file.
4. FastAPI returns that HTML; the frontend displays it.

**Fallback:** if Neo4j is unavailable, a reduced graph is built from `drug_protein_disease.csv` only (no rich side-effect / similarity neighborhood from Aura). GraphRAG still will not work without Neo4j.

### 5.7 How GraphRAG uses Neo4j (step by step)

**GraphRAG = Graph retrieval + RAG-style generation.**

```text
User question
    → intent + entity detection (names loaded from Neo4j, cached)
    → pick Cypher template
    → run read query on Aura
    → turn records into human-readable paths + entity list
    → Groq synthesizes a short answer grounded ONLY in those paths
    → UI shows answer + evidence paths
```

#### Step 1 — Entity resolution

On first use (and periodically thereafter), CURA loads all Drug / Disease / Protein / SideEffect names from Neo4j into a short-lived cache. The question text is scanned with **longest substring / case-insensitive** matching so any seeded entity can be asked about — not only a hardcoded demo list.

#### Step 2 — Intent routing

Keywords + which entities were found select a query template:

| Intent | Typical question | What Cypher focuses on |
|--------|------------------|------------------------|
| `drug_neighborhood` | “Tell me about Erlotinib” | Bindings, treatments, side effects, similar drugs |
| `treats_disease` | “Which drugs treat lung cancer?” | `TREATS` into that disease (+ binders) |
| `disease_neighborhood` | “What is connected to melanoma?” | Treating drugs, proteins, binders |
| `protein_neighborhood` | “Which drugs bind EGFR?” | Drugs that `BINDS` the protein |
| `shared_targets` | “What shares targets with Imatinib?” | Other drugs sharing a protein |
| `similar_drugs` | “Drugs similar to Gefitinib” | `SIMILAR_TO` ordered by score |
| `side_effects` | “Side effects of Metformin” | Outgoing `HAS_SIDE_EFFECT` |
| `drugs_by_side_effect` | “Which drugs cause nausea?” | Reverse lookup from SideEffect |
| `list_drugs` | “List drugs in the graph” | Catalog with light disease/target context |
| `overview` | General graph sample | Broad sample of bind/treat paths |

#### Step 3 — Path materialization

Records become readable evidence lines, for example:

- `Erlotinib -BINDS-> EGFR`
- `Erlotinib -TREATS-> Lung cancer`
- `Erlotinib -HAS_SIDE_EFFECT-> Rash`
- `Erlotinib -SIMILAR_TO(0.72)-> Gefitinib`

These paths are the **only** factual context the LLM is allowed to use.

#### Step 4 — LLM synthesis (Groq)

Instructions to the model:

- Answer **only** using the provided Neo4j paths.  
- Cite entity names from the graph.  
- Admit when the graph lacks evidence.

If `GROQ_API_KEY` is missing, CURA returns a deterministic bullet list of paths instead of polished prose.

#### Step 5 — Response shape

`POST /graphrag` with `{ "query": "Which drugs treat lung cancer?" }` returns roughly:

- `answer` — prose (or path list fallback)
- `paths` — evidence strings
- `entities` — sorted entity names touched
- `intent` / `params` — which template ran and with which bindings

### 5.8 Structured graph search (`POST /graph/search`)

For programmatic retrieval **without** LLM synthesis:

```json
{
  "drug": "Erlotinib",
  "disease": null,
  "protein": null,
  "side_effect": null,
  "intent": "similar_drugs"
}
```

**Behavior:**

- If `intent` is set, that Cypher template runs (with defaults when a required entity is omitted).  
- If `intent` is omitted, filters choose a template (e.g. `disease` alone → `treats_disease`; nothing → `list_drugs`).  
- Response includes `intent`, `params`, `records`, `paths`, `entities`.

**Security note:** callers never send arbitrary Cypher. Only named, server-side templates run.

**Example curls** (backend must be running):

```bash
curl -s -X POST http://127.0.0.1:8002/graphrag \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which drugs treat lung cancer?"}'

curl -s -X POST http://127.0.0.1:8002/graph/search \
  -H 'Content-Type: application/json' \
  -d '{"disease":"Melanoma"}'
```

### 5.9 How Neo4j relates to SIDER / similarity features (avoiding confusion)

| Concern | Runtime path | Data origin |
|---------|--------------|-------------|
| Predict side effects from a **new SMILES** | Fingerprint neighbor transfer (`/side-effects`) | Flat SIDER CSV + SMILES JSON |
| Ask “side effects of Metformin” in GraphRAG | Neo4j `HAS_SIDE_EFFECT` | Same SIDER CSV **after seeding** |
| Find ChEMBL analogs of a SMILES | Fingerprint library (`/similar-drugs`) | `chembl_*` files |
| Ask “drugs similar to Erlotinib” in GraphRAG | Neo4j `SIMILAR_TO` | Curated pairs + RDKit pairs among seeded SMILES |

Same *domain*, different *engines*.

### 5.10 End-to-end Neo4j data flow

```text
  data/*.csv + sider_smiles.json
           │
           ▼
   scripts/seed_neo4j.py  (MERGE + constraints + SIMILAR_TO)
           │
           ▼
      Neo4j Aura
           │
     ┌─────┴──────────────┬────────────────────┐
     ▼                    ▼                    ▼
 /network            /graphrag            /graph/search
 (viz HTML)     (paths → Groq answer)   (paths, no LLM)
     ▲
     └── CSV fallback if Neo4j missing
```

### 5.11 Expanding the graph later

1. Add rows to `drug_protein_disease.csv` and/or `sider_side_effects.csv`.  
2. Optionally add SMILES to `sider_smiles.json` (enables richer fingerprint `SIMILAR_TO`).  
3. Re-run `python scripts/seed_neo4j.py`.  
4. Restart or keep using the backend — GraphRAG entity cache refreshes on a short TTL, so new names become askable automatically.

Out of scope for the current design: live pulls from ChEMBL/DrugBank APIs, and arbitrary user-supplied Cypher.

---

## 6. Example Journeys (How Pieces Fit Together)

### Journey A — Generate drugs for EGFR

1. Open **Protein Target Drug Generation**.  
2. Enter `EGFR` (or UniProt `P00533`).  
3. Backend fetches sequence from UniProt.  
4. Pipeline: structure → pocket → REINVENT → filters → binding/ADMET → rank.  
5. UI shows candidate cards (QED, affinity-like scores, etc.).  
6. Optionally open **3D Structure** for `P00533` to inspect AlphaFold’s published fold (separate from the structure used during generation).

### Journey B — Explore biology of Erlotinib (Neo4j)

1. Open **Drug Network**, enter drug `Erlotinib` (and optionally a disease).  
2. Neo4j returns the binding / treatment / side-effect / similarity neighborhood as an interactive graph.  
3. Ask GraphRAG: “What proteins does Erlotinib bind, and what side effects are linked?”  
4. Intent `drug_neighborhood` runs; paths are retrieved; Groq writes a short grounded answer.  
5. Optionally call `/graph/search` with `{ "drug": "Erlotinib", "intent": "similar_drugs" }` for raw similar-drug records.

### Journey C — Literature + graph (complementary)

1. **Research Agent** — PubMed-backed narrative on a target or molecule (broad, noisy, literature-grounded).  
2. **GraphRAG** — structured facts that exist in the seeded Neo4j demo graph (narrow, explicit, edge-grounded).  
3. Do not expect them to always agree: the graph is a curated demo subset; the literature is wider.

### Journey D — Analyze a known SMILES, then ask the graph

1. Paste a SMILES into **Molecule Analysis** for descriptors.  
2. Use **Similar Drugs** / **Side Effects** for fingerprint-based neighbors and predicted ADRs.  
3. If the molecule corresponds to a **named drug in the seed graph**, switch to Network/GraphRAG for curated relationships (targets, indications, catalogued side effects).

---

## 7. Dependencies and Degraded Modes

CURA is designed to keep working when optional pieces are missing, with reduced quality:

| Missing piece | Effect |
|---------------|--------|
| Neo4j | Network may use CSV; GraphRAG and `/graph/search` unavailable |
| Groq key | GraphRAG/research fall back to path lists or limited synthesis |
| ESMFold / structure services | Structure stage may fail or use weaker fallbacks; generation quality drops |
| DeepPurpose | Binding prefilter weakened |
| AutoDock Vina / OpenBabel | Docking scores omitted |
| ADMETlab network | More reliance on local heuristics |
| Strict filters + small sample size | Empty or very short candidate lists (often with diagnostics) |
| Backend not running | Frontend API calls fail (e.g. `curl` connection refused on `:8002`) |

---

## 8. Glossary

| Term | Meaning in CURA |
|------|-----------------|
| SMILES | Text encoding of a molecule |
| UniProt | Protein sequence / accession database |
| Pocket | Estimated ligand-binding region on a protein |
| REINVENT | Generative model that samples new SMILES |
| ADMET | Absorption, Distribution, Metabolism, Excretion, Toxicity-style profiling |
| Tox21 | Public toxicology assay suite; CURA uses a fingerprint ML screen inspired by it |
| DeepPurpose | ML framework used for protein–ligand binding prefiltering |
| Vina | Physics-based docking engine |
| Tanimoto | Similarity score between two fingerprints (0–1) |
| Cypher | Neo4j’s query language (templates only in CURA — not free-form from the UI) |
| GraphRAG | Retrieve graph paths, then generate an answer with an LLM |
| Document RAG | Retrieve text chunks (e.g. PubMed), then generate (Research Agent) |
| AlphaFold DB | Public predicted structures (Structure page) |
| ESMFold | Structure predictor used inside the **generation** pipeline when needed |
| MERGE | Neo4j upsert-style create-or-match used by the seed script |

---

## 9. Summary

- **CURA’s product surface** spans analysis, similarity, side-effect prediction, molecule generation (protein- and seed-oriented), literature research, knowledge-graph exploration, AlphaFold 3D viewing, and exports.  
- **The generation pipeline** is a multi-stage funnel: protein structure → pocket → REINVENT sampling → chemical/3D/drug-likeness filters → binding & ADMET/Tox (± docking) → multi-objective ranking.  
- **Neo4j** stores a compact biomedical graph (drugs, proteins, diseases, side effects, similarity) used for **interactive network visualization**, **GraphRAG Q&A**, and **structured `/graph/search`**. It is not the engine behind fingerprint Similar Drugs, Tox21 screening, PubMed research, or molecule generation.  
- **Two structure stories** coexist by design: AlphaFold for public 3D viewing, and PDB/ColabFold/ESMFold tiering for pocket-aware generation and docking.  
- **Two similarity / side-effect stories** also coexist: flat-file fingerprint tools for new SMILES, versus Neo4j edges for named entities in the seeded demo graph.

For setup commands and environment variables, see the project `README.md` and `.env.example`. For live API contracts, open the FastAPI interactive docs when the backend is running (for example `http://127.0.0.1:8002/docs`).
