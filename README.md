# CURA — AI Drug Generation Platform



CURA is an AI-powered computational drug discovery platform that automates the early stages of pharmaceutical research.

Given a target protein (and optionally a seed molecule), CURA:


Predicts the protein's 3D structure and identifies its binding pocket
Generates novel, drug-like molecules using AI (REINVENT, SAFE-GPT)
Predicts binding affinity and runs molecular docking (DeepPurpose, AutoDock Vina)
Screens candidates for toxicity and drug-likeness (ADMET, Tox21)
Ranks the best candidates and retrieves related research literature


Tech stack: Next.js/React/TypeScript frontend, FastAPI/Python backend, with RDKit, DeepChem, and ESMFold powering the AI pipeline.

## Repository layout

```
backend/       FastAPI service (primary API entry)
frontend/      Next.js web UI
models/        ML pipelines (generation, docking, binding, ADMET, runtime)
pipeline/      API-facing pipeline shims (migrating to models/runtime)
benchmarks/    Evaluation and benchmark suites
scripts/       Training, data prep, and one-off utilities
tools/         Research helpers (PubMed, visualization)
utils/         Shared backend utilities
lib/           Vendored external tools (P2Rank, etc.)
```

## Quick start

### Backend

```bash
conda activate cura
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:3000

## Key runtime paths

| Path | Purpose |
|------|---------|
| `models/runtime/cura_pipeline.py` | Canonical orchestration engine |
| `backend/main.py` | FastAPI routes (`/discover`, `/denovo`, etc.) |
| `result/` | Docking cache and debug output (gitignored, created at runtime) |

## External dependencies

- **AutoDock Vina** — docking executable
- **OpenBabel** (`obabel`) — ligand preparation
- **RDKit**, **PyTorch**, **DeepPurpose** — cheminformatics and binding
- **Neo4j Aura** — knowledge graph + GraphRAG (hackathon demo)

## Neo4j GraphRAG (hackathon demo)

CURA’s Drug Network page can run on a **Neo4j Aura** knowledge graph (drug → protein → disease, side effects, similarity) with a GraphRAG Q&A path.

### 1. Create Aura Free

1. Open [Neo4j Aura Console](https://console.neo4j.io) and create an **AuraDB Free** instance.
2. Download the credentials file when prompted (password is shown once).

### 2. Configure `.env`

```bash
cp .env.example .env
# Edit .env:
#   NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=...
#   GROQ_API_KEY=...   # optional but recommended for LLM synthesis
```

### 3. Install + seed

```bash
pip install neo4j   # or: pip install -r requirements.txt
python scripts/seed_neo4j.py
```

The seed script is idempotent (`MERGE` + uniqueness constraints). It loads:

- `data/drug_protein_disease.csv`
- `data/sider_side_effects.csv`
- curated + RDKit `SIMILAR_TO` edges
- curated research subgraph: `Paper` / `Author` / `Concept` / `Method` / `Limitation` / `Document` (`data/research_notes/`) with real PubMed IDs (demo provenance only — not SUPPORTS/CONTRADICTS)

### 4. Run the app

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
# frontend
cd frontend && NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

- Network + GraphRAG UI: http://localhost:3000/network
- Research Agent (PubMed document RAG): http://localhost:3000/research
- API: `POST /graphrag` `{ "query": "..." }` and `POST /graph/search` — docs at `/docs`

If Neo4j env vars are missing, `/network` falls back to the local CSV (demo should use Aura). `/graphrag` and `/graph/search` require Neo4j.

### 5. Demo script (~60s — research discovery)

1. Open **http://localhost:3000/network** (note the curated research-graph banner).
2. Click the first suggested chip (or paste):

   > Which approaches target EGFR, what limitations are recorded, and which papers discuss related drugs?

3. Confirm `intent=research_discovery`, paths like `Paper -PROPOSES-> Method -HAS_LIMITATION-> …`, authors, and PubMed links.
4. Optional second chip for classic drug multi-hop (`intent=multi_hop_research`).
5. For open PubMed surveys (not the seeded graph), use **Research Agent** at `/research`.

### 6. Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Deep dive: [`docs/CURA_PIPELINE_AND_NEO4J.md`](docs/CURA_PIPELINE_AND_NEO4J.md).


