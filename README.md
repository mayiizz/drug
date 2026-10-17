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
- demo `SIMILAR_TO` edges between related drugs

### 4. Run the app

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
# frontend
cd frontend && NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

- Network + GraphRAG UI: http://localhost:3000/network
- API: `POST /graphrag` with `{ "query": "..." }` — docs at `/docs`

If Neo4j env vars are missing, `/network` falls back to the local CSV (demo should use Aura). `/graphrag` requires Neo4j.

### 5. Demo script

1. **Aura Explore / Browser** — show `Drug`, `Protein`, `Disease` nodes and `BINDS` / `ASSOCIATED_WITH` / `TREATS` relationships.
2. **CURA `/network`** — load e.g. drug `Erlotinib`, disease `Lung cancer` and show the pyvis subgraph.
3. **GraphRAG** — ask: *Which drugs share a target with Erlotinib and what diseases are linked?* and show answer + retrieved paths.

Example Cypher (in Aura):

```cypher
MATCH (d:Drug {name: "Erlotinib"})-[:BINDS]->(p:Protein)<-[:BINDS]-(other:Drug)
MATCH (other)-[:BINDS]->(p)-[:ASSOCIATED_WITH]->(dis:Disease)
RETURN d.name, p.name, other.name, dis.name
LIMIT 25
```


