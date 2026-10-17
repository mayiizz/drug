# CURA Full Tech Stack Inventory

This inventory is built from project manifests and Python import usage across:
- `backend/`
- `models/`
- `scripts/`
- `frontend/package.json`

## 1) Frontend Stack (Node/TypeScript)

Source: `frontend/package.json`

- Framework: `next` (Next.js)
- UI runtime: `react`, `react-dom`
- Language/tooling: `typescript`, `eslint`, `eslint-config-next`
- Styling: `tailwindcss`, `@tailwindcss/postcss`
- UI helpers: `framer-motion`, `lucide-react`, `clsx`, `tailwind-merge`
- HTTP client: `axios`
- Type packages: `@types/node`, `@types/react`, `@types/react-dom`

Install command:
- `cd frontend && npm install`

## 2) Backend API / Service Stack (Python)

Primary API:
- `fastapi`
- `uvicorn`
- `pydantic`
- `python-multipart`

Core networking / config:
- `requests`
- `httpx`
- `python-dotenv`

## 3) Drug Discovery / Chemistry Stack

Cheminformatics:
- `rdkit`
- `pubchempy`
- `chembl_webresource_client`

Scientific computing:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`

## 4) Generative Model Stack

Internal modules:
- `models/generation/reinvent4/*` (REINVENT-style SMILES LSTM RL)
- `models/generation/safe_gpt/*` (SAFE filtering/generation utilities)

Important clarification:
- REINVENT in CURA is **internal project code**, not a pip dependency named `reinvent`.

Model/runtime libs:
- `torch`
- `transformers`
- `tokenizers`
- `sentence-transformers`
- `safetensors`
- `huggingface-hub`

## 5) Binding + Docking Stack

Binding prefilter:
- `DeepPurpose`
- (sometimes needed for compatibility: `deepchem`, `dgl`, `dgllife`)

Docking orchestration modules:
- `models/docking/*` (internal)

External executables (not pip-installed):
- AutoDock Vina executable
- OpenBabel (`obabel`)
- MGLTools (`prepare_ligand4.py`, receptor prep scripts)

## 6) Research Agent Stack

Research agent modules:
- `models/research/research_agent.py`
- `models/research/rag_agent.py`

Dependencies:
- `langgraph`
- `langchain-core`
- `langchain-groq`
- `langsmith`
- `groq`
- `nltk`
- `faiss-cpu`
- `sentence-transformers`

LLM provider/model in code:
- Groq via `ChatGroq`
- Default model string: `llama-3.3-70b-versatile`

## 7) Benchmarking / Evaluation Stack

Internal benchmarking package:
- `benchmarks/*`

Dependencies:
- `matplotlib`
- `molsets` (MOSES metrics package)
- `fcd_torch` (FCD metric)

## 8) Data / Persistence / Assets

- ChEMBL CSV/TXT datasets for prior training + benchmarking
- Prior checkpoints under `checkpoints/`
- Benchmark outputs under `results/benchmarks/`
- Docking cache / temp outputs under `result/docking_cache`, `result/docking_tmp`

## 9) Single-file Dependency Install

Use:
- `pip install -r requirements.txt`

For frontend:
- `cd frontend && npm install`
