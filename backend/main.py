from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
from dotenv import load_dotenv
from pipeline.drug_discovery_pipeline import DrugDiscoveryPipeline
from pipeline.denovo_pipeline import DeNovoPipeline
from utils.protein_fetcher import fetch_protein_sequence
from models.admet.local_admet_heuristics import LocalADMETHeuristics
from models.similarity.similarity_engine import FingerprintSimilarityEngine
from models.legacy.toxicity_old.side_effect_engine import SideEffectEngine
from models.experimental.knowledge_graph_engine import KnowledgeGraphEngine
from models.utils.safe_numeric import safe_float, safe_int, safe_round
from models.utils.smiles_validation import validate_smiles_detailed

import os
import json
import logging
import sys
import importlib

load_dotenv()

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# ── Global engine refs (populated at startup, never at import time) ───────────
pipeline           = None
local_admet_heuristics = None
similarity_engine  = None
side_effect_engine = None
kg_engine          = None
denovo_pipeline    = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, local_admet_heuristics, similarity_engine, side_effect_engine, kg_engine, denovo_pipeline
    logger.info("[CURA] Initializing models (startup self-check enabled, degraded mode).")
    logger.info("[CURA] Python executable: %s", sys.executable)
    try:
        from models.graph.neo4j_client import is_configured, verify_connectivity

        if is_configured():
            if verify_connectivity():
                logger.info("[CURA] Neo4j connectivity PASS.")
            else:
                logger.warning("[CURA] Neo4j configured but connectivity FAILED.")
        else:
            logger.warning(
                "[CURA] Neo4j not configured — /network falls back to CSV; "
                "/graphrag requires NEO4J_* env vars."
            )
    except Exception as e:
        logger.warning("[CURA] Neo4j startup check WARN: %s", e)
    if "miniconda3/envs/cura" not in sys.executable.replace("\\", "/").lower():
        logger.warning(
            "[CURA] Interpreter does not appear to be conda env 'cura'. "
            "Recommended launch: python -m uvicorn backend.main:app --reload"
        )
    for mod_name in ("safe", "safe_mol"):
        try:
            mod = importlib.import_module(mod_name)
            logger.info("[CURA] Dependency check PASS: %s -> %s", mod_name, getattr(mod, "__file__", "<unknown>"))
        except Exception as e:
            logger.warning("[CURA] Dependency check WARN: %s unavailable (%s)", mod_name, e)
    try:
        from models.generation.reinvent4.checkpoint_utils import (
            checkpoint_metadata,
            resolve_best_prior_checkpoint,
            state_param_count,
            torch_load_checkpoint,
            unwrap_model_state_dict,
        )
        import torch

        resolved = resolve_best_prior_checkpoint()
        state = torch_load_checkpoint(resolved.path, torch.device("cpu"))
        sd = unwrap_model_state_dict(state)
        logger.info("[PRIOR] checkpoint=%s", resolved.path)
        logger.info("[PRIOR] params=%d", state_param_count(sd))
        logger.info("[PRIOR] vocab=%d", int(sd.get("embed.weight").shape[0]) if "embed.weight" in sd else -1)
        meta = checkpoint_metadata(state)
        if meta:
            logger.info("[PRIOR] metadata=%s", meta)
        if resolved.is_bootstrap_fallback:
            logger.warning("[PRIOR] warning: using bootstrap prior weights")
        logger.info("[CURA] REINVENT prior check PASS: %s", resolved.path)
    except Exception as e:
        logger.warning("[CURA] REINVENT prior check WARN: %s", e)
    try:
        from models.docking.config import VinaConfig, resolve_vina_executable

        try:
            vina_exe = resolve_vina_executable()
            logger.info("[VINA] Executable resolved: %s", vina_exe)
        except Exception as ex:
            logger.warning("[VINA] Executable resolution WARN: %s", ex)
        logger.info("[CURA] Docking availability: %s", VinaConfig().effective_docking_enabled())
    except Exception as e:
        logger.warning("[CURA] Docking availability check WARN: %s", e)
    try:
        from models.runtime.model_registry import preload_defaults

        preload_status = preload_defaults()
        logger.info("[CURA] Model registry preload status: %s", preload_status)
    except Exception as e:
        logger.warning("[CURA] Model registry preload WARN: %s", e)
    # ── Build the unified CuraPipeline once and share across both shims ──
    # so warmed REINVENT prior, docking caches, ESMFold client, and ADMET
    # heuristics are not duplicated.
    try:
        from models.runtime.cura_pipeline import CuraPipeline

        cura_runtime = CuraPipeline()
        logger.info("[CURA] Unified CuraPipeline runtime ready.")
    except Exception as e:
        logger.warning("[CURA] CuraPipeline init failed (%s) - falling back to per-shim runtimes.", e)
        cura_runtime = None

    pipeline           = DrugDiscoveryPipeline(cura=cura_runtime)
    local_admet_heuristics = LocalADMETHeuristics()
    similarity_engine  = FingerprintSimilarityEngine()
    side_effect_engine = SideEffectEngine()
    kg_engine          = KnowledgeGraphEngine()
    denovo_pipeline    = DeNovoPipeline(cura=cura_runtime)
    logger.info("[CURA] Startup complete — server accepting requests.")
    yield
    try:
        from models.graph.neo4j_client import close_driver

        close_driver()
    except Exception:
        pass
    logger.info("[CURA] Shutdown complete.")

app = FastAPI(title="CURA AI Drug Discovery API", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────
class DrugRequest(BaseModel):
    smiles: str

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        ok, _mol, _reason = validate_smiles_detailed(v.strip())
        if not ok:
            raise ValueError(f"Invalid SMILES string: '{v}'")
        return v.strip()


class DiscoverRequest(BaseModel):
    smiles: str
    target: str = "unknown"
    protein_sequence: str

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        ok, _mol, _reason = validate_smiles_detailed(v.strip())
        if not ok:
            raise ValueError(
                f"Invalid seed SMILES: '{v}'. Please provide the SMILES of a "
                "drug known to bind your target protein."
            )
        return v.strip()

    @field_validator("protein_sequence")
    @classmethod
    def validate_protein(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("protein_sequence is required. Fetch your protein target first.")
        if len(v) < 10:
            raise ValueError("protein_sequence is too short to be valid.")
        return v


class DeNovoRequest(BaseModel):
    gene_name:     str
    num_molecules: int = 10
    sample_size:   int = 500
    # Optional seed used as a *reward bias* (Tanimoto similarity) only. When
    # omitted (None / blank) the pipeline runs in pure protein-guided de novo
    # mode. When supplied it must be a valid RDKit-parseable SMILES.
    seed_smiles:   str | None = None

    @field_validator("gene_name")
    @classmethod
    def validate_gene_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("gene_name is required. Enter a gene name like EGFR, BRAF, CDK4.")
        if len(v) < 2:
            raise ValueError("gene_name is too short. Use a valid gene symbol like EGFR.")
        if len(v) > 20:
            raise ValueError("gene_name is too long. Use a gene symbol (e.g. EGFR) not a full name.")
        return v

    @field_validator("num_molecules")
    @classmethod
    def validate_num_molecules(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("num_molecules must be between 1 and 10.")
        return v

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, v: int) -> int:
        if v < 50 or v > 1000:
            raise ValueError("sample_size must be between 50 and 1000.")
        return v

    @field_validator("seed_smiles")
    @classmethod
    def validate_seed_smiles(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        ok, _mol, _reason = validate_smiles_detailed(v)
        if not ok:
            raise ValueError(
                f"Invalid seed_smiles: '{v}'. Provide a valid SMILES "
                "(e.g. for EGFR, the Erlotinib SMILES) or omit the field for "
                "pure protein-guided de novo generation."
            )
        return v


class ValidateRequest(BaseModel):
    smiles: str
    seed_smiles: str | None = None


class ResearchRequest(BaseModel):
    """Send ``query`` for a free-text literature survey, or ``smiles`` (+ optional target/context) for legacy molecule-style prompts."""
    query: str = ""
    smiles: str = ""
    target: str = ""
    context: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_valid_smiles(smiles: str) -> None:
    """Raise HTTP 422 if smiles is not parseable by RDKit."""
    ok, _mol, _reason = validate_smiles_detailed(smiles.strip())
    if not ok:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES string: '{smiles}'")


def _structured_error(
    *,
    error_code: str,
    stage: str,
    message: str,
    hints: list[str] | None = None,
    retriable: bool = True,
    dependency_status: dict | None = None,
) -> dict:
    return {
        "error_code": error_code,
        "stage": stage,
        "message": message,
        "hints": hints or [],
        "retriable": retriable,
        "dependency_status": dependency_status or {},
    }


def _extract_pipeline_diagnostics(shim) -> dict:
    """Best-effort extraction of CuraPipeline.last_diagnostics through a shim.

    Both ``DeNovoPipeline`` and ``DrugDiscoveryPipeline`` hold the underlying
    :class:`CuraPipeline` on ``self._cura``. We surface the diagnostics
    sidecar dict there. Falls back to an empty dict if the shim is missing,
    the pipeline failed to initialise, or the attribute does not exist yet.
    """
    if shim is None:
        return {}
    cura = getattr(shim, "_cura", None)
    if cura is None:
        return {}
    diag = getattr(cura, "last_diagnostics", None)
    if isinstance(diag, dict):
        return diag
    return {}


RESEARCH_INPUT_ERROR = (
    "Please provide a valid molecule or use a natural language query."
)


def _is_placeholder_smiles(smiles: str) -> bool:
    """True if the user left the default placeholder or an empty SMILES field."""
    s = (smiles or "").strip()
    if not s:
        return True
    low = s.lower().rstrip(".")
    if low in ("smiles string", "smiles string…", "smiles string..."):
        return True
    if low.startswith("smiles string"):
        return True
    return False
# ── Molecule image generator ───────────────────────────────────────────────
def smiles_to_image(smiles: str):
    """
    Convert SMILES → base64 PNG image using RDKit.
    Returned string can be rendered directly in <img src="">
    """
    try:
        ok, mol, reason = validate_smiles_detailed(smiles)

        if not ok or mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
            return None

        img = Draw.MolToImage(mol, size=(320, 220))

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{encoded}"

    except Exception:
        return None

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "CURA AI Drug Discovery API v2"}


@app.post("/analyze")
def analyze(data: DrugRequest):
    result = local_admet_heuristics.predict(data.smiles)
    if result is None:
        raise HTTPException(status_code=422, detail="Invalid SMILES — RDKit could not parse this molecule.")
    return result


@app.post("/similar-drugs")
def similar_drugs(data: DrugRequest):
    results   = similarity_engine.find_similar(data.smiles, k=10)
    drug_list = results.to_dict(orient="records")
    formatted = []
    for i, drug in enumerate(drug_list):
        formatted.append({
            "name":   f"ChEMBL Molecule {i+1}",
            "smiles": drug["smiles"],
            "score":  float(drug["similarity"]),
        })
    return {"similar_drugs": formatted}


@app.post("/side-effects")
def side_effects(data: DrugRequest):
    results        = side_effect_engine.predict_side_effects(data.smiles)
    sorted_effects = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_effects    = [effect for effect, _ in sorted_effects[:15]]
    return {"side_effects": top_effects}


@app.post("/network")
def get_network(data: dict):
    drug    = data.get("drug")
    disease = data.get("disease")
    drug    = drug    if drug    and drug.strip()    else None
    disease = disease if disease and disease.strip() else None
    try:
        kg_engine.visualize(drug=drug, disease=disease)
    except Exception as e:
        from models.graph.neo4j_client import Neo4jNotConfiguredError

        if isinstance(e, Neo4jNotConfiguredError):
            raise HTTPException(status_code=503, detail=str(e)) from e
        msg = str(e)
        # DNS / Aura paused / both Neo4j+CSV failed
        if (
            "Cannot resolve address" in msg
            or "Neo4j" in msg
            or "neo4j" in msg.lower()
            or "Unable to retrieve" in msg
            or "unreachable" in msg.lower()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Knowledge graph unavailable: {msg}. "
                    "If Aura DNS fails, check the instance is running at console.neo4j.io "
                    "and your network can resolve NEO4J_URI. "
                    "Network visualization falls back to CSV when Neo4j is briefly down; "
                    "GraphRAG still requires a live Neo4j connection."
                ),
            ) from e
        raise HTTPException(status_code=500, detail=f"Graph could not be generated: {msg}") from e
    if os.path.exists("drug_knowledge_graph.html"):
        with open("drug_knowledge_graph.html", "rb") as f:
            content = f.read()
        return Response(content=content, media_type="text/html")
    raise HTTPException(status_code=500, detail="Graph could not be generated.")


class GraphRAGRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("query is required")
        if len(v) > 2000:
            raise ValueError("query is too long (max 2000 characters)")
        return v


@app.post("/graphrag")
def graphrag_query(req: GraphRAGRequest):
    """Ask a natural-language question over the Neo4j drug knowledge graph."""
    try:
        from models.graph.graphrag import answer
        from models.graph.neo4j_client import Neo4jNotConfiguredError

        return answer(req.query)
    except Neo4jNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("[GraphRAG] failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"GraphRAG failed: {e}. Ensure Neo4j is seeded (scripts/seed_neo4j.py).",
        ) from e


class GraphSearchRequest(BaseModel):
    """Structured Neo4j retrieval (no LLM)."""

    drug: str | None = None
    disease: str | None = None
    protein: str | None = None
    side_effect: str | None = None
    intent: str | None = None

    @field_validator("drug", "disease", "protein", "side_effect", "intent")
    @classmethod
    def strip_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


@app.post("/graph/search")
def graph_search(req: GraphSearchRequest):
    """Structured Cypher-backed retrieval from Neo4j (paths + entities, no LLM)."""
    try:
        from models.graph.graphrag import search
        from models.graph.neo4j_client import Neo4jNotConfiguredError

        return search(
            drug=req.drug,
            disease=req.disease,
            protein=req.protein,
            side_effect=req.side_effect,
            intent=req.intent,
        )
    except Neo4jNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("[GraphSearch] failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Graph search failed: {e}. Ensure Neo4j is seeded (scripts/seed_neo4j.py).",
        ) from e


@app.get("/protein/{name}")
def get_protein(name: str):
    """
    Returns protein data with keys: name, uniprot_id, sequence.
    Raises 404 if protein not found or name is invalid.
    """
    result = fetch_protein_sequence(name)
    if result is None or "error" in result:
        raise HTTPException(
            status_code=404,
            detail=f"Protein '{name}' not found in UniProt. "
                   "Please use a valid gene name (e.g. EGFR, HER2, TP53) or UniProt accession."
        )
    if "protein" in result and "name" not in result:
        result["name"] = result.pop("protein")
    return result


@app.post("/discover")
def discover(data: DiscoverRequest):
    """
    Seed-based drug discovery pipeline:
      1. Validates seed SMILES (RDKit)
      2. Validates protein_sequence (non-empty)
      3. DeepPurpose binding pre-check on seed
      4. ChemGPT + fragment growth + REINVENT generation
      5. Drug-likeness + pharmacophore + novelty filter
      6. ChemBERTa deduplication
      7. Binding scoring of candidates
      8. ADMET filter
      9. Multi-objective ranking
    """
    logger.info(
        "[CURA] running seed-biased mode | target=%s | seq_len=%d | seed=%s",
        data.target,
        len(data.protein_sequence),
        data.smiles,
    )
    try:
        ranked = pipeline.run(
            seed_smiles=data.smiles,
            protein_sequence=data.protein_sequence,
            target=data.target,
            num_molecules=30,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected discovery error")
        raise HTTPException(status_code=500, detail=f"Discovery pipeline error: {str(e)}")

    candidates = []
    for r in ranked:
        admet = r.get("admet", {}) or {}
        binding_score = safe_float(r.get("binding_score", r.get("dp_prefilter_score")))
        kd_nm = safe_float(r.get("binding_nm", binding_score))
        docking_score = safe_float(r.get("docking_score", r.get("vina_score")))
        candidates.append({
            "smiles":        r.get("smiles", ""),
            "image":         smiles_to_image(r.get("smiles", "")),
            "activity_score": safe_float(r.get("activity_score")),
            "tox21":          r.get("tox21"),
            "admet_score":   safe_float(admet.get("drug_likeness")),
            "drug_likeness": safe_float(r.get("drug_score")),
            "binding_score": binding_score,
            "binding_nm":    kd_nm,
            "kd_nm":         kd_nm,
            "docking_score": docking_score,
            "vina_score":    safe_float(r.get("vina_score", docking_score)),
            "pose_energy":   safe_float(r.get("pose_energy", r.get("vina_score"))),
            "docking_success": r.get("docking_success"),
            "logp":          safe_float(admet.get("logp")),
            "tpsa":          safe_float(admet.get("tpsa")),
            "mol_weight":    safe_float(admet.get("molecular_weight")),
            "sa_score":      safe_float(admet.get("sa_score")),
            "qed":           safe_float(admet.get("drug_likeness")),
            "admet":         admet,
            "validation":    r.get("validation", {}),
        })

    diagnostics = _extract_pipeline_diagnostics(pipeline)
    logger.info("[CURA][RESULT] /discover returning %d candidates", len(candidates))
    return {
        "seed":        data.smiles,
        "target":      data.target,
        "candidates":  candidates,
        "diagnostics": diagnostics,
    }


@app.post("/denovo")
def denovo(data: DeNovoRequest):
    """
    De novo drug discovery — protein only, no seed molecule required.
      1. Validates gene name format
      2. Fetches + validates protein from UniProt
      3. REINVENT prior sampling (no seed — free generation)
      4. Drug-likeness + pharmacophore filter
      5. RDKit 3D conformer + MMFF94 energy filter
      6. ChemBERTa deduplication
      7. DeepPurpose neural pre-screen vs protein
      8. ADMET filter
      9. Multi-objective ranking
    """
    # Fetch and validate protein from UniProt
    protein_data = fetch_protein_sequence(data.gene_name.strip())

    if protein_data is None or "error" in protein_data:
        detail = (
            protein_data["error"]
            if isinstance(protein_data, dict) and "error" in protein_data
            else (
                f"'{data.gene_name}' not found in UniProt. "
                "Use a valid gene name like EGFR, BRAF, CDK4, TP53, HER2, KRAS, ACE2 "
                "or a UniProt accession like P00533."
            )
        )
        raise HTTPException(
            status_code=422,
            detail=_structured_error(
                error_code="DENOVO_UNIPROT_LOOKUP_FAILED",
                stage="input_validation",
                message=detail,
                hints=["Use official HGNC gene symbol", "Try a UniProt accession (e.g. P00533)"],
                retriable=True,
            ),
        )

    # Normalise key if needed
    if "protein" in protein_data and "name" not in protein_data:
        protein_data["name"] = protein_data.pop("protein")

    protein_sequence = protein_data["sequence"]
    target_name      = protein_data.get("name", data.gene_name)
    uniprot_id       = protein_data.get("uniprot_id", "")
    gene             = protein_data.get("gene", data.gene_name)

    mode = "seed-biased" if data.seed_smiles else "protein-only"
    logger.info(
        "[CURA] running %s mode | gene=%s | uniprot=%s | seq_len=%d | sample=%d | n=%d",
        mode,
        gene,
        uniprot_id,
        len(protein_sequence),
        data.sample_size,
        data.num_molecules,
    )
    if data.seed_smiles:
        logger.info("[CURA] seed_smiles=%s (used as Tanimoto reward bias only)", data.seed_smiles)

    # Run de novo pipeline
    try:
        ranked = denovo_pipeline.run(
            protein_sequence=protein_sequence,
            target=target_name,
            num_molecules=data.num_molecules,
            sample_size=data.sample_size,
            seed_smiles=data.seed_smiles,
        )
    except Exception as e:
        # Stage-aware pipeline error when available.
        if hasattr(e, "to_dict"):
            payload = getattr(e, "to_dict")()
            payload["dependency_status"] = {
                "safe_gpt": bool(getattr(denovo_pipeline, "generator", None)),
                "binding_model": bool(getattr(denovo_pipeline, "_binding_model", None)),
                "admet": bool(getattr(denovo_pipeline, "admet", None)),
            }
            raise HTTPException(status_code=422, detail=payload)
        msg = str(e)
        # Structured runtime failure mapping (degraded mode).
        if "SAFEGPT_EMPTY_GENERATION" in msg:
            raise HTTPException(
                status_code=422,
                detail=_structured_error(
                    error_code="SAFEGPT_EMPTY_GENERATION",
                    stage="safe_gpt_generation",
                    message="SAFE-GPT generated no valid candidates.",
                    hints=["Increase sample_size", "Check SAFE dependencies in active env"],
                    retriable=True,
                ),
            )
        if "safe-mol" in msg or "safe_mol" in msg or "SAFE-GPT" in msg:
            raise HTTPException(
                status_code=422,
                detail=_structured_error(
                    error_code="SAFEGPT_DEPENDENCY_UNAVAILABLE",
                    stage="dependency_init",
                    message=msg,
                    hints=[
                        "Activate conda env 'cura'",
                        "Install safe-mol in active interpreter",
                        "Launch backend with python -m uvicorn",
                    ],
                    retriable=False,
                ),
            )
        if "prior checkpoint" in msg or "smiles_lstm_prior.pt" in msg:
            raise HTTPException(
                status_code=422,
                detail=_structured_error(
                    error_code="REINVENT_PRIOR_UNAVAILABLE",
                    stage="reinvent_prior_load",
                    message=msg,
                    hints=[
                        "Ensure checkpoint exists in checkpoints/prior_best_validity.pt",
                        "Verify checkpoint readability in active interpreter",
                    ],
                    retriable=False,
                ),
            )
        if "Vina" in msg or "docking" in msg:
            raise HTTPException(
                status_code=422,
                detail=_structured_error(
                    error_code="DOCKING_RUNTIME_UNAVAILABLE",
                    stage="docking_refinement",
                    message=msg,
                    hints=["Verify Vina and preparation tools", "Use prefilter-only mode as fallback"],
                    retriable=True,
                ),
            )
        logger.exception("De novo pipeline error")
        raise HTTPException(
            status_code=500,
            detail=_structured_error(
                error_code="DENOVO_RUNTIME_ERROR",
                stage="pipeline_runtime",
                message=f"De novo pipeline error: {msg}",
                retriable=True,
            ),
        )

    diagnostics = _extract_pipeline_diagnostics(denovo_pipeline)

    if not ranked:
        # Empty results are a valid runtime outcome (filters legitimately
        # rejected everything). Return 200 with candidates=[] and a
        # diagnostics block so the frontend can explain *why* nothing made it.
        logger.info(
            "[CURA][RESULT] /denovo returning 0 candidates (drop_stage=%s)",
            (diagnostics or {}).get("drop_stage"),
        )
        return {
            "target":         target_name,
            "gene":           gene,
            "uniprot_id":     uniprot_id,
            "candidates":     [],
            "total_sampled":  data.sample_size,
            "total_returned": 0,
            "message":        "Pipeline completed but returned no candidates after filtering.",
            "diagnostics":    diagnostics,
        }

    # Build response — same shape as /discover for frontend reuse.
    # Every numeric field is run through safe_float / safe_round so a NaN /
    # Inf / None coming from optional models (DeepPurpose, Vina, Tox21,
    # ADMET) cannot break JSON serialisation.
    candidates = []
    for r in ranked:
        admet      = r.get("admet", {})
        qed        = safe_float(r.get("qed",        admet.get("drug_likeness")))
        binding_score = safe_float(r.get("binding_score", r.get("dp_prefilter_score")))
        binding_nm = safe_float(r.get("binding_nm", binding_score))
        docking_score = safe_float(r.get("docking_score", r.get("vina_score")))
        sa_score   = safe_float(r.get("sa_score",   admet.get("sa_score")))
        score = safe_float(r.get("score"))
        if score is None:
            score = safe_float(r.get("drug_score"))
        if score is None:
            val = r.get("validation", {}) or {}
            score = safe_float(val.get("composite_score"))
        logp       = safe_float(r.get("logp",       admet.get("logp")))

        # Status badge — only computed when we have all three drivers.
        if qed is not None and binding_nm is not None and sa_score is not None:
            if qed >= 0.55 and binding_nm <= 1000 and sa_score <= 4.0:
                status = "PASS"
            elif qed < 0.35 or binding_nm > 5000 or sa_score > 6.0:
                status = "WARN"
            else:
                status = "PASS"
        else:
            status = "UNKNOWN"

        # AI reasoning — guards against missing values.
        reasoning_parts = []
        if binding_nm is None:
            reasoning_parts.append(f"Binding affinity for {gene} unavailable (pre-screen skipped or failed).")
        elif binding_nm <= 100:
            reasoning_parts.append(
                f"Potent {gene} binder — BindingAffinity={binding_nm:.1f} nM indicates strong target engagement."
            )
        elif binding_nm <= 1000:
            reasoning_parts.append(
                f"Moderate {gene} binder — BindingAffinity={binding_nm:.1f} nM is within drug-like range."
            )
        else:
            reasoning_parts.append(
                f"Weak binder — BindingAffinity={binding_nm:.1f} nM may require optimization."
            )

        if logp is None:
            reasoning_parts.append("Lipophilicity unavailable.")
        elif logp < 3.0:
            reasoning_parts.append("Good aqueous solubility — low LogP favours oral bioavailability.")
        elif logp < 5.0:
            reasoning_parts.append("Balanced lipophilicity — LogP in acceptable range for cell penetration.")
        else:
            reasoning_parts.append("High lipophilicity — LogP may limit solubility.")

        if sa_score is None:
            reasoning_parts.append("Synthetic accessibility unavailable.")
        elif sa_score <= 3.0:
            reasoning_parts.append(f"Synthetically accessible — SA={sa_score:.1f} indicates straightforward synthesis.")
        elif sa_score <= 5.0:
            reasoning_parts.append(f"Moderate complexity — SA={sa_score:.1f}, feasible with standard chemistry.")
        else:
            reasoning_parts.append(f"Complex synthesis — SA={sa_score:.1f}, may require multi-step route.")

        candidates.append({
            "smiles":      r.get("smiles", ""),
            "image":       smiles_to_image(r.get("smiles", "")),
            "qed":         safe_round(qed, 2),
            "kd_nm":       safe_round(binding_nm, 1),
            "binding_nm":  safe_round(binding_nm, 1),
            "binding_score": safe_round(binding_score, 4),
            "docking_score": safe_round(docking_score, 4),
            "sa_score":    safe_round(sa_score, 1),
            "score":       safe_round((score * 100) if score is not None else None, 1),
            "mol_weight":  safe_round(r.get("mol_weight",  admet.get("molecular_weight")), 2),
            "logp":        safe_round(logp, 2),
            "tpsa":        safe_round(r.get("tpsa",        admet.get("tpsa")), 2),
            "rot_bonds":   safe_int(r.get("rot_bonds")),
            "h_donors":    safe_int(r.get("h_donors")),
            "h_acceptors": safe_int(r.get("h_acceptors")),
            "status":      status,
            "reasoning":   " ".join(reasoning_parts),
            "admet":       admet,
            "validation":  r.get("validation", {}),
            "vina_score":  safe_float(r.get("vina_score", docking_score)),
            "pose_energy": safe_float(r.get("pose_energy", r.get("vina_score"))),
            "docking_success": r.get("docking_success"),
        })

    logger.info("[CURA][RESULT] /denovo returning %d candidates", len(candidates))
    return {
        "target":         target_name,
        "gene":           gene,
        "uniprot_id":     uniprot_id,
        "candidates":     candidates,
        "total_sampled":  data.sample_size,
        "total_returned": len(candidates),
        "diagnostics":    diagnostics,
    }


@app.post("/validate")
def validate_molecule(data: ValidateRequest):
    """Validate a single molecule on all 9 drug parameters."""
    _require_valid_smiles(data.smiles)
    from models.multi_objective_ranker import _validate_molecule
    result = _validate_molecule(data.smiles, seed_smiles=data.seed_smiles)
    return result


@app.get("/export/csv")
def get_export_csv():
    file_path = "generated_molecules.csv"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
        return Response(
            content=content, 
            media_type="text/csv", 
            headers={"Content-Disposition": "attachment; filename=cura_molecules.csv"}
        )
    raise HTTPException(status_code=404, detail="No export file found. Run discovery first.")


@app.get("/export/sdf")
def get_export_sdf():
    file_path = "generated_molecules.sdf"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
        return Response(
            content=content, 
            media_type="application/octet-stream", 
            headers={"Content-Disposition": "attachment; filename=cura_molecules.sdf"}
        )
    raise HTTPException(status_code=404, detail="No export file found. Run discovery first.")


@app.post("/research")
async def run_research_agent(req: ResearchRequest):
    """
    Mode 1 — Natural language: non-empty ``query`` is used as the PubMed / RAG query.
    Mode 2 — Molecule + target: valid non-placeholder ``smiles`` → PubChem name + target + mechanism.
    """
    try:
        from models.research import research_agent
        from models.research.rag_agent import smiles_to_name, SMILES_TO_NAME
        from langchain_core.messages import HumanMessage
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=_structured_error(
                error_code="RESEARCH_DEPENDENCY_UNAVAILABLE",
                stage="research_endpoint_init",
                message=f"Research endpoint dependencies unavailable: {e}",
                hints=["Install research dependencies (langgraph/langchain stack)"],
                retriable=False,
            ),
        )

    nl = (req.query or "").strip()
    sm = (req.smiles or "").strip()
    tgt = (req.target or "").strip()

    if nl:
        query = nl
    elif sm and not _is_placeholder_smiles(sm):
        _require_valid_smiles(sm)
        name = (await smiles_to_name(sm)) or SMILES_TO_NAME.get(sm, "")
        query = f"{name} {tgt} mechanism".strip()
        if (req.context or "").strip():
            query = f"{query} {req.context.strip()}".strip()
    else:
        raise HTTPException(status_code=422, detail=RESEARCH_INPUT_ERROR)

    async def stream_agent():
        try:
            async for chunk in research_agent.astream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():
                    for msg in node_output.get("messages", []):
                        content = getattr(msg, "content", "")
                        if not content or not isinstance(content, str):
                            continue
                        yield f"data: {json.dumps({'node': node_name, 'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'node': 'error', 'content': f'Agent error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_agent(), media_type="text/event-stream")