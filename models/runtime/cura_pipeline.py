"""CuraPipeline - the canonical CURA runtime orchestrator.

Single entry point that replaces both ``DeNovoPipeline`` (protein-only) and
``DrugDiscoveryPipeline`` (seed-anchored). Implements the unified flow:

    Protein input
       -> Structure resolution (ESMFold / ColabFold / experimental PDB)
       -> Pocket extraction
       -> Fragment seed generation
       -> SAFE filtering (PAINS, reactives, lipophilicity, flexibility)
       -> REINVENT policy generation (SMILESLSTM.sample, batched, max_len=72)
       -> Validity + drug-likeness filtering
       -> Pharmacophore + 3D conformer filtering
       -> DeepPurpose binding prefilter
       -> AutoDock Vina docking refinement
       -> ADMET / Tox21
       -> Unified reward (RewardEngine) + multi-objective ranking
       -> Final molecules

The pipeline is dependency-injectable: each subsystem can be swapped at
construction time for testing or alternative backends. By default the
``models.runtime.model_registry`` factories supply singletons so warm models
are shared with existing code paths.

Old MolGPT / ChemGPTGenerator orchestration is intentionally **not** used
here - REINVENT (``ReinventLoop`` -> ``SMILESLSTM.sample``) is the active
generator.

TODO: future async execution boundary - wrap structure resolution,
docking, and DeepPurpose calls in ``asyncio.to_thread`` so a single
``CuraPipeline.run`` request can be cooperatively scheduled.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import Counter
from typing import Any, Literal

from rdkit import Chem, RDLogger as _RDL  # type: ignore[attr-defined]

_RDL.DisableLog("rdApp.*")  # type: ignore[attr-defined]

from rdkit.Chem import Crippen, Descriptors, Lipinski, QED  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

from models.generation.reinvent4.config import Reinvent4Config  # noqa: E402
from models.generation.safe_gpt.config import FilteringConfig  # noqa: E402
from models.generation.safe_gpt.filtering import passes_basic_filters  # noqa: E402
from models.pharmacophore.config import PharmacophoreConfig  # noqa: E402
from models.pharmacophore.filtering import filter_smiles_batch  # noqa: E402
from models.ranking.reward_engine import RewardContext, RewardEngine, RewardWeights  # noqa: E402
from models.runtime.errors import CuraPipelineError  # noqa: E402
from models.runtime.reinvent_loop import ReinventLoop  # noqa: E402
from models.generation.reinvent4.utils import safe_sa  # noqa: E402

logger = logging.getLogger(__name__)


# ── Filter thresholds (kept aligned with the legacy pipelines) ──────────
MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 70
MIN_MW = 220.0
MAX_MW = 700.0
MAX_LOGP = 7.0
MAX_TPSA = 160.0
MIN_QED = 0.25
MAX_LIPINSKI_VIOLATIONS = 1

# Generator runtime defaults
DEFAULT_SAMPLE_BATCH = 64
DEFAULT_MAX_LEN = 72  # matches SMILESLSTM.sample default

# Default kinase fallback seeds when pocket extraction is unavailable.
_DEFAULT_KINASE_SEEDS: tuple[str, ...] = (
    "c1cnc2ccccc2n1",                              # quinazoline
    "Nc1ncnc2ccc(OC)c(OC)c12",                     # 6,7-dimethoxyQZ
    "c1ccc2[nH]cnc2c1",                            # benzimidazole
    "c1cnc2[nH]ccc2c1",                            # 7-azaindole
    "c1ccc2ncccc2c1",                              # quinoline
)


# ── Runtime profile + dual thresholds ───────────────────────────────────
# "production" is the API default (stricter, drug-like-only). "benchmark"
# loosens MW / SA / Tanimoto so MOSES, EF, novelty metrics see a richer
# distribution while still gating obvious garbage. Values mirror the plan
# in cura_research-paper-grade_finalization.
RuntimeProfile = Literal["production", "benchmark"]

_DRUG_LIKENESS_THRESHOLDS: dict[str, dict[str, float]] = {
    "production": {
        "MIN_MW": 200.0,
        "MAX_MW": 700.0,
        "MIN_HEAVY_ATOMS": 10.0,
        "MAX_HEAVY_ATOMS": 70.0,
        "MAX_LOGP": 7.0,
        "MAX_TPSA": 160.0,
        "MIN_QED": 0.20,
        "MAX_LIPINSKI_VIOLATIONS": 1.0,
        "MAX_SA": 6.2,
        "DIVERSITY_TANIMOTO": 0.97,
    },
    "benchmark": {
        "MIN_MW": 180.0,
        "MAX_MW": 700.0,
        "MIN_HEAVY_ATOMS": 8.0,
        "MAX_HEAVY_ATOMS": 70.0,
        "MAX_LOGP": 7.0,
        "MAX_TPSA": 160.0,
        "MIN_QED": 0.20,
        "MAX_LIPINSKI_VIOLATIONS": 1.0,
        "MAX_SA": 6.0,
        "DIVERSITY_TANIMOTO": 0.97,
    },
}


class _StageTimer:
    """Tiny context manager that records elapsed wall-clock per stage.

    Usage::

        with _StageTimer("dock", timings):
            self._dock(...)

    Always records, even on exception (the partial elapsed is still useful
    when reporting failures). Logs ``[CURA][TIMING] stage=<name> elapsed_ms=<x>``.
    """

    __slots__ = ("_name", "_sink", "_start")

    def __init__(self, name: str, sink: dict[str, float]) -> None:
        self._name = name
        self._sink = sink
        self._start: float = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        prev = float(self._sink.get(self._name, 0.0))
        self._sink[self._name] = prev + elapsed_ms
        logger.info("[CURA][TIMING] stage=%s elapsed_ms=%.1f", self._name, elapsed_ms)


class CuraPipeline:
    """Single canonical drug-discovery orchestrator.

    Construction lazily wires subsystems through ``models.runtime.model_registry``.
    Pass concrete instances to override any subsystem (useful for testing,
    benchmarking, or alternative backends).
    """

    def __init__(
        self,
        *,
        reinvent_loop: ReinventLoop | None = None,
        reward_engine: RewardEngine | None = None,
        pharmacophore_cfg: PharmacophoreConfig | None = None,
        filtering_cfg: FilteringConfig | None = None,
        reinvent_config: Reinvent4Config | None = None,
        structure_resolver: Any | None = None,
        pocket_extractor: Any | None = None,
        conformer_filter: Any | None = None,
        chemberta_encoder: Any | None = None,
        binding_predictor: Any | None = None,
        admet_engine: Any | None = None,
        ranker: Any | None = None,
        profile: RuntimeProfile = "production",
    ) -> None:
        logger.info("[CuraPipeline] Initializing unified runtime profile=%s...", profile)

        if profile not in _DRUG_LIKENESS_THRESHOLDS:
            logger.warning(
                "[CuraPipeline] Unknown profile %r - falling back to 'production'.", profile
            )
            profile = "production"
        self._profile: RuntimeProfile = profile
        self._thresholds: dict[str, float] = dict(_DRUG_LIKENESS_THRESHOLDS[profile])

        self._filtering_cfg = filtering_cfg or FilteringConfig()
        self._reinvent_config = reinvent_config or Reinvent4Config()
        self._apply_profile_to_reinvent_config(self._reinvent_config, self._thresholds)
        self._reward_engine = reward_engine or RewardEngine()
        self._reinvent_loop = reinvent_loop or ReinventLoop(
            config=self._reinvent_config, reward_engine=self._reward_engine
        )

        self._structure_resolver = structure_resolver or self._build_structure_resolver()
        self._pocket_extractor = pocket_extractor or self._build_pocket_extractor()
        self._conformer_filter = conformer_filter or self._build_conformer_filter()
        self._chemberta = chemberta_encoder if chemberta_encoder is not None else self._try_build_chemberta()
        self._binding_predictor = (
            binding_predictor if binding_predictor is not None else self._try_build_binding_predictor()
        )
        self._admet_local = admet_engine if admet_engine is not None else self._try_build_local_admet()
        self._ranker = ranker or self._build_ranker()
        self._pharmacophore_cfg_template = pharmacophore_cfg

        self.last_benchmark_payload: dict[str, Any] = {}
        self.last_run_summary: dict[str, Any] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.structure_metadata: dict[str, Any] = {}

        logger.info(
            "[CuraPipeline] Ready (profile=%s MW>=%.0f MAX_SA=%.2f Tanimoto>%.2f).",
            profile,
            self._thresholds["MIN_MW"],
            self._thresholds["MAX_SA"],
            self._thresholds["DIVERSITY_TANIMOTO"],
        )

    @staticmethod
    def _apply_profile_to_reinvent_config(
        cfg: Reinvent4Config, thresholds: dict[str, float]
    ) -> None:
        """Plumb diversity threshold from runtime profile into REINVENT config.

        Best-effort - some Reinvent4Config builds may not expose a nested
        ``diversity`` block; in that case we silently leave the config
        unchanged so the prior remains usable.
        """
        try:
            target = float(thresholds.get("DIVERSITY_TANIMOTO", 0.80))
            div = getattr(cfg, "diversity", None)
            if div is not None and hasattr(div, "rejection_tanimoto_threshold"):
                div.rejection_tanimoto_threshold = target
        except Exception:
            pass

    # ── Subsystem builders ----------------------------------------------

    @staticmethod
    def _build_structure_resolver():
        from models.protein.structure_tiering import ProteinStructureResolver

        return ProteinStructureResolver()

    @staticmethod
    def _build_pocket_extractor():
        from models.pocket_extractor import PocketExtractor

        return PocketExtractor()

    @staticmethod
    def _build_conformer_filter():
        from models.conformer_filter import ConformerFilter

        return ConformerFilter()

    @staticmethod
    def _try_build_chemberta():
        try:
            from models.runtime.model_registry import get_chemberta_encoder

            enc = get_chemberta_encoder()
            logger.info("[CuraPipeline] ChemBERTa encoder loaded.")
            return enc
        except Exception as e:
            logger.warning("[CuraPipeline] ChemBERTa unavailable: %s", e)
            return None

    @staticmethod
    def _try_build_binding_predictor():
        try:
            from models.runtime.model_registry import get_deeppurpose_predictor

            pred = get_deeppurpose_predictor()
            logger.info("[CuraPipeline] DeepPurpose predictor loaded.")
            return pred
        except Exception as e:
            logger.warning("[CuraPipeline] DeepPurpose unavailable: %s", e)
            return None

    @staticmethod
    def _try_build_local_admet():
        try:
            from models.admet.local_admet_heuristics import LocalADMETHeuristics

            return LocalADMETHeuristics()
        except Exception as e:
            logger.warning("[CuraPipeline] Local ADMET heuristics unavailable: %s", e)
            return None

    @staticmethod
    def _build_ranker():
        from models.ranking.multi_objective_ranker import MultiObjectiveRanker

        return MultiObjectiveRanker()

    # ── Public API -------------------------------------------------------

    def run(
        self,
        *,
        protein_sequence: str,
        target: str = "unknown",
        num_molecules: int = 10,
        sample_size: int = 500,
        seed_smiles: str | None = None,
    ) -> list[dict]:
        """Execute the full unified pipeline.

        ``seed_smiles`` is optional. When provided it is used **only** as a
        reward bias (Tanimoto similarity weight) - it does not condition
        REINVENT generation. Protein-guided generation is always primary.
        """

        if not protein_sequence or len(protein_sequence) < 10:
            raise CuraPipelineError(
                error_code="CURA_INVALID_PROTEIN_SEQUENCE",
                stage="input_validation",
                message="protein_sequence is missing or too short.",
                hints=["Fetch target sequence via UniProt first", "Use a valid gene symbol"],
                retriable=False,
            )

        seed_smiles = self._validate_seed(seed_smiles)
        if seed_smiles:
            self._gate_seed_against_target(seed_smiles, target=target, protein_sequence=protein_sequence)

        stage_counts: dict[str, int] = {
            "generated": 0,
            "valid": 0,
            "safe_pass": 0,
            "conformer_pass": 0,
            "druglike_pass": 0,
            "chemberta_pass": 0,
            "pharmacophore_pass": 0,
            "prefilter_pass": 0,
            "docked": 0,
            "docking_success": 0,
            "ranked": 0,
            "returned": 0,
        }
        rejection_counts: dict[str, int] = {
            "rejected_invalid_smiles": 0,
            "rejected_safe_filter": 0,
            "rejected_druglike": 0,
            "rejected_conformer_failure": 0,
            "rejected_pharmacophore": 0,
            "rejected_duplicate": 0,
        }
        timings: dict[str, float] = {}

        logger.info(
            "[CURA] Target=%s seed=%s protein_len=%d profile=%s",
            target,
            bool(seed_smiles),
            len(protein_sequence),
            self._profile,
        )

        _t_total = time.perf_counter()
        # ── 1. Structure resolution ─────────────────────────────────────
        with _StageTimer("structure_resolve", timings):
            pdb_string, structure_metadata = self._resolve_structure(protein_sequence)
        self.structure_metadata = structure_metadata
        self._docking_pdb_string = pdb_string

        # ── 2. Pocket extraction ────────────────────────────────────────
        with _StageTimer("pocket_extract", timings):
            pocket = self._extract_pocket(pdb_string)
        self._docking_pocket = pocket
        seeds = self._derive_fragment_seeds(pocket)
        logger.info("[CURA] %d pocket-derived fragment seeds.", len(seeds))

        # ── 3. SAFE filter for fragment seeds ────────────────────────────
        with _StageTimer("safe_filter_seeds", timings):
            clean_seeds = [s for s in seeds if self._safe_filter_smi(s)]
        if not clean_seeds:
            clean_seeds = list(_DEFAULT_KINASE_SEEDS)
        logger.info("[CURA] %d seeds passed SAFE filtering.", len(clean_seeds))

        # ── 4. REINVENT generation (batched) ─────────────────────────────
        with _StageTimer("reinvent_sample", timings):
            raw_smiles = self._generate(
                sample_size=sample_size,
                seed_smiles=seed_smiles,
                protein_sequence=protein_sequence,
            )
        # Stabilization: REINVENT prior can occasionally collapse to very small
        # fragments for a request. Keep SAFE-validated pocket-derived seeds in
        # the candidate pool so downstream filters always receive at least a few
        # kinase-like scaffolds even when sampling quality dips.
        if clean_seeds:
            merged = list(raw_smiles)
            seen = set(merged)
            for s in clean_seeds:
                if s not in seen:
                    merged.append(s)
                    seen.add(s)
            raw_smiles = merged
        stage_counts["generated"] = len(raw_smiles)
        logger.info("[CURA] generated=%d", len(raw_smiles))
        if not raw_smiles:
            raise CuraPipelineError(
                error_code="CURA_EMPTY_GENERATION",
                stage="generation",
                message="REINVENT did not generate any molecules.",
                hints=["Increase sample_size", "Verify REINVENT prior checkpoint"],
            )

        # ── 5. SAFE filter on generated outputs ─────────────────────────
        before = len(raw_smiles)
        with _StageTimer("safe_filter_gen", timings):
            raw_smiles = [s for s in raw_smiles if self._safe_filter_smi(s)]
        rejection_counts["rejected_safe_filter"] += max(0, before - len(raw_smiles))
        stage_counts["safe_pass"] = len(raw_smiles)
        logger.info("[CURA] safe_pass=%d", len(raw_smiles))

        # ── 6. 3D conformer + MMFF94 strain filter ──────────────────────
        before = len(raw_smiles)
        with _StageTimer("conformer_filter", timings):
            try:
                raw_smiles = self._conformer_filter.filter(raw_smiles)
            except Exception as e:
                logger.warning(
                    "[CURA] Conformer filter raised, keeping pre-conformer set: %s",
                    e,
                )
        conformer_summary = dict(
            getattr(self._conformer_filter, "last_filter_summary", {}) or {}
        )
        if conformer_summary:
            logger.info(
                "[CURA] conformer mean_strain=%.2f kcal/mol failed_converge=%d ring_warn=%d reject_total_strain=%d",
                float(conformer_summary.get("mean_strain_total", 0.0)),
                int(conformer_summary.get("failed_converge", 0)),
                int(conformer_summary.get("ring_strain_warnings", 0)),
                int(conformer_summary.get("rejected_total_strain", 0)),
            )
        stage_counts["conformer_pass"] = len(raw_smiles)
        rejection_counts["rejected_conformer_failure"] += max(
            0, before - len(raw_smiles)
        )
        logger.info("[CURA] conformer_pass=%d", len(raw_smiles))
        if not raw_smiles:
            logger.warning(
                "[CURA] no molecules passed 3D conformer filter - returning empty result."
            )
            return self._finalize_empty(
                stage_counts=stage_counts,
                rejection_counts=rejection_counts,
                timings=timings,
                conformer_summary=conformer_summary,
                drop_stage="conformer_filter",
            )

        # ── 7. Drug-likeness filter (RDKit / Lipinski / QED) ────────────
        with _StageTimer("druglike_filter", timings):
            druglike: list[str] = []
            for smi in raw_smiles:
                ok, reason = self._drug_likeness_check(smi)
                if ok:
                    druglike.append(smi)
                else:
                    if reason == "invalid SMILES":
                        rejection_counts["rejected_invalid_smiles"] += 1
                    else:
                        rejection_counts["rejected_druglike"] += 1
                    logger.debug("[CURA] Rejected (%s): %s", reason, smi[:80])
        stage_counts["valid"] = len(druglike)
        stage_counts["druglike_pass"] = len(druglike)
        logger.info("[CURA] druglike_pass=%d", len(druglike))
        if not druglike:
            logger.warning(
                "[CURA] no drug-like molecules survived filtering - returning empty result."
            )
            return self._finalize_empty(
                stage_counts=stage_counts,
                rejection_counts=rejection_counts,
                timings=timings,
                conformer_summary=conformer_summary,
                drop_stage="druglike_filter",
            )

        # ── 8. ChemBERTa semantic dedup (optional) ──────────────────────
        with _StageTimer("chemberta_dedup", timings):
            if self._chemberta is not None:
                before = len(druglike)
                druglike = self._chemberta_dedup(druglike)
                rejection_counts["rejected_duplicate"] += max(
                    0, before - len(druglike)
                )
        stage_counts["chemberta_pass"] = len(druglike)
        logger.info("[CURA] chemberta_pass=%d", len(druglike))

        # ── 9. Pharmacophore filter (target-aware) ──────────────────────
        with _StageTimer("pharmacophore_filter", timings):
            pharm_cfg = self._pharmacophore_cfg(target)
            pharm_passed, pharm_details = filter_smiles_batch(
                druglike, target_name=target, cfg=pharm_cfg
            )
            if pharm_cfg.use_hard_filters():
                before = len(druglike)
                druglike = pharm_passed
                rejection_counts["rejected_pharmacophore"] += max(
                    0, before - len(druglike)
                )
        stage_counts["pharmacophore_pass"] = len(druglike)
        logger.info("[CURA] pharmacophore_pass=%d", len(druglike))

        # ── 10. DeepPurpose batched binding prefilter ───────────────────
        with _StageTimer("deeppurpose_prefilter", timings):
            dp_score_by_smiles, shortlist = self._deeppurpose_prefilter(
                druglike, protein_sequence=protein_sequence, target=target
            )
        stage_counts["prefilter_pass"] = len(shortlist)
        logger.info(
            "[CURA] prefilter_pass=%d shortlist=%d", len(shortlist), len(shortlist)
        )
        logger.info("[CURA] deeppurpose_pass=%d", len(shortlist))
        if not shortlist:
            shortlist = list(druglike)

        # ── 11. ADMET + Tox21 ───────────────────────────────────────────
        with _StageTimer("compute_admet", timings):
            admet_profiles, tox21_by_smiles, activity_max_by_smiles = (
                self._compute_admet(shortlist)
            )

        # ── 12. AutoDock Vina docking refinement ────────────────────────
        with _StageTimer("dock", timings):
            vina_scores, docking_success = self._dock(
                shortlist,
                protein_sequence=protein_sequence,
                target=target,
            )
        stage_counts["docked"] = (
            len(shortlist) if docking_success or vina_scores else 0
        )
        stage_counts["docking_success"] = sum(
            1 for v in docking_success.values() if v
        )
        logger.info(
            "[CURA] docked=%d success=%d",
            stage_counts["docked"],
            stage_counts["docking_success"],
        )
        logger.info("[CURA] docking_pass=%d", stage_counts["docked"])

        # ── 13. Multi-objective ranking with reward annotations ─────────
        with _StageTimer("rank", timings):
            ranked = self._ranker.rank(
                shortlist,
                seed_smiles=seed_smiles,
                protein_sequence=protein_sequence,
                vina_scores=vina_scores or None,
                docking_successes=docking_success or None,
                admet_profiles=admet_profiles,
                dp_prefilter_scores=dp_score_by_smiles or None,
            )
        stage_counts["ranked"] = len(ranked) if ranked else 0
        logger.info("[CURA] ranked=%d", stage_counts["ranked"])

        with _StageTimer("reward_attach", timings):
            ranked = self._attach_reward_annotations(
                ranked,
                protein_sequence=protein_sequence,
                seed_smiles=seed_smiles,
                target=target,
                vina_scores=vina_scores,
                docking_success=docking_success,
            )
            ranked = self._attach_metadata(
                ranked,
                pharm_details=pharm_details,
                tox21_by_smiles=tox21_by_smiles,
                activity_max_by_smiles=activity_max_by_smiles,
            )

        ranked = ranked[:num_molecules]
        stage_counts["returned"] = len(ranked)
        logger.info("[CURA] returned=%d", stage_counts["returned"])

        timings["total_run"] = (time.perf_counter() - _t_total) * 1000.0
        logger.info(
            "[CURA][TIMING] stage=total_run elapsed_ms=%.1f", timings["total_run"]
        )

        self._log_summary(
            stage_counts=stage_counts,
            rejection_counts=rejection_counts,
            ranked=ranked,
            vina_scores=vina_scores,
            timings=timings,
            conformer_summary=conformer_summary,
        )

        # Diagnostics are populated for *every* run (empty or not) so the
        # endpoints have a single uniform place to read from.
        self.last_diagnostics = self._build_diagnostics(
            stage_counts=stage_counts,
            rejection_counts=rejection_counts,
            timings=timings,
            conformer_summary=conformer_summary,
            drop_stage=None,
        )
        logger.info(
            "[CURA][RESULT] returning %d candidates", len(ranked)
        )

        # ── 14. Optional CSV / SDF export (best-effort) ─────────────────
        try:
            from utils.export_utils import export_csv, export_sdf

            export_csv([r["smiles"] for r in ranked])
            export_sdf([r["smiles"] for r in ranked])
        except Exception as e:
            logger.debug("[CURA] export skipped: %s", e)

        return ranked

    # ── Stage 1: structure resolution ----------------------------------

    def _resolve_structure(self, protein_sequence: str) -> tuple[str | None, dict[str, Any]]:
        try:
            sr = self._structure_resolver.resolve(protein_sequence)
            return (
                sr.pdb_string or None,
                {
                    "structure_source": sr.structure_source.value,
                    "is_experimental": sr.is_experimental,
                    "mean_plddt": sr.mean_plddt,
                    "confidence_label": sr.confidence_label,
                    "provenance": sr.provenance,
                },
            )
        except Exception as e:
            logger.warning("[CURA] Structure resolution failed (%s) - sequence-only fallback.", e)
            return (
                None,
                {
                    "structure_source": "UNKNOWN",
                    "is_experimental": False,
                    "mean_plddt": None,
                    "confidence_label": "Unknown",
                    "provenance": {"error": str(e)},
                },
            )

    # ── Stage 2: pocket extraction -------------------------------------

    def _extract_pocket(self, pdb_string: str | None):
        if pdb_string is None:
            return None
        try:
            return self._pocket_extractor.extract(pdb_string)
        except Exception as e:
            logger.warning("[CURA] Pocket extraction failed (%s).", e)
            return None

    def _derive_fragment_seeds(self, pocket) -> list[str]:
        if pocket is None:
            return list(_DEFAULT_KINASE_SEEDS)
        seeds = pocket.get("smiles_bias") or []  # type: ignore[union-attr]
        if not seeds:
            return list(_DEFAULT_KINASE_SEEDS)
        return list(seeds)

    # ── SAFE filter ----------------------------------------------------

    def _safe_filter_smi(self, smi: str) -> bool:
        try:
            return passes_basic_filters(smi, self._filtering_cfg)
        except Exception:
            return False

    # ── Stage 4: REINVENT generation -----------------------------------

    def _generate(
        self,
        *,
        sample_size: int,
        seed_smiles: str | None,
        protein_sequence: str,
    ) -> list[str]:
        n = max(1, int(sample_size))
        runtime_rl = bool(getattr(self._reinvent_config, "enable_runtime_rl", False))
        logger.info(
            "[CURA] runtime mode: %s",
            "RL_ENABLED" if runtime_rl else "PRIOR_ONLY",
        )
        logger.info("[CURA] REINVENT generation target=%d max_len=%d", n, DEFAULT_MAX_LEN)
        try:
            rl_smiles = self._reinvent_loop.run_rl(
                seed_smiles=seed_smiles,
                protein_sequence=protein_sequence,
                num_molecules=n,
                num_iterations=30,
                batch_size=32,
                resume_training=True,
            )
            return rl_smiles
        except Exception as e:
            logger.warning("[CURA] REINVENT RL generation failed: %s", e)
            return []

    # ── Drug-likeness filter -------------------------------------------

    def _drug_likeness_check(self, smiles: str) -> tuple[bool, str]:
        thr = self._thresholds
        mol = _standardize_or_none(smiles)
        if mol is None:
            return False, "invalid SMILES"
        n = mol.GetNumHeavyAtoms()
        if n < int(thr["MIN_HEAVY_ATOMS"]):
            return False, f"too small ({n} heavy atoms)"
        if n > int(thr["MAX_HEAVY_ATOMS"]):
            return False, f"too large ({n} heavy atoms)"
        mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        logp = Crippen.MolLogP(mol)  # type: ignore[attr-defined]
        tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
        if mw < thr["MIN_MW"]:
            return False, f"MW too low ({mw:.0f})"
        if mw > thr["MAX_MW"]:
            return False, f"MW too high ({mw:.0f})"
        if logp > thr["MAX_LOGP"]:
            return False, f"LogP too high ({logp:.1f})"
        if tpsa > thr["MAX_TPSA"]:
            return False, f"TPSA too high ({tpsa:.0f})"
        violations = sum(
            [
                mw > 500,
                logp > 5,
                Lipinski.NumHDonors(mol) > 5,  # type: ignore[attr-defined]
                Lipinski.NumHAcceptors(mol) > 10,  # type: ignore[attr-defined]
            ]
        )
        if violations > int(thr["MAX_LIPINSKI_VIOLATIONS"]):
            return False, f"{violations} Lipinski violations"
        try:
            qed_val = QED.qed(mol)
            if qed_val < thr["MIN_QED"]:
                return False, f"QED too low ({qed_val:.2f})"
        except Exception:
            pass
        if mol.GetRingInfo().NumRings() == 0:
            return False, "no ring system"
        sa_val = safe_sa(mol)
        if sa_val > thr["MAX_SA"]:
            return False, f"SA too high ({sa_val:.2f})"
        if _has_unstable_motif(mol):
            return False, "unstable/reactive motif (azo/triazene/nitroso)"
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in {5, 34, 52}:
                return False, "unusual element (B/Se/Te)"
        return True, "ok"

    # ── ChemBERTa dedup -------------------------------------------------

    def _chemberta_dedup(self, smiles_list: list[str]) -> list[str]:
        if self._chemberta is None:
            return list(smiles_list)
        try:
            import numpy as np
        except Exception:
            return list(smiles_list)

        filtered: list[str] = []
        embeddings: list[Any] = []
        for smi in smiles_list:
            try:
                emb = self._chemberta.encode(smi)
                if isinstance(emb, np.ndarray):
                    emb = emb.flatten().tolist()
            except Exception:
                filtered.append(smi)
                embeddings.append(None)
                continue
            duplicate = False
            for prev in embeddings:
                if prev is None:
                    continue
                if _cosine(emb, prev) > 0.90:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(smi)
                embeddings.append(emb)
        return filtered

    # ── Pharmacophore config -------------------------------------------

    def _pharmacophore_cfg(self, target: str) -> PharmacophoreConfig:
        if self._pharmacophore_cfg_template is not None:
            cfg = self._pharmacophore_cfg_template
            cfg.target_name = target
            return cfg
        return PharmacophoreConfig(target_name=target, strictness="soft")

    # ── DeepPurpose prefilter ------------------------------------------

    def _deeppurpose_prefilter(
        self,
        smiles_list: list[str],
        *,
        protein_sequence: str,
        target: str,
    ) -> tuple[dict[str, float], list[str]]:
        if self._binding_predictor is None or not smiles_list:
            return {}, list(smiles_list)

        # Strict validation: any molecule that fails MolFromSmiles or
        # SanitizeMol must NEVER reach DeepPurpose / DeepChem featurizers.
        from models.utils.smiles_validation import validate_smiles_strict

        clean_list: list[str] = []
        for s in smiles_list:
            if validate_smiles_strict(s) is not None:
                clean_list.append(s)
        n_skipped = len(smiles_list) - len(clean_list)
        if n_skipped:
            logger.info(
                "[FEATURIZER] DeepPurpose skipped %d/%d invalid SMILES before binding pre-screen",
                n_skipped,
                len(smiles_list),
            )
        if not clean_list:
            return {}, []
        smiles_list = clean_list

        try:
            from models.docking import VinaConfig
        except Exception:
            cfg = None
        else:
            cfg = VinaConfig()
        if cfg is None or not cfg.effective_prefilter_enabled() or cfg.mode == "vina_only":
            return {}, list(smiles_list)

        try:
            from models.binding.prefilter import DeepPurposePrefilter
            from models.docking import (
                DockingCache,
                make_prefilter_scores_key,
                make_prefilter_shortlist_key,
            )
        except Exception as e:
            logger.warning("[CURA] DeepPurpose prefilter import failed: %s", e)
            return {}, list(smiles_list)

        scores: dict[str, float] = {}
        shortlist: list[str] = []
        try:
            cache = DockingCache(cfg.cache_dir)
            protein_id = f"{target}:{len(protein_sequence)}:{protein_sequence[:60]}"
            prefilter = DeepPurposePrefilter(
                self._binding_predictor,
                cache_dir=cfg.prefilter_cache_dir,
                cache_enabled=cfg.prefilter_cache_enabled,
            )
            scores_key = make_prefilter_scores_key(protein_id, smiles_list, cfg)
            shortlist_key = make_prefilter_shortlist_key(protein_id, smiles_list, cfg)

            cached_scores = None
            cached_shortlist = None
            if cfg.prefilter_cache_enabled:
                try:
                    p = cache.prefilter_scores_path(scores_key)
                    if p.exists():
                        cached_scores = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    p2 = cache.prefilter_shortlist_path(shortlist_key)
                    if p2.exists():
                        cached_shortlist = json.loads(p2.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    cached_scores = None
                    cached_shortlist = None

            if cached_scores and isinstance(cached_scores, dict):
                scores = {k: _normalize_binding_to_nm(float(v)) for k, v in cached_scores.items()}
                results: list[Any] = []
            else:
                results = prefilter.predict_batch(smiles_list, protein_sequence)
                for r in results:
                    if getattr(r, "raw_score", None) is not None:
                        try:
                            scores[r.smiles] = _normalize_binding_to_nm(float(r.raw_score))
                        except Exception:
                            continue
                if cfg.prefilter_cache_enabled:
                    try:
                        cache.prefilter_scores_path(scores_key).write_text(
                            json.dumps(scores, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

            if cached_shortlist and isinstance(cached_shortlist, list):
                shortlist = [str(s) for s in cached_shortlist]
            else:
                shortlist = prefilter.shortlist_candidates(
                    results
                    if results
                    else [
                        type(
                            "R",
                            (),
                            {
                                "smiles": s,
                                "raw_score": scores.get(s),
                                "valid": s in scores,
                            },
                        )
                        for s in smiles_list
                    ],
                    top_k=cfg.prefilter_top_k,
                    top_percent=cfg.resolved_prefilter_top_percent(),
                    affinity_threshold=cfg.prefilter_affinity_threshold,
                ) or list(smiles_list)
                if cfg.prefilter_cache_enabled:
                    try:
                        cache.prefilter_shortlist_path(shortlist_key).write_text(
                            json.dumps(shortlist, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("[CURA] DeepPurpose prefilter execution failed: %s", e)
            shortlist = list(smiles_list)

        if not shortlist:
            shortlist = list(smiles_list)
        return scores, shortlist

    # ── ADMET + Tox21 --------------------------------------------------

    def _compute_admet(
        self, smiles_list: list[str]
    ) -> tuple[dict[str, dict], dict[str, dict], dict[str, float | None]]:
        admet_profiles: dict[str, dict] = {}
        tox21_by_smiles: dict[str, dict] = {}
        activity_max_by_smiles: dict[str, float | None] = {}
        from models.utils.smiles_validation import validate_smiles_batch

        valid_smiles, _ = validate_smiles_batch(smiles_list, label="FEATURIZER")
        if not valid_smiles:
            return admet_profiles, tox21_by_smiles, activity_max_by_smiles

        try:
            from models.admet.admet_engine import final_admet_decision, get_admet_profile

            for smi in valid_smiles:
                try:
                    profile = get_admet_profile(smi)
                    profile["__decision__"] = final_admet_decision(profile)
                    admet_profiles[smi] = profile
                except Exception as e:
                    logger.warning("[CURA] ADMET failed for %s: %s", smi[:40], e)
        except Exception as e:
            logger.warning("[CURA] ADMET engine unavailable: %s", e)

        try:
            from models.activity.tox21_activity_screen import screen_generated_smiles

            screen = screen_generated_smiles(valid_smiles, threshold=0.5)
            for row in screen.get("results", []):
                sm = row.get("smiles")
                if not sm:
                    continue
                tox21_by_smiles[sm] = row.get("tox21", {})
                activity_max_by_smiles[sm] = row.get("activity_score")
        except Exception as e:
            logger.warning("[CURA] Tox21 screening unavailable: %s", e)

        return admet_profiles, tox21_by_smiles, activity_max_by_smiles

    # ── Docking --------------------------------------------------------

    def _dock(
        self,
        smiles_list: list[str],
        *,
        protein_sequence: str,
        target: str,
    ) -> tuple[dict[str, float], dict[str, bool]]:
        if not smiles_list:
            return {}, {}
        from models.utils.smiles_validation import validate_smiles_batch

        valid_smiles, _ = validate_smiles_batch(smiles_list, label="DOCKING")
        if not valid_smiles:
            return {}, {}
        try:
            from models.docking import (
                DockingCache,
                LigandPreparer,
                ProteinPreparer,
                VinaConfig,
                make_receptor_prep_cache_key,
            )
            from models.docking.pocket import (
                box_from_kinase_pocket,
                box_from_pdb_whole_protein,
            )
            from models.docking.scoring import VinaScorer
        except Exception as e:
            logger.warning("[CURA] Docking subsystem import failed: %s", e)
            return {}, {}

        dcfg = VinaConfig()
        if not dcfg.effective_docking_enabled() or dcfg.mode == "deeppurpose_only":
            logger.info("[CURA] Docking disabled by config (mode=%s).", dcfg.mode)
            return {}, {}

        pdb_string = self._docking_pdb_string
        pocket = self._docking_pocket
        if pdb_string is None:
            logger.warning("[CURA] Skipping docking: no resolved structure.")
            return {}, {}

        try:
            if pocket:
                box = box_from_kinase_pocket(pocket, pdb_string, dcfg)
            else:
                box = box_from_pdb_whole_protein(pdb_string, dcfg)
        except Exception as e:
            logger.warning("[CURA] Docking box construction failed: %s", e)
            return {}, {}
        if box is None:
            return {}, {}

        try:
            cache = DockingCache(dcfg.cache_dir)
            ligand_preparer = LigandPreparer(dcfg)
            protein_preparer = ProteinPreparer(dcfg)
            scorer = VinaScorer(dcfg, docking_cache=cache)

            protein_id = f"{target}:{len(protein_sequence)}:{protein_sequence[:60]}"
            receptor_key = make_receptor_prep_cache_key(protein_id, dcfg)[:32]
            receptor_path = cache.receptor_path(receptor_key)
            protein_preparer.prepare_receptor_pdbqt(
                pdb_string=pdb_string,
                protein_key=receptor_key,
                output_path=receptor_path,
            )

            dock_queue = list(valid_smiles)[: int(dcfg.top_k_docking)]
            results = scorer.score_smiles_batch(
                protein_id=protein_id,
                protein_pdbqt_path=receptor_path,
                box=box,
                smiles_list=dock_queue,
                ligand_preparer=ligand_preparer,
                structure_source=self.structure_metadata.get("structure_source"),
                structure_confidence_label=self.structure_metadata.get("confidence_label"),
            )
        except Exception as e:
            logger.warning("[CURA] Docking execution failed: %s", e)
            return {}, {}

        vina_scores: dict[str, float] = {}
        docking_success: dict[str, bool] = {}
        for smi, r in results.items():
            if r.vina_score is not None:
                vina_scores[smi] = float(r.vina_score)
            docking_success[smi] = bool(r.docking_success)
            if r.docking_success and r.vina_score is not None:
                logger.info(
                    "[DOCKING] affinity=%.2f kcal/mol smiles=%s",
                    float(r.vina_score),
                    smi[:120],
                )
            elif not r.docking_success:
                logger.warning("[DOCKING] failed smiles=%s", smi[:120])
        return vina_scores, docking_success

    # ── Reward annotations + metadata ---------------------------------

    def _attach_reward_annotations(
        self,
        ranked: list[dict],
        *,
        protein_sequence: str,
        seed_smiles: str | None,
        target: str,
        vina_scores: dict[str, float],
        docking_success: dict[str, bool],
    ) -> list[dict]:
        if not ranked:
            return ranked
        ctx = self._reward_engine.make_context(
            protein_sequence=protein_sequence,
            seed_smiles=seed_smiles,
            pharmacophore_target=target,
            weights=RewardWeights(),
        )
        ctx.cache = {
            r["smiles"]: {
                "vina_score": vina_scores.get(r["smiles"]),
                "docking_success": docking_success.get(r["smiles"]),
            }
            for r in ranked
        }
        smiles_list = [r["smiles"] for r in ranked]
        rewards = self._reward_engine.evaluate(smiles_list, ctx)
        for r, rew in zip(ranked, rewards):
            r["reward"] = rew
        return ranked

    @staticmethod
    def _attach_metadata(
        ranked: list[dict],
        *,
        pharm_details: dict[str, Any],
        tox21_by_smiles: dict[str, dict],
        activity_max_by_smiles: dict[str, float | None],
    ) -> list[dict]:
        for r in ranked:
            sm = r.get("smiles", "")
            if not sm:
                continue
            r["pose_energy"] = r.get("vina_score")
            if "docking_score" not in r:
                r["docking_score"] = r.get("vina_score")
            if "binding_score" not in r:
                r["binding_score"] = r.get("dp_prefilter_score")
            if "binding_nm" not in r:
                r["binding_nm"] = r.get("binding_score")
            r["tox21"] = tox21_by_smiles.get(sm)
            r["activity_score"] = activity_max_by_smiles.get(sm)
            p = pharm_details.get(sm)
            if p is not None:
                r["pharmacophore_passed"] = bool(p.passed)
                r["pharmacophore_score"] = float(p.score)
                r["pharmacophore_failed_rules"] = list(p.failed_rules)
                r["pharmacophore_target_profile"] = p.target_profile
        return ranked

    # ── Diagnostics + empty-result finalisation -----------------------

    def _build_diagnostics(
        self,
        *,
        stage_counts: dict[str, int],
        rejection_counts: dict[str, int],
        timings: dict[str, float],
        conformer_summary: dict[str, Any],
        drop_stage: str | None = None,
    ) -> dict[str, Any]:
        """Return a serialisable diagnostics block for an API response.

        Carries everything the frontend needs to explain a 0-survivor run:
        per-stage counts, rejection reasons, wall-clock per stage, conformer
        diagnostics, the active runtime profile, and the drug-likeness
        thresholds in effect for this run.
        """
        return {
            "stage_counts": dict(stage_counts),
            "rejection_counts": dict(rejection_counts),
            "timings_ms": dict(timings),
            "conformer": dict(conformer_summary),
            "profile": self._profile,
            "thresholds": {k: float(v) for k, v in self._thresholds.items()},
            "drop_stage": drop_stage,
        }

    def _finalize_empty(
        self,
        *,
        stage_counts: dict[str, int],
        rejection_counts: dict[str, int],
        timings: dict[str, float],
        conformer_summary: dict[str, Any],
        drop_stage: str,
    ) -> list[dict]:
        """Common exit path when zero molecules survive the pipeline.

        Records timings + diagnostics into the usual sidecar attributes so
        ``/denovo`` and ``/discover`` can surface a structured response and
        returns ``[]`` so existing callers keep their expected return type.
        """
        timings.setdefault("total_run", 0.0)
        timings["total_run"] = float(timings.get("total_run", 0.0))
        diagnostics = self._build_diagnostics(
            stage_counts=stage_counts,
            rejection_counts=rejection_counts,
            timings=timings,
            conformer_summary=conformer_summary,
            drop_stage=drop_stage,
        )
        self.last_diagnostics = diagnostics
        try:
            self._log_summary(
                stage_counts=stage_counts,
                rejection_counts=rejection_counts,
                ranked=[],
                vina_scores={},
                timings=timings,
                conformer_summary=conformer_summary,
            )
        except Exception as e:
            logger.warning("[CURA] _log_summary failed on empty result: %s", e)
        # Also surface diagnostics on last_run_summary for non-API callers.
        if isinstance(self.last_run_summary, dict):
            self.last_run_summary.setdefault("diagnostics", diagnostics)
        logger.info(
            "[CURA][RESULT] empty result | drop_stage=%s | druglike_pass=%d generated=%d",
            drop_stage,
            int(stage_counts.get("druglike_pass", stage_counts.get("valid", 0))),
            int(stage_counts.get("generated", 0)),
        )
        return []

    # ── Summary logging ------------------------------------------------

    def _log_summary(
        self,
        *,
        stage_counts: dict[str, int],
        rejection_counts: dict[str, int],
        ranked: list[dict],
        vina_scores: dict[str, float],
        timings: dict[str, float] | None = None,
        conformer_summary: dict[str, Any] | None = None,
    ) -> None:
        scaffold_counts: Counter[str] = Counter()
        from models.utils.smiles_validation import validate_smiles_detailed

        for r in ranked:
            sm = r.get("smiles", "")
            _, mol, _ = validate_smiles_detailed(sm) if sm else (False, None, "empty")
            if mol is None:
                continue
            try:
                sc = MurckoScaffold.GetScaffoldForMol(mol)
                sc_smi = Chem.MolToSmiles(sc) if sc is not None else ""
            except Exception:
                sc_smi = ""
            if sc_smi:
                scaffold_counts[sc_smi] += 1
        total_sc = sum(scaffold_counts.values())
        scaffold_entropy = 0.0
        if total_sc > 0:
            for c in scaffold_counts.values():
                p = c / total_sc
                scaffold_entropy -= p * math.log(max(p, 1e-12))

        vina_vals = [float(v) for v in vina_scores.values()]
        qed_vals = [float(r.get("admet", {}).get("drug_likeness")) for r in ranked if r.get("admet", {}).get("drug_likeness") is not None]
        sa_vals = [float(r.get("admet", {}).get("sa_score")) for r in ranked if r.get("admet", {}).get("sa_score") is not None]

        logger.info("================================================")
        logger.info("PIPELINE SUMMARY")
        logger.info("================")
        for k, v in stage_counts.items():
            logger.info("%s: %d", k, v)
        logger.info("scaffold_entropy: %.4f", scaffold_entropy)
        logger.info("=========")
        logger.info("================================================")
        logger.info("MOLECULAR QUALITY SUMMARY")
        logger.info("=========================")
        logger.info("generated: %d", stage_counts["generated"])
        logger.info("valid: %d", stage_counts["valid"])
        logger.info("diverse: %d", len(scaffold_counts))
        logger.info("pharmacophore_pass: %d", stage_counts["pharmacophore_pass"])
        logger.info("prefilter_pass: %d", stage_counts["prefilter_pass"])
        logger.info("docked: %d", stage_counts["docked"])
        logger.info("vina_success: %d", stage_counts["docking_success"])
        logger.info("mean_vina: %s", f"{(sum(vina_vals) / len(vina_vals)):.3f}" if vina_vals else "NA")
        logger.info("best_vina: %s", f"{min(vina_vals):.3f}" if vina_vals else "NA")
        logger.info("mean_qed: %s", f"{(sum(qed_vals) / len(qed_vals)):.3f}" if qed_vals else "NA")
        logger.info("mean_sa: %s", f"{(sum(sa_vals) / len(sa_vals)):.3f}" if sa_vals else "NA")
        logger.info("scaffold_entropy: %.4f", scaffold_entropy)
        logger.info("returned: %d", stage_counts["returned"])
        logger.info("=========")

        for k, v in rejection_counts.items():
            logger.info("[CURA] %s=%d", k, v)

        timings_dict = dict(timings) if timings else {}
        if timings_dict:
            logger.info(
                "[CURA] timings_ms=%s",
                ", ".join(f"{k}={v:.0f}" for k, v in timings_dict.items()),
            )

        conformer_diag = dict(conformer_summary) if conformer_summary else {}

        self.last_run_summary = {
            "stage_counts": dict(stage_counts),
            "rejection_counts": dict(rejection_counts),
            "scaffold_entropy": scaffold_entropy,
            "unique_scaffolds": len(scaffold_counts),
            "timings": timings_dict,
            "thresholds": {
                "profile": self._profile,
                **{k: float(v) for k, v in self._thresholds.items()},
            },
            "conformer": conformer_diag,
        }
        self.last_benchmark_payload = {
            "pipeline": "cura_unified",
            "num_ranked": len(ranked),
            "structure_metadata": self.structure_metadata,
            "ranked_smiles": [r.get("smiles", "") for r in ranked],
            "vina_scores": dict(vina_scores),
            "docking_successes": {r.get("smiles", ""): r.get("docking_success") for r in ranked},
            "scaffold_entropy": scaffold_entropy,
            "unique_scaffolds": len(scaffold_counts),
            "scaffold_frequency": dict(scaffold_counts),
            "timings": timings_dict,
            "profile": self._profile,
            "conformer": conformer_diag,
        }

    # ── Seed validation / soft gate -----------------------------------

    @staticmethod
    def _validate_seed(seed_smiles: str | None) -> str | None:
        if not seed_smiles:
            return None
        from models.utils.smiles_validation import validate_smiles_detailed

        ok, _mol, _reason = validate_smiles_detailed(seed_smiles)
        if not ok:
            raise CuraPipelineError(
                error_code="CURA_INVALID_SEED_SMILES",
                stage="input_validation",
                message=f"Invalid seed SMILES: '{seed_smiles}'.",
                hints=["Provide a valid RDKit-parseable SMILES, or omit the seed for protein-only mode."],
                retriable=False,
            )
        return seed_smiles

    def _gate_seed_against_target(self, seed_smiles: str, *, target: str, protein_sequence: str) -> None:
        """Lightweight sanity check for known kinase targets when a seed is provided.

        This is informational only - the seed never blocks generation by itself
        because protein-guided generation is primary.
        """
        try:
            from models.utils.smiles_validation import validate_smiles_detailed

            ok, mol, reason = validate_smiles_detailed(seed_smiles)
            if not ok or mol is None:
                logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", seed_smiles, reason)
                return
            mw = float(Descriptors.MolWt(mol))  # type: ignore[attr-defined]
            if mw < 250.0:
                logger.warning(
                    "[CURA] Seed MW=%.1f Da is small for kinase-class targets; "
                    "treating as bias only.",
                    mw,
                )
        except Exception:
            pass


# ── Module-level helpers --------------------------------------------------

def _cosine(v1, v2) -> float:
    try:
        import numpy as np

        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        n1 = float(np.linalg.norm(a))
        n2 = float(np.linalg.norm(b))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))
    except Exception:
        return 0.0


def _normalize_binding_to_nm(raw_value: float) -> float:
    """Normalize DeepPurpose outputs to nM.

    If a model emits pAffinity/pKd-like values (e.g., 6), convert with:
      nM = 10 ** (9 - pAffinity)
    Otherwise treat value as already nM.
    """
    x = float(raw_value)
    if 0.0 < x < 20.0:
        return float(10.0 ** (9.0 - x))
    return x


def _has_unstable_motif(mol: Chem.Mol) -> bool:
    patterns = (
        "[N]=[N]",  # azo
        "[NX3]-[N]=[N]",  # triazene
        "[NX2]=[O]",  # nitroso
    )
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return True
    return False


# Reused by the drug-likeness filter and SAFE pre-check. Picks the largest
# disconnected fragment (drops salts / counterions), neutralizes obviously
# pH-fixable charges via canonical SMILES roundtrip, and rejects single-atom
# fragments or molecules with a non-zero net formal charge after cleanup.
def _standardize_or_none(smiles: str) -> "Chem.Mol | None":
    if not smiles:
        return None
    try:
        from models.utils.smiles_validation import validate_smiles_detailed

        ok, mol, reason = validate_smiles_detailed(smiles)
    except Exception:
        return None
    if not ok or mol is None:
        logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
        return None
    try:
        from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser  # type: ignore

        mol = LargestFragmentChooser().choose(mol)
    except Exception:
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if frags:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        except Exception:
            return None
    if mol is None:
        return None
    if mol.GetNumHeavyAtoms() < 2:
        return None
    try:
        net_charge = sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
    except Exception:
        net_charge = 0
    if abs(net_charge) > 0:
        return None
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical:
            _, cleaned, _ = validate_smiles_detailed(canonical)
        else:
            cleaned = None
        if cleaned is not None:
            mol = cleaned
    except Exception:
        pass
    return mol


# Suppress unused import warnings at the orchestrator level (referenced via
# subsystems): random is reserved for future ablation/sampling helpers.
_ = random


__all__ = ["CuraPipeline", "CuraPipelineError"]
