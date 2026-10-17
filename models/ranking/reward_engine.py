"""Unified reward engine for CURA runtime and REINVENT RL.

Consolidates QED, SA, docking, pharmacophore, diversity, pose stability, and
optional seed similarity into a single composite reward. Reuses primitives
from :mod:`models.generation.reinvent4` so the trainer and the
``CuraPipeline`` share the same source of truth.

The engine is pure-function: docking, DeepPurpose, ADMET etc. are computed by
the orchestrator and passed in via :class:`RewardContext.cache`. The engine
only combines numbers, so it is safe to call from RL inner loops.

TODO: extend pharmacophore profiles beyond EGFR via :class:`PharmacophoreConfig`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from models.generation.reinvent4.config import DockingRLConfig
from models.generation.reinvent4.rewards import (
    apply_ema_smoothing,
    apply_reward_normalization,
    compose_docking_rl_reward,
    normalize_vina_score,
    sanitize_and_clip_rewards,
)
from models.generation.reinvent4.utils import morgan_fp, safe_sa
from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.egfr_pharmacophore import (
    score_egfr_pharmacophore,
    score_egfr_reference_overlap,
    score_kinase_descriptor_bias,
    score_medchem_penalty,
)
from models.pharmacophore.filtering import evaluate_smiles, select_profile
from models.utils.smiles_validation import validate_smiles_detailed

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RewardWeights:
    """Composite reward weights.

    Defaults mirror :class:`DockingRLConfig` so RL and pipeline stay aligned.
    Seed similarity is opt-in: it only contributes when a seed fingerprint is
    provided in :class:`RewardContext`.
    """

    docking: float = 0.60
    qed: float = 0.15
    sa: float = 0.10
    pharmacophore: float = 0.10
    diversity: float = 0.05
    pose_stability: float = 0.03
    seed_similarity: float = 0.0
    egfr_pharmacophore: float = 0.12
    medchem_penalty: float = 0.18
    kinase_bias: float = 0.08
    high_similarity_penalty: float = 0.0
    egfr_reference_overlap: float = 0.03
    toxicity_penalty: float = 0.08

    @classmethod
    def from_docking_rl_config(cls, cfg: DockingRLConfig) -> "RewardWeights":
        return cls(
            docking=float(cfg.docking_reward_weight),
            qed=float(cfg.qed_reward_weight),
            sa=float(cfg.sa_reward_weight),
            pharmacophore=float(cfg.pharmacophore_bonus_weight),
            diversity=float(cfg.diversity_bonus_weight),
            pose_stability=float(cfg.pose_stability_bonus_weight),
            seed_similarity=0.0,
            egfr_pharmacophore=0.12,
            medchem_penalty=0.18,
            kinase_bias=0.08,
            high_similarity_penalty=0.0,
            egfr_reference_overlap=0.03,
            toxicity_penalty=0.08,
        )

    def as_docking_rl_config(self) -> DockingRLConfig:
        """Project these weights into a :class:`DockingRLConfig` view so that
        :func:`compose_docking_rl_reward` can be reused without divergence."""
        cfg = DockingRLConfig()
        cfg.docking_reward_weight = float(self.docking)
        cfg.qed_reward_weight = float(self.qed)
        cfg.sa_reward_weight = float(self.sa)
        cfg.pharmacophore_bonus_weight = float(self.pharmacophore)
        cfg.diversity_bonus_weight = float(self.diversity)
        cfg.pose_stability_bonus_weight = float(self.pose_stability)
        return cfg


@dataclass(slots=True)
class RewardContext:
    """Per-batch evaluation context.

    The ``cache`` is a per-SMILES dict of optional precomputed values
    (``vina_score``, ``docking_success``, ``dp_score`` ...) so heavy I/O
    happens once in the orchestrator, not inside the reward engine.
    """

    protein_sequence: str = ""
    seed_smiles: str | None = None
    seed_fp: Any | None = None
    pharmacophore_cfg: PharmacophoreConfig | None = None
    pharmacophore_target: str = "EGFR"
    weights: RewardWeights = field(default_factory=RewardWeights)
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    scaffold_counts: dict[str, int] = field(default_factory=dict)
    batch_fps: dict[str, Any] = field(default_factory=dict)
    motif_counts: dict[str, int] = field(default_factory=dict)
    motif_hit_rate: float = 0.0


class RewardEngine:
    """Compute a unified composite reward per molecule.

    Use :meth:`make_context` to build a :class:`RewardContext`, then call
    :meth:`compute_reward` per SMILES or :meth:`evaluate` for a batch.
    """

    def __init__(self, *, default_weights: RewardWeights | None = None) -> None:
        self._default_weights = default_weights or RewardWeights()

    # ── Construction helpers ─────────────────────────────────────────────

    def make_context(
        self,
        *,
        protein_sequence: str = "",
        seed_smiles: str | None = None,
        weights: RewardWeights | None = None,
        pharmacophore_target: str = "EGFR",
        pharmacophore_cfg: PharmacophoreConfig | None = None,
    ) -> RewardContext:
        cfg = pharmacophore_cfg or PharmacophoreConfig(
            enabled=True,
            strictness="soft",
            enforce_hard_filters=False,
            target_name=pharmacophore_target,
        )
        seed_fp = morgan_fp(seed_smiles) if seed_smiles else None
        w = weights or self._default_weights
        if seed_fp is not None and w.seed_similarity == 0.0:
            # Default seed bias only kicks in when caller actually supplied a seed.
            w = RewardWeights(
                docking=w.docking,
                qed=w.qed,
                sa=w.sa,
                pharmacophore=w.pharmacophore,
                diversity=w.diversity,
                pose_stability=w.pose_stability,
                seed_similarity=0.2,
                egfr_pharmacophore=w.egfr_pharmacophore,
                medchem_penalty=w.medchem_penalty,
                kinase_bias=w.kinase_bias,
                high_similarity_penalty=w.high_similarity_penalty,
                egfr_reference_overlap=w.egfr_reference_overlap,
                toxicity_penalty=w.toxicity_penalty,
            )
        return RewardContext(
            protein_sequence=protein_sequence,
            seed_smiles=seed_smiles,
            seed_fp=seed_fp,
            pharmacophore_cfg=cfg,
            pharmacophore_target=pharmacophore_target,
            weights=w,
        )

    def update_scaffold_counts(self, ctx: RewardContext, smiles_list: list[str]) -> None:
        """Refresh scaffold occurrence counters from the active batch.

        Caller invokes this once per batch (or :meth:`evaluate` does it).
        """
        counts: dict[str, int] = {}
        for smi in smiles_list:
            sc = self._scaffold(smi)
            if sc:
                counts[sc] = counts.get(sc, 0) + 1
        ctx.scaffold_counts = counts
        ctx.batch_fps = {}
        for smi in smiles_list:
            fp = morgan_fp(smi)
            if fp is not None:
                ctx.batch_fps[smi] = fp

        motif_counts: dict[str, int] = {}
        motif_hits = 0
        for smi in smiles_list:
            pharm = score_egfr_pharmacophore(smi)
            matched = list(pharm.get("matched_motifs", []))
            if matched:
                motif_hits += 1
            for m in matched:
                motif_counts[m] = motif_counts.get(m, 0) + 1
        ctx.motif_counts = motif_counts
        ctx.motif_hit_rate = float(motif_hits) / float(max(1, len(smiles_list)))
        if motif_counts:
            top = sorted(motif_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "[PHARMACOPHORE] motif frequency top=%s | hit_rate=%.3f",
                top,
                ctx.motif_hit_rate,
            )

    # ── Reward computation ───────────────────────────────────────────────

    def compute_reward(self, smiles: str, context: RewardContext) -> dict[str, Any]:
        """Compute composite reward for one SMILES."""
        ok, mol, reason = validate_smiles_detailed(smiles)
        if not ok or mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
            return self._zero_reward("invalid_smiles")

        item = context.cache.get(smiles, {})
        vina_score = item.get("vina_score")
        docking_success = item.get("docking_success")
        if isinstance(vina_score, (int, float)) and (
            docking_success is True or docking_success is None
        ):
            vina_norm = normalize_vina_score(float(vina_score))
            dock_ok = bool(docking_success) if docking_success is not None else True
        else:
            vina_norm = 0.0
            dock_ok = False

        try:
            qed_val = float(QED.qed(mol))
        except Exception:
            qed_val = 0.5
        sa_norm = max(0.0, (10.0 - safe_sa(mol)) / 9.0)

        cfg = context.pharmacophore_cfg or PharmacophoreConfig(
            enabled=True,
            strictness="soft",
            enforce_hard_filters=False,
            target_name=context.pharmacophore_target,
        )
        try:
            profile = select_profile(context.pharmacophore_target, cfg)
            pharm = evaluate_smiles(smiles, profile, cfg)
            pharm_bonus = float(pharm.score)
        except Exception:
            pharm_bonus = 0.0
        motif_bonus = self._kinase_motif_bonus(mol)
        unstable_penalty = self._unstable_motif_penalty(mol)
        pharm_total_bonus = max(0.0, min(1.0, pharm_bonus + motif_bonus))
        egfr_pharm = score_egfr_pharmacophore(smiles)
        egfr_bonus = float(egfr_pharm.get("pharmacophore_score", 0.0))
        medchem = score_medchem_penalty(smiles)
        medchem_penalty = float(medchem.get("medchem_penalty", 0.0))
        kinase_bias = score_kinase_descriptor_bias(smiles)
        kinase_bias_score = float(kinase_bias.get("kinase_bias_score", 0.0))
        egfr_ref = score_egfr_reference_overlap(smiles)
        egfr_ref_overlap = float(egfr_ref.get("reference_similarity", 0.0))
        toxicity_risk = item.get("toxicity_risk")
        if toxicity_risk is None:
            toxicity_risk = item.get("toxicity")
        try:
            toxicity_penalty = max(0.0, min(1.0, float(toxicity_risk))) if toxicity_risk is not None else 0.0
        except Exception:
            toxicity_penalty = 0.0

        sc = self._scaffold(smiles)
        if sc and context.scaffold_counts:
            freq = float(context.scaffold_counts.get(sc, 1))
        else:
            freq = 1.0
        diversity_bonus = 1.0 / max(1.0, freq)
        high_similarity_penalty = 0.0
        fp_self = context.batch_fps.get(smiles)
        if fp_self is not None and context.batch_fps:
            max_tani = 0.0
            for other_smi, other_fp in context.batch_fps.items():
                if other_smi == smiles or other_fp is None:
                    continue
                try:
                    sim = float(DataStructs.TanimotoSimilarity(fp_self, other_fp))
                    if sim > max_tani:
                        max_tani = sim
                except Exception:
                    continue
            if max_tani > 0.85:
                high_similarity_penalty = min(1.0, (max_tani - 0.85) / 0.15)

        pose_stability = 0.0
        if dock_ok and isinstance(vina_score, (int, float)):
            pose_stability = min(1.0, max(0.0, ((-float(vina_score)) - 5.5) / 4.5))

        seed_similarity = 0.0
        if context.seed_fp is not None:
            try:
                fp = morgan_fp(smiles)
                if fp is not None:
                    seed_similarity = float(
                        DataStructs.TanimotoSimilarity(fp, context.seed_fp)
                    )
            except Exception:
                seed_similarity = 0.0

        primary = compose_docking_rl_reward(
            normalized_vina=vina_norm,
            qed=qed_val,
            sa_norm=sa_norm,
            pharmacophore_bonus=pharm_total_bonus,
            diversity_bonus=diversity_bonus,
            pose_stability_bonus=pose_stability,
            cfg=context.weights.as_docking_rl_config(),
        )
        total = (
            primary
            + context.weights.seed_similarity * seed_similarity
            + context.weights.egfr_pharmacophore * egfr_bonus
            + context.weights.kinase_bias * kinase_bias_score
            + context.weights.egfr_reference_overlap * egfr_ref_overlap
            - context.weights.medchem_penalty * medchem_penalty
            - context.weights.toxicity_penalty * toxicity_penalty
            - context.weights.high_similarity_penalty * high_similarity_penalty
            - unstable_penalty
        )
        logger.info(
            "[RL] pharmacophore reward=%.3f medchem_penalty=%.3f kinase_bias=%.3f high_similarity_penalty=%.3f",
            egfr_bonus,
            medchem_penalty,
            kinase_bias_score,
            high_similarity_penalty,
        )
        if context.scaffold_counts:
            unique_scaffolds = len(context.scaffold_counts)
            logger.info(
                "[RL] scaffold diversity unique=%d motif_hit_rate=%.3f",
                unique_scaffolds,
                context.motif_hit_rate,
            )

        return {
            "total_reward": float(total),
            "components": {
                "docking": float(vina_norm),
                "qed": float(qed_val),
                "sa": float(sa_norm),
                "pharmacophore": float(pharm_total_bonus),
                "kinase_motif_bonus": float(motif_bonus),
                "diversity": float(diversity_bonus),
                "pose_stability": float(pose_stability),
                "seed_similarity": float(seed_similarity),
                "unstable_motif_penalty": float(unstable_penalty),
                "egfr_pharmacophore": float(egfr_bonus),
                "medchem_penalty": float(medchem_penalty),
                "kinase_bias": float(kinase_bias_score),
                "high_similarity_penalty": float(high_similarity_penalty),
                "egfr_reference_overlap": float(egfr_ref_overlap),
                "toxicity_penalty": float(toxicity_penalty),
                "matched_motifs": list(egfr_pharm.get("matched_motifs", [])),
                "medchem_reasons": list(medchem.get("reasons", [])),
            },
            "weights": {
                "docking": context.weights.docking,
                "qed": context.weights.qed,
                "sa": context.weights.sa,
                "pharmacophore": context.weights.pharmacophore,
                "diversity": context.weights.diversity,
                "pose_stability": context.weights.pose_stability,
                "seed_similarity": context.weights.seed_similarity,
                "egfr_pharmacophore": context.weights.egfr_pharmacophore,
                "medchem_penalty": context.weights.medchem_penalty,
                "kinase_bias": context.weights.kinase_bias,
                "high_similarity_penalty": context.weights.high_similarity_penalty,
                "egfr_reference_overlap": context.weights.egfr_reference_overlap,
                "toxicity_penalty": context.weights.toxicity_penalty,
            },
            "vina_score": float(vina_score) if isinstance(vina_score, (int, float)) else None,
            "docking_success": dock_ok,
        }

    def evaluate(
        self, smiles_list: list[str], context: RewardContext
    ) -> list[dict[str, Any]]:
        """Compute composite rewards for a batch."""
        if not smiles_list:
            return []
        self.update_scaffold_counts(context, smiles_list)
        return [self.compute_reward(s, context) for s in smiles_list]

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _scaffold(smiles: str) -> str:
        ok, mol, _reason = validate_smiles_detailed(smiles)
        if not ok or mol is None:
            return ""
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            return ""

    @staticmethod
    def _zero_reward(reason: str) -> dict[str, Any]:
        return {
            "total_reward": 0.0,
            "components": {
                "docking": 0.0,
                "qed": 0.0,
                "sa": 0.0,
                "pharmacophore": 0.0,
                "kinase_motif_bonus": 0.0,
                "diversity": 0.0,
                "pose_stability": 0.0,
                "seed_similarity": 0.0,
                "unstable_motif_penalty": 0.0,
                "egfr_pharmacophore": 0.0,
                "medchem_penalty": 0.0,
                "kinase_bias": 0.0,
                "high_similarity_penalty": 0.0,
                "egfr_reference_overlap": 0.0,
                "toxicity_penalty": 0.0,
            },
            "weights": {},
            "vina_score": None,
            "docking_success": False,
            "skip_reason": reason,
        }

    @staticmethod
    def _kinase_motif_bonus(mol: Chem.Mol) -> float:
        """SMARTS-driven soft bonus for kinase-like hinge-binding chemotypes."""
        patterns = [
            ("c1cnc2ccccc2n1", 0.22),  # quinazoline-like
            ("c1nccnc1", 0.18),  # pyrimidine-like
            ("c1cc2[nH]cnc2[nH]1", 0.16),  # azaindole-like family
            ("c1ccc2ncccc2c1", 0.14),  # fused hinge-binding heteroaromatic
        ]
        bonus = 0.0
        for motif_smiles, w in patterns:
            patt = Chem.MolFromSmiles(motif_smiles)
            if patt is not None and mol.HasSubstructMatch(patt):
                bonus += w
        return float(min(0.35, bonus))

    @staticmethod
    def _unstable_motif_penalty(mol: Chem.Mol) -> float:
        """Penalty for unstable/reactive motifs that should not rank high."""
        patterns = [
            ("[N]=[N]", 0.30),  # azo
            ("[NX3]-[N]=[N]", 0.35),  # triazene
            ("[NX2]=[O]", 0.35),  # nitroso
        ]
        penalty = 0.0
        for smarts, w in patterns:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                penalty += w
        return float(min(0.8, penalty))


__all__ = [
    "RewardEngine",
    "RewardWeights",
    "RewardContext",
    "apply_ema_smoothing",
    "apply_reward_normalization",
    "sanitize_and_clip_rewards",
]
