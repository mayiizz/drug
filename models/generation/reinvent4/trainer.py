"""
Core REINVENT-style docking-guided RL: prior–agent augmented likelihood with
docking-in-the-loop optimization.
Not PPO: loss = mean((log π_agent - log π_prior - σ * r_aligned)²) per trajectory batch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.checkpoint_utils import checkpoint_metadata, state_param_count
from models.generation.reinvent4.config import DiversityConfig, Reinvent4Config
from models.generation.reinvent4.curriculum import CurriculumScheduler
from models.generation.reinvent4.diversity import RollingDiversityMemory, select_diverse_top
from models.generation.reinvent4.experience import ReplayBuffer
from models.generation.reinvent4.logging_utils import log_rl_iteration
from models.generation.reinvent4.prior import build_and_load_prior, prior_log_prob
from models.generation.reinvent4.rewards import (
    apply_ema_smoothing,
    apply_reward_normalization,
    sanitize_and_clip_rewards,
)
from models.generation.reinvent4.sampler import diversity_acceptance_filter, sample_smiles_batch
from models.generation.reinvent4.scoring import CompositeScorer
from models.generation.reinvent4.utils import encode_batch_aligned, morgan_fp
from models.generation.reinvent4.vocab import SMILESVocabulary
from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.filtering import filter_smiles_batch
from models.ranking.reward_engine import RewardEngine

logger = logging.getLogger(__name__)


class ReinventTrainer:
    def __init__(self, config: Reinvent4Config | None = None):
        self.config = config or Reinvent4Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = SMILESVocabulary()
        logger.info("Initializing REINVENT SMILES-LSTM generator...")
        self.prior = build_and_load_prior(self.vocab, self.device, self.config)
        self.prior.train(False)
        self.scorer = CompositeScorer(self.config)
        self.curriculum = CurriculumScheduler(self.config)
        self._reward_ema_mean: float | None = None
        self.last_training_summary: dict[str, Any] = {}
        self.diversity_memory = RollingDiversityMemory(
            buffer_size=int(self.config.diversity.memory_buffer_size),
            tanimoto_threshold=float(self.config.diversity.rejection_tanimoto_threshold),
        )
        self.reward_engine = RewardEngine()
        self.agent = SMILESLSTM(
            len(self.vocab),
            embed_dim=self.config.lstm_embed_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout,
        ).to(self.device)
        self.agent.load_state_dict(self.prior.state_dict())
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay)
        self.rl_metrics_history: list[dict[str, Any]] = []
        self._dock_context_cache: dict[str, dict[str, Any]] = {}
        logger.info("REINVENT generator ready on %s.", self.device)

    def _dock_shortlist(
        self,
        *,
        smiles_list: list[str],
        protein_sequence: str,
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """
        Dock a shortlist with VinaScorer and return score/success maps.
        Failures are non-fatal: caller applies fallback reward.
        """
        if not smiles_list or not protein_sequence:
            return {}, {}
        logger.info("[VINA] Docking started stage=reinvent shortlist=%d", len(smiles_list))
        try:
            from models.docking import (
                DockingCache,
                LigandPreparer,
                ProteinPreparer,
                VinaConfig,
                make_receptor_prep_cache_key,
            )
            from models.docking.pocket import box_from_pdb_whole_protein, box_from_pocket
            from models.docking.scoring import VinaScorer
            from models.pocket_extractor import PocketExtractor
            from models.runtime.model_registry import get_esmfold_client
        except Exception as e:
            logger.warning("Docking modules unavailable: %s", e)
            return {}, {}

        dcfg = VinaConfig()
        dcfg.docking_timeout = int(self.config.docking_rl.docking_timeout)
        if not dcfg.effective_docking_enabled():
            return {}, {}

        try:
            docking_cache = DockingCache(dcfg.cache_dir)
            ligand_preparer = LigandPreparer(dcfg)
            protein_preparer = ProteinPreparer(dcfg)
            vina_scorer = VinaScorer(dcfg, docking_cache=docking_cache)

            protein_id = f"rl:{len(protein_sequence)}:{protein_sequence[:60]}"
            ctx = self._dock_context_cache.get(protein_id)
            if ctx is None:
                logger.info("[RL] building docking context cache for protein_id=%s", protein_id)
                receptor_key = make_receptor_prep_cache_key(protein_id, dcfg)[:32]
                receptor_pdbqt_path = docking_cache.receptor_path(receptor_key)
                esm_fold = get_esmfold_client()
                pdb_string = esm_fold.fold(protein_sequence)
                pocket = None
                try:
                    pe = PocketExtractor()
                    pocket = pe.extract(pdb_string)
                except Exception:
                    pocket = None
                box = box_from_pocket(pocket, dcfg) if pocket else box_from_pdb_whole_protein(pdb_string, dcfg)
                protein_preparer.prepare_receptor_pdbqt(
                    pdb_string=pdb_string,
                    protein_key=receptor_key,
                    output_path=receptor_pdbqt_path,
                )
                ctx = {
                    "receptor_pdbqt_path": receptor_pdbqt_path,
                    "box": box,
                }
                self._dock_context_cache[protein_id] = ctx
            else:
                logger.info("[RL] reusing cached docking context for protein_id=%s", protein_id)
            receptor_pdbqt_path = ctx["receptor_pdbqt_path"]
            box = ctx["box"]
            dock_results = vina_scorer.score_smiles_batch(
                protein_id=protein_id,
                protein_pdbqt_path=receptor_pdbqt_path,
                box=box,
                smiles_list=smiles_list,
                ligand_preparer=ligand_preparer,
                structure_source="ESMFOLD",
                structure_confidence_label="Unknown",
            )
        except Exception as e:
            logger.warning("Docking execution failed: %s", e)
            return {}, {}

        vina_scores: dict[str, float] = {}
        docking_success: dict[str, bool] = {}
        for smi, r in dock_results.items():
            if r.vina_score is not None:
                vina_scores[smi] = float(r.vina_score)
            docking_success[smi] = bool(r.docking_success)
            if r.docking_success:
                logger.info("[VINA] Docking completed smiles=%s", smi)
                if r.vina_score is not None:
                    logger.info("[VINA] Affinity=%.4f kcal/mol smiles=%s", float(r.vina_score), smi)
            else:
                logger.warning("[VINA] Docking failed: smiles=%s", smi)
        logger.info(
            "[VINA] Pipeline summary stage=reinvent attempted=%d successful=%d failed=%d mean_affinity=%s best_affinity=%s",
            len(smiles_list),
            sum(1 for s in smiles_list if docking_success.get(s, False)),
            max(0, len(smiles_list) - sum(1 for s in smiles_list if docking_success.get(s, False))),
            f"{(sum(vina_scores.values()) / len(vina_scores)):.3f}" if vina_scores else "NA",
            f"{min(vina_scores.values()):.3f}" if vina_scores else "NA",
        )
        return vina_scores, docking_success

    def generate(
        self,
        seed_smiles: str,
        num_molecules: int = 20,
        protein_sequence: str = "",
        binding_model: Any = None,
        *,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        checkpoint_interval: int | None = None,
        resume_training: bool = False,
    ) -> list[str]:
        cfg = self.config

        seed_fp = morgan_fp(seed_smiles)
        if seed_fp is None:
            raise ValueError(f"Invalid seed SMILES: {seed_smiles}")

        agent = self.agent
        optimizer = self.optimizer
        replay = self.replay
        before_mean = float(next(agent.parameters()).detach().mean().item())
        print(f"[RL-CHECK] agent_param_mean_before={before_mean:.10f}")
        if learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)

        if resume_training:
            ckpt_dir = cfg.checkpoint.checkpoint_dir
            if ckpt_dir:
                ckpt_root = Path(ckpt_dir)
                latest = None
                if ckpt_root.exists():
                    pts = sorted(ckpt_root.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
                    if pts:
                        latest = pts[-1]
                if latest is not None:
                    from models.generation.reinvent4 import checkpointing as ckpt

                    state = ckpt.load_checkpoint(latest, map_location=self.device)
                    if "agent_state_dict" in state:
                        agent.load_state_dict(state["agent_state_dict"])
                        logger.info("[PRIOR] checkpoint=%s", latest)
                        logger.info("[PRIOR] params=%d", state_param_count(state["agent_state_dict"]))
                        logger.info("[PRIOR] vocab=%d", len(self.vocab))
                        md = checkpoint_metadata(state)
                        if md:
                            logger.info("[PRIOR] metadata=%s", md)
                    if state.get("optimizer_state_dict"):
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                    if state.get("replay"):
                        try:
                            replay.load_state_dict(state["replay"])
                        except Exception as e:
                            logger.warning("[RL] replay restore failed: %s", e)
                    if state.get("reward_state"):
                        self._reward_ema_mean = state["reward_state"].get("ema_mean")
                    if state.get("diversity_state"):
                        try:
                            self.diversity_memory.load_state_dict(state["diversity_state"])
                        except Exception as e:
                            logger.warning("[RL] diversity restore failed: %s", e)
                    logger.info("[RL] resumed training from checkpoint %s", latest)
                    logger.info("[RL] loaded checkpoint: %s", latest)

        all_generated: dict[str, float] = {}
        all_generated_before: dict[str, float] = {}

        n_iterations = int(num_iterations if num_iterations is not None else cfg.rl_iterations)
        bs = int(batch_size if batch_size is not None else cfg.batch_size)
        ckpt_every = int(
            checkpoint_interval
            if checkpoint_interval is not None
            else cfg.checkpoint.auto_save_every
        )

        logger.info(
            "[REINVENT] Starting RL loop iterations=%d batch=%d",
            n_iterations,
            bs,
        )
        logger.info(
            "[REINVENT] Protein sequence: %s",
            f"{len(protein_sequence or '')} aa" if protein_sequence else "empty (binding reward neutral)",
        )
        logger.info("[REINVENT] Binding model: %s", "yes" if binding_model else "no (neutral binding)")
        logger.info("[REINVENT] Augmented-likelihood objective (REINVENT-style), not PPO.")

        try:
            from rdkit import RDLogger as _RDL
            _RDL.DisableLog("rdApp.*")  # type: ignore[attr-defined]
        except Exception:
            pass

        for it in range(n_iterations):
            try:
                cw = self.curriculum.weights(it)

                agent.eval()
                with torch.no_grad():
                    sampled_smiles, token_ids = sample_smiles_batch(
                        agent,
                        self.vocab,
                        bs,
                        device=self.device,
                        max_len=cfg.max_sample_length,
                        temperature=cfg.sample_temperature,
                        greedy=cfg.greedy_sampling,
                    )
                sampled_count = len(sampled_smiles) if isinstance(sampled_smiles, list) else 0

                if token_ids is None or not hasattr(token_ids, "shape"):
                    logger.warning("Iter %s: token_ids invalid — skipping.", it)
                    continue
                if token_ids.shape[0] == 0 or token_ids.shape[1] < 2:
                    logger.warning("Iter %s: token_ids empty — skipping.", it)
                    continue
                if not sampled_smiles or not isinstance(sampled_smiles, list):
                    logger.warning("Iter %s: sampled_smiles invalid — skipping.", it)
                    continue
                if it == 0:
                    for s in sampled_smiles:
                        if Chem.MolFromSmiles(s) is not None and len(s) > 5:
                            all_generated_before[s] = all_generated_before.get(s, 0.0)

                # Diversity-aware filtering before docking/replay insertion.
                filtered_smiles, _rej = diversity_acceptance_filter(
                    sampled_smiles,
                    self.diversity_memory,
                    threshold=float(cfg.diversity.rejection_tanimoto_threshold),
                )
                if not filtered_smiles:
                    logger.warning("Iter %s: all sampled molecules rejected by diversity filter.", it)
                    continue
                diversity_survivors = len(filtered_smiles)

                # 1) DeepPurpose shortlist acceleration (optional)
                top_k = max(1, int(cfg.docking_rl.docking_top_k))
                pharm_cfg = PharmacophoreConfig(
                    enabled=True,
                    strictness="hard",
                    enforce_hard_filters=True,
                    target_name="EGFR",
                )
                pharm_passed, pharm_details = filter_smiles_batch(
                    filtered_smiles,
                    target_name=pharm_cfg.target_name,
                    cfg=pharm_cfg,
                )
                pharmacophore_survivors = len(pharm_passed)
                shortlist, dp_scores = self.scorer.prefilter_shortlist(
                    pharm_passed,
                    binding_model=binding_model,
                    protein_sequence=protein_sequence or "",
                    top_k=top_k,
                )
                prefilter_survivors = len(shortlist)

                # 2) Dock shortlist at configured frequency.
                vina_scores: dict[str, float] = {}
                docking_success: dict[str, bool] = {}
                if cfg.docking_rl.enable_docking_rl and ((it % max(1, cfg.docking_rl.docking_frequency)) == 0):
                    vina_scores, docking_success = self._dock_shortlist(
                        smiles_list=shortlist,
                        protein_sequence=protein_sequence or "",
                    )

                rewards_raw, _, reward_meta = self.scorer.docking_reward_vector(
                    filtered_smiles,
                    vina_scores=vina_scores,
                    docking_successes=docking_success,
                )
                # Diversity-aware scaffold family balancing:
                # softly reduce reward for over-represented scaffolds and
                # encourage underrepresented families from rolling memory.
                alpha = float(cfg.diversity.scaffold_balance_alpha)
                if alpha > 0.0:
                    adjusted = rewards_raw.tolist()
                    for i, smi in enumerate(filtered_smiles):
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            continue
                        try:
                            sc = MurckoScaffold.GetScaffoldForMol(mol)
                            sc_smi = Chem.MolToSmiles(sc) if sc is not None else ""
                        except Exception:
                            sc_smi = ""
                        freq = float(self.diversity_memory.scaffold_counter.get(sc_smi, 0))
                        scaffold_factor = 1.0 / (1.0 + alpha * freq)
                        adjusted[i] = float(adjusted[i]) * scaffold_factor
                        meta = reward_meta.get(smi, {})
                        meta["scaffold_balance_factor"] = scaffold_factor
                        reward_meta[smi] = meta
                    rewards_raw = torch.tensor(adjusted, dtype=torch.float32, device=rewards_raw.device)
                # Pharmacophore-incompatible molecules receive low fallback reward
                # and are explicitly marked as docking-failed for this iteration.
                reward_list = rewards_raw.tolist()
                floor = float(cfg.docking_rl.failed_docking_reward_floor)
                for i, smi in enumerate(filtered_smiles):
                    p = pharm_details.get(smi)
                    if p is not None and not p.passed:
                        reward_list[i] = min(reward_list[i], floor)
                        meta = reward_meta.get(smi, {})
                        meta["pharmacophore_passed"] = False
                        meta["pharmacophore_score"] = float(p.score)
                        meta["pharmacophore_failed_rules"] = list(p.failed_rules)
                        meta["docking_success"] = False
                        meta["reward"] = float(reward_list[i])
                        reward_meta[smi] = meta
                rewards_raw = torch.tensor(reward_list, dtype=torch.float32, device=rewards_raw.device)
                if rewards_raw is None or rewards_raw.numel() == 0:
                    logger.warning("Iter %s: empty rewards — skipping.", it)
                    continue

                # Route docking/context into the unified RewardEngine so reward
                # components directly shape the policy gradient objective.
                ctx = self.reward_engine.make_context(
                    protein_sequence=protein_sequence or "",
                    seed_smiles=seed_smiles,
                    pharmacophore_target="EGFR",
                )
                ctx.cache = {
                    smi: {
                        "vina_score": reward_meta.get(smi, {}).get("vina_score"),
                        "docking_success": reward_meta.get(smi, {}).get("docking_success"),
                    }
                    for smi in filtered_smiles
                }
                rew_details = self.reward_engine.evaluate(filtered_smiles, ctx)
                if not rew_details:
                    logger.warning("Iter %s: reward_engine returned empty details — skipping.", it)
                    continue
                rewards_from_engine = [float(x.get("total_reward", 0.0)) for x in rew_details]
                rewards_raw = torch.tensor(rewards_from_engine, dtype=torch.float32)
                assert rewards_raw.numel() > 0, "rewards must be non-empty"

                mean_raw = float(rewards_raw.mean().item())
                if abs(mean_raw) < 1e-9:
                    logger.warning("[RL] iteration=%d mean_reward is ~0; learning signal may be weak", it + 1)
                r = rewards_raw.to(self.device)
                r = apply_reward_normalization(
                    r,
                    mode=cfg.docking_rl.reward_normalization,
                    eps=cfg.normalized_reward_eps,
                )
                r, self._reward_ema_mean = apply_ema_smoothing(
                    r,
                    factor=cfg.docking_rl.reward_smoothing_factor,
                    prev_mean=self._reward_ema_mean,
                )
                r_norm = sanitize_and_clip_rewards(
                    r,
                    clip_min=cfg.docking_rl.reward_clip_min,
                    clip_max=cfg.docking_rl.reward_clip_max,
                )

                for smi, r in zip(filtered_smiles, rewards_raw.tolist()):
                    if Chem.MolFromSmiles(smi) is not None and len(smi) > 5:
                        if smi not in all_generated or all_generated[smi] < r:
                            all_generated[smi] = r
                        meta = reward_meta.get(smi, {})
                        self.diversity_memory.add(
                            smi,
                            reward=float(r),
                            docking_success=bool(meta.get("docking_success", False)),
                            vina_score=(float(meta["vina_score"]) if meta.get("vina_score") is not None else None),
                        )

                replay.push(
                    filtered_smiles,
                    rewards_raw.tolist(),
                    metadata_by_smiles=reward_meta,
                )

                agent.train()
                optimizer.zero_grad()

                # CRITICAL: align token IDs to the exact smiles that produced rewards.
                aligned_ids, aligned_rewards = encode_batch_aligned(
                    self.vocab,
                    filtered_smiles,
                    r_norm.detach().cpu().tolist(),
                    self.device,
                )
                if aligned_ids is None or not aligned_rewards:
                    logger.warning("Iter %s: aligned token encoding empty — skipping.", it)
                    continue
                with torch.no_grad():
                    p_lp = prior_log_prob(self.prior, aligned_ids.to(self.device))
                a_lp = agent.log_prob(aligned_ids.to(self.device))
                r_aligned = torch.tensor(aligned_rewards, dtype=torch.float32, device=self.device)
                augmented = p_lp + cfg.sigma * r_aligned
                loss = ((a_lp - augmented) ** 2).mean()

                if len(replay) >= cfg.replay.batch_size:
                    rep_smiles, rep_rewards = replay.sample(cfg.replay.batch_size)
                    rep_ids, rep_kept = encode_batch_aligned(
                        self.vocab,
                        rep_smiles,
                        rep_rewards,
                        self.device,
                    )
                    if rep_ids is not None and rep_kept:
                        rep_t = torch.tensor(rep_kept, dtype=torch.float32, device=self.device)
                        rep_norm = apply_reward_normalization(
                            rep_t,
                            mode=cfg.docking_rl.reward_normalization,
                            eps=cfg.normalized_reward_eps,
                        )
                        rep_norm = sanitize_and_clip_rewards(
                            rep_norm,
                            clip_min=cfg.docking_rl.reward_clip_min,
                            clip_max=cfg.docking_rl.reward_clip_max,
                        )
                        rep_a = agent.log_prob(rep_ids)
                        rep_p = prior_log_prob(self.prior, rep_ids).detach()
                        rep_aug = rep_p + cfg.sigma * rep_norm
                        # Keep replay influence modest during early stabilization to
                        # avoid overpowering fresh on-policy gradients.
                        replay_loss = ((rep_a - rep_aug) ** 2).mean()
                        loss = loss + (0.30 * replay_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                assert torch.isfinite(loss).item(), "policy loss became non-finite"

                valid_count = sum(1 for s in filtered_smiles if Chem.MolFromSmiles(s))
                dock_attempted = len(shortlist)
                dock_successes = sum(1 for s in shortlist if docking_success.get(s, False))
                best_vina = min(vina_scores.values()) if vina_scores else None
                logger.info(
                    "[REINVENT][SURVIVAL] iter=%d sampled=%d valid=%d diversity_survivors=%d pharmacophore_survivors=%d prefilter_survivors=%d docked_count=%d docking_success_count=%d",
                    it + 1,
                    sampled_count,
                    valid_count,
                    diversity_survivors,
                    pharmacophore_survivors,
                    prefilter_survivors,
                    dock_attempted,
                    dock_successes,
                )
                if (it + 1) % 10 == 0 or it == 0:
                    logger.info(
                        "[REINVENT] Iter=%d/%d valid=%d/%d mean_reward=%.3f dock_success=%d/%d best_vina=%s loss=%.3f unique=%d",
                        it + 1,
                        n_iterations,
                        valid_count,
                        bs,
                        mean_raw,
                        dock_successes,
                        max(1, dock_attempted),
                        f"{best_vina:.4f}" if best_vina is not None else "NA",
                        float(loss.item()),
                        len(all_generated),
                    )
                log_rl_iteration(
                    it,
                    n_iterations,
                    valid_count,
                    bs,
                    mean_raw,
                    float(r_norm.mean().item()),
                    float(loss.item()),
                    len(all_generated),
                    docking_success_rate=(dock_successes / max(1, dock_attempted)),
                    best_vina_score=(float(best_vina) if best_vina is not None else None),
                    diversity_buffer_occupancy=self.diversity_memory.stats().buffer_occupancy,
                    diversity_rejected=self.diversity_memory.stats().rejected_count,
                    diversity_mean_tanimoto=self.diversity_memory.stats().mean_tanimoto,
                    diversity_unique_scaffolds=self.diversity_memory.stats().unique_scaffolds,
                    diversity_scaffold_entropy=self.diversity_memory.stats().scaffold_entropy,
                    top_smiles=sorted(
                        all_generated.keys(),
                        key=lambda k: all_generated[k],
                        reverse=True,
                    )[:3],
                )
                docking_comp = [
                    float((x.get("components", {}) or {}).get("docking", 0.0))
                    for x in rew_details
                ]
                pharm_comp = [
                    float((x.get("components", {}) or {}).get("egfr_pharmacophore", 0.0))
                    for x in rew_details
                ]
                div_comp = [
                    float((x.get("components", {}) or {}).get("diversity", 0.0))
                    for x in rew_details
                ]
                motif_hits = sum(
                    1
                    for x in rew_details
                    if (x.get("components", {}) or {}).get("matched_motifs")
                )
                motif_hit_rate = float(motif_hits) / float(max(1, len(rew_details)))
                quin_freq = sum(
                    1
                    for x in rew_details
                    if "quinazoline" in ((x.get("components", {}) or {}).get("matched_motifs") or [])
                ) / float(max(1, len(rew_details)))
                pyr_freq = sum(
                    1
                    for x in rew_details
                    if "pyrimidine" in ((x.get("components", {}) or {}).get("matched_motifs") or [])
                ) / float(max(1, len(rew_details)))
                hinge_freq = sum(
                    1
                    for x in rew_details
                    if "hinge_donor_acceptor" in ((x.get("components", {}) or {}).get("matched_motifs") or [])
                ) / float(max(1, len(rew_details)))
                logger.info(
                    "[RL] iteration=%d mean_reward=%.4f policy_loss=%.4f docking_reward=%.4f pharmacophore_reward=%.4f diversity_score=%.4f valid_fraction=%.4f motif_hit_rate=%.4f",
                    it + 1,
                    mean_raw,
                    float(loss.item()),
                    float(sum(docking_comp) / max(1, len(docking_comp))),
                    float(sum(pharm_comp) / max(1, len(pharm_comp))),
                    float(sum(div_comp) / max(1, len(div_comp))),
                    float(valid_count) / float(max(1, bs)),
                    motif_hit_rate,
                )
                logger.info(
                    "[RL] motif_frequency quinazoline=%.4f pyrimidine=%.4f hinge=%.4f",
                    quin_freq,
                    pyr_freq,
                    hinge_freq,
                )
                iter_metrics = {
                    "iteration": int(it + 1),
                    "mean_reward": mean_raw,
                    "policy_loss": float(loss.item()),
                    "docking_reward": float(sum(docking_comp) / max(1, len(docking_comp))),
                    "pharmacophore_reward": float(sum(pharm_comp) / max(1, len(pharm_comp))),
                    "diversity_score": float(sum(div_comp) / max(1, len(div_comp))),
                    "valid_fraction": float(valid_count) / float(max(1, bs)),
                    "motif_hit_rate": motif_hit_rate,
                    "quinazoline_frequency": quin_freq,
                    "pyrimidine_frequency": pyr_freq,
                    "hinge_frequency": hinge_freq,
                    "survival": {
                        "sampled": int(sampled_count),
                        "diversity_survivors": int(diversity_survivors),
                        "pharmacophore_survivors": int(pharmacophore_survivors),
                        "prefilter_survivors": int(prefilter_survivors),
                        "docked_count": int(dock_attempted),
                        "docking_success_count": int(dock_successes),
                        "final_candidate_pool": int(len(all_generated)),
                    },
                }
                self.rl_metrics_history.append(iter_metrics)
                metrics_path = Path("results") / "rl_metrics.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "iterations": self.rl_metrics_history,
                            "before_random_distribution_count": len(all_generated_before),
                            "post_rl_distribution_count": len(all_generated),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if ckpt_every and (it + 1) % ckpt_every == 0:
                    if cfg.checkpoint.checkpoint_dir:
                        from models.generation.reinvent4 import checkpointing as ckpt

                        ckpt.save_checkpoint(
                            Path(cfg.checkpoint.checkpoint_dir) / f"step_{it+1}.pt",
                            agent_state=agent.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            replay_state=replay.state_dict(),
                            iteration=it + 1,
                            config_dict=ckpt.config_to_jsonable(cfg),
                            reward_state={
                                "ema_mean": self._reward_ema_mean,
                                "history": self.rl_metrics_history,
                            },
                            diversity_state=self.diversity_memory.state_dict(),
                        )
            except Exception as ex:
                logger.warning("  RL iter %s failed: %s — continuing.", it, ex)
                continue

        logger.info("[REINVENT] RL done. unique_valid_molecules=%d", len(all_generated))
        after_mean = float(next(agent.parameters()).detach().mean().item())
        print(f"[RL-CHECK] agent_param_mean_after={after_mean:.10f}")
        logger.info(
            "[RL-CHECK] agent_param_mean_before=%.10f after=%.10f delta=%.10f",
            before_mean,
            after_mean,
            (after_mean - before_mean),
        )
        div_stats = self.diversity_memory.stats()
        self.last_training_summary = {
            "unique_valid_molecules": len(all_generated),
            "before_random_distribution_count": len(all_generated_before),
            "post_rl_distribution_count": len(all_generated),
            "diversity": {
                "buffer_occupancy": div_stats.buffer_occupancy,
                "rejected_count": div_stats.rejected_count,
                "accepted_count": div_stats.accepted_count,
                "mean_tanimoto": div_stats.mean_tanimoto,
                "unique_scaffolds": div_stats.unique_scaffolds,
                "scaffold_entropy": div_stats.scaffold_entropy,
            },
            "reward_ema_mean": self._reward_ema_mean,
            "docking_reward_enabled": bool(cfg.docking_rl.enable_docking_rl),
            "rl_iterations": n_iterations,
            "batch_size": bs,
            "metrics_path": str(Path("results") / "rl_metrics.json"),
        }
        if cfg.checkpoint.checkpoint_dir:
            from models.generation.reinvent4 import checkpointing as ckpt

            final_path = Path(cfg.checkpoint.checkpoint_dir) / "latest.pt"
            ckpt.save_checkpoint(
                final_path,
                agent_state=agent.state_dict(),
                optimizer_state=optimizer.state_dict(),
                replay_state=replay.state_dict(),
                iteration=n_iterations,
                config_dict=ckpt.config_to_jsonable(cfg),
                reward_state={
                    "ema_mean": self._reward_ema_mean,
                    "history": self.rl_metrics_history,
                },
                diversity_state=self.diversity_memory.state_dict(),
            )
            logger.info("[RL] final checkpoint saved: %s", final_path)
        div_loose = DiversityConfig(
            intra_tanimoto_max=max(0.90, float(cfg.diversity.intra_tanimoto_max)),
            seed_tanimoto_max=max(0.999, float(cfg.diversity.seed_tanimoto_max)),
            mw_min=float(cfg.diversity.mw_min),
            mw_max=float(cfg.diversity.mw_max),
            forbidden_atomic_nums=tuple(cfg.diversity.forbidden_atomic_nums),
            use_scaffold_diversity=False,
            memory_buffer_size=int(cfg.diversity.memory_buffer_size),
            rejection_tanimoto_threshold=float(cfg.diversity.rejection_tanimoto_threshold),
            scaffold_balance_alpha=0.0,
        )
        final = select_diverse_top(
            all_generated,
            seed_fp,
            num_molecules * 3,
            div_loose,
        )
        if not final:
            # Temporary stabilization: if diversity gate is too strict early in RL,
            # return top scored molecules directly instead of failing hard.
            final = sorted(
                all_generated.keys(),
                key=lambda s: all_generated[s],
                reverse=True,
            )[: max(1, num_molecules * 3)]
        if not final:
            raise ValueError(
                "REINVENT did not generate any drug-like molecules. "
                "Check seed SMILES and prior weights."
            )
        logger.info("[REINVENT] Returning %d candidates.", min(len(final), num_molecules))
        return final[:num_molecules]
