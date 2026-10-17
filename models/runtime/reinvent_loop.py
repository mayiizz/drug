"""Thin orchestrator around :class:`ReinventTrainer`.

Wraps the existing REINVENT augmented-likelihood trainer so the rest of the
pipeline can use a small, stable surface area:

* :meth:`ReinventLoop.sample` - batched ``SMILESLSTM.sample(batch_size)``
* :meth:`ReinventLoop.run_rl` - drive a structure-conditioned RL session,
  optionally using :class:`RewardEngine` for reward override.
* :meth:`ReinventLoop.update`, :meth:`checkpoint`, :meth:`load` - delegate to
  trainer primitives so we keep replay buffer, diversity memory, curriculum,
  and docking-in-the-loop without duplicating that logic here.

TODO: support swappable RL algorithms beyond REINVENT augmented-likelihood
(e.g. PPO) by accepting an algorithm strategy in the constructor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from rdkit import Chem

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.config import Reinvent4Config
from models.generation.reinvent4.checkpoint_utils import (
    checkpoint_metadata,
    resolve_best_prior_checkpoint,
    state_param_count,
    torch_load_checkpoint,
    unwrap_model_state_dict,
)
from models.generation.reinvent4.diversity import (
    RollingDiversityMemory,
    select_diverse_top,
)
from models.generation.reinvent4.sampler import (
    diversity_acceptance_filter,
    sample_smiles_batch,
)
from models.generation.reinvent4.trainer import ReinventTrainer
from models.ranking.reward_engine import RewardContext, RewardEngine

logger = logging.getLogger(__name__)


class ReinventLoop:
    """Single-entry orchestrator around :class:`ReinventTrainer`.

    Construction is lazy: if no ``trainer`` is provided one is built from the
    given ``config`` (defaulting to :class:`Reinvent4Config`). The same trainer
    holds the prior and (during ``run_rl``) the agent network, replay buffer,
    diversity memory, and curriculum scheduler.
    """

    def __init__(
        self,
        *,
        config: Reinvent4Config | None = None,
        trainer: ReinventTrainer | None = None,
        reward_engine: RewardEngine | None = None,
    ) -> None:
        self.config = config or (trainer.config if trainer is not None else Reinvent4Config())
        self.trainer = trainer if trainer is not None else ReinventTrainer(self.config)
        self.reward_engine = reward_engine
        self._runtime_health_checked = False

    # ── Properties -------------------------------------------------------

    @property
    def vocab(self):
        return self.trainer.vocab

    @property
    def device(self) -> torch.device:
        return self.trainer.device

    @property
    def diversity_memory(self) -> RollingDiversityMemory:
        return self.trainer.diversity_memory

    # ── Sampling --------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int = 16,
        max_len: int = 72,
        temperature: float = 0.6,
        greedy: bool = False,
        validate: bool = True,
    ) -> list[str]:
        """Batched sampling from the prior policy.

        Always uses the underlying :class:`SMILESLSTM`'s vectorised
        ``sample(batch_size=N)`` - no per-molecule loops. When ``validate`` is
        true, output is parsed by RDKit and only canonicalised valid SMILES
        are returned.
        """
        if batch_size <= 0:
            return []
        try:
            resolved = resolve_best_prior_checkpoint()
            state = torch_load_checkpoint(resolved.path, self.device)
            sd = unwrap_model_state_dict(state)
            if hasattr(self.trainer, "prior"):
                self.trainer.prior.load_state_dict(sd, strict=False)
            if hasattr(self.trainer, "agent"):
                self.trainer.agent.load_state_dict(sd, strict=False)
            logger.info("[PRIOR] checkpoint=%s", resolved.path)
            logger.info("[PRIOR] params=%d", state_param_count(sd))
            logger.info("[PRIOR] vocab=%d", len(self.vocab))
            meta = checkpoint_metadata(state)
            if meta:
                logger.info("[PRIOR] metadata=%s", meta)
            if resolved.is_bootstrap_fallback:
                logger.warning("[PRIOR] warning: using bootstrap prior weights")
        except Exception as e:
            logger.warning("[RL] checkpoint preload failed: %s", e)
        if not self._runtime_health_checked:
            self._runtime_prior_health_check()
            self._runtime_health_checked = True
        # Use trainable agent if present so sampling reflects RL updates.
        policy: SMILESLSTM = getattr(self.trainer, "agent", self.trainer.prior)  # type: ignore[assignment]
        policy.eval()
        sampled, _ = sample_smiles_batch(
            policy,
            self.vocab,
            batch_size,
            device=self.device,
            max_len=max_len,
            temperature=temperature,
            greedy=greedy,
        )
        if not validate:
            return sampled
        out: list[str] = []
        for raw in sampled:
            if not raw:
                continue
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                continue
            try:
                out.append(Chem.MolToSmiles(mol, canonical=True))
            except Exception:
                continue
        return out

    def _runtime_prior_health_check(self) -> None:
        try:
            policy: SMILESLSTM = getattr(self.trainer, "agent", self.trainer.prior)  # type: ignore[assignment]
            policy.eval()
            _sampled, _ids = sample_smiles_batch(
                policy,
                self.vocab,
                64,
                device=self.device,
                max_len=72,
                temperature=0.6,
                greedy=False,
            )
            stats = getattr(policy, "last_sample_stats", {}) or {}
            raw_valid_fraction = float(stats.get("raw_valid_fraction", 0.0))
            logger.info("[PRIOR] runtime_health_raw_valid_fraction=%.4f", raw_valid_fraction)
            if raw_valid_fraction < 0.80:
                logger.warning("[PRIOR] runtime prior quality below expected threshold")
        except Exception as e:
            logger.warning("[PRIOR] runtime health check failed: %s", e)

    def sample_unique(
        self,
        *,
        n: int,
        max_len: int = 72,
        temperature: float = 1.0,
        per_batch: int | None = None,
        max_attempts: int = 8,
        diversity_threshold: float | None = None,
    ) -> list[str]:
        """Sample until ``n`` unique valid SMILES are collected (or attempts exhausted).

        Optionally pipes results through the trainer's rolling diversity memory
        (``diversity_threshold`` Tanimoto) to keep the active set chemically diverse.
        """
        bs = per_batch if per_batch is not None else max(8, min(64, n))
        seen: dict[str, None] = {}
        for _ in range(max_attempts):
            batch = self.sample(batch_size=bs, max_len=max_len, temperature=temperature)
            if diversity_threshold is not None:
                batch, _ = diversity_acceptance_filter(
                    batch, self.diversity_memory, threshold=float(diversity_threshold)
                )
            for s in batch:
                if s and s not in seen:
                    seen[s] = None
                if len(seen) >= n:
                    break
            if len(seen) >= n:
                break
        return list(seen.keys())[:n]

    # ── RL driving ------------------------------------------------------

    def run_rl(
        self,
        *,
        seed_smiles: str | None,
        protein_sequence: str = "",
        binding_model: Any = None,
        num_molecules: int = 20,
        num_iterations: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        checkpoint_interval: int | None = None,
        resume_training: bool = False,
        on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[str]:
        """Drive the existing REINVENT trainer's augmented-likelihood loop.

        ``ReinventTrainer.generate`` already performs batched sampling, replay
        buffer updates, diversity-aware filtering, optional docking-in-loop,
        EMA reward smoothing, and checkpointing. We delegate to it directly so
        we don't fork that algorithm.

        ``on_iteration`` is reserved for future per-iteration callbacks; the
        underlying trainer logs iteration summaries via the standard logger
        in the meantime.

        TODO: thread ``on_iteration`` through the trainer once
        ``ReinventTrainer.generate`` accepts an iteration callback.
        TODO: swap reward composition to use ``self.reward_engine`` once the
        trainer exposes a hook for per-iteration reward override.
        """
        if not seed_smiles:
            # The trainer needs a seed fingerprint for diversity selection. When
            # the caller did not pass one (pure protein-guided mode), use a
            # representative kinase-like seed so that ``select_diverse_top``
            # has a non-degenerate reference fingerprint.
            seed_smiles = "c1cnc2ccccc2n1"  # quinazoline scaffold

        if on_iteration is not None:
            logger.debug(
                "[ReinventLoop] on_iteration callback provided but not yet wired; "
                "falling back to trainer-managed logging."
            )
        if not bool(getattr(self.config, "enable_runtime_rl", False)):
            logger.info("[CURA] runtime mode: PRIOR_ONLY")
            logger.info("[RL] skipped runtime policy updates")
            return self.sample_unique(
                n=max(1, int(num_molecules)),
                max_len=72,
                temperature=0.6,
                per_batch=min(128, max(16, int(batch_size or 64))),
                max_attempts=12,
                diversity_threshold=float(getattr(self.config.diversity, "rejection_tanimoto_threshold", 0.97)),
            )
        logger.info("[CURA] runtime mode: RL_ENABLED")
        # Runtime safeguard: RL cannot optimize chemistry if the prior fails
        # basic SMILES grammar. Probe raw validity before launching RL.
        try:
            _ = self.sample(batch_size=64, max_len=72, temperature=0.6, validate=False)
            stats = getattr(getattr(self.trainer, "agent", None), "last_sample_stats", {}) or {}
            raw_valid_fraction = float(stats.get("raw_valid_fraction", 1.0))
            if raw_valid_fraction < 0.50:
                logger.warning(
                    "[RL] skipped due to weak prior validity: raw_valid_fraction=%.3f < 0.50. "
                    "Please retrain prior with scripts/train_prior.py first.",
                    raw_valid_fraction,
                )
                return []
        except Exception as e:
            logger.warning("[RL] prior-quality precheck failed: %s", e)

        return self.trainer.generate(
            seed_smiles=seed_smiles,
            num_molecules=num_molecules,
            protein_sequence=protein_sequence or "",
            binding_model=binding_model,
            num_iterations=(num_iterations if num_iterations is not None else 60),
            batch_size=(batch_size if batch_size is not None else 32),
            learning_rate=learning_rate,
            checkpoint_interval=checkpoint_interval,
            resume_training=resume_training,
        )

    def evaluate_rewards(
        self,
        smiles_list: list[str],
        context: RewardContext,
    ) -> list[dict[str, Any]]:
        """Convenience: compute composite rewards via :class:`RewardEngine`.

        Useful when caller wants to attach reward scores to a batch sampled
        from :meth:`sample` outside of the inner RL loop.
        """
        if self.reward_engine is None:
            self.reward_engine = RewardEngine()
        return self.reward_engine.evaluate(smiles_list, context)

    # ── Diversity helpers ----------------------------------------------

    def select_top(
        self,
        rewarded: dict[str, float],
        seed_fp,
        n: int,
    ) -> list[str]:
        """Greedy diverse top-N pick using the trainer's diversity config."""
        return select_diverse_top(rewarded, seed_fp, n, self.config.diversity)

    # ── Checkpointing ---------------------------------------------------

    def checkpoint(self, path: str | Path) -> None:
        """Persist trainer state for resume.

        Saves the prior agent state and the rolling diversity memory state.
        For full replay/optimizer state checkpoints, use the trainer's
        ``run_rl`` path which writes auto-saves into
        ``CheckpointConfig.checkpoint_dir``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prior_state_dict": self.trainer.prior.state_dict(),
                "diversity_state": self.trainer.diversity_memory.state_dict(),
                "reward_ema_mean": getattr(self.trainer, "_reward_ema_mean", None),
            },
            p,
        )
        logger.info("[ReinventLoop] checkpoint saved to %s", p)

    def load(self, path: str | Path) -> None:
        """Restore trainer state previously persisted via :meth:`checkpoint`."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Reinvent loop checkpoint not found: {p}")
        state = torch.load(p, map_location=self.device)
        if "prior_state_dict" in state:
            self.trainer.prior.load_state_dict(state["prior_state_dict"])
        if "diversity_state" in state and hasattr(self.trainer.diversity_memory, "load_state_dict"):
            try:
                self.trainer.diversity_memory.load_state_dict(state["diversity_state"])
            except Exception as e:
                logger.warning("[ReinventLoop] diversity restore failed: %s", e)
        if "reward_ema_mean" in state:
            self.trainer._reward_ema_mean = state["reward_ema_mean"]  # type: ignore[attr-defined]
        logger.info("[ReinventLoop] checkpoint loaded from %s", p)
        logger.info("[RL] loaded checkpoint: %s", p)


__all__ = ["ReinventLoop"]
