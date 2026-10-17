from __future__ import annotations

import logging
import time

from rdkit import Chem
from rdkit import RDLogger as _RDL

from models.generation.safe_gpt.benchmark import benchmark_stats
from models.generation.safe_gpt.config import SAFEGPTConfig
from models.generation.safe_gpt.designer import load_designer
from models.generation.safe_gpt.diversity import DiversityMemory, select_diverse_top
from models.generation.safe_gpt.filtering import passes_basic_filters
from models.generation.safe_gpt.generation import generate_candidates, reset_scaffold_latch
from models.generation.safe_gpt.logging_utils import log_round, log_summary
from models.generation.safe_gpt.scoring import compute_reward
from models.generation.safe_gpt.utils import morgan_fp

logger = logging.getLogger(__name__)


class SAFEGPTFramework:
    """
    Protein-conditioned molecular proposal framework using frozen SAFE-GPT and
    reward-guided candidate optimization (no policy optimization in this module).
    """

    def __init__(self, cfg: SAFEGPTConfig | None = None):
        self.cfg = cfg or SAFEGPTConfig()
        self._designer: object | None = None
        self._memory = DiversityMemory(self.cfg.diversity.rolling_memory_size)
        self.last_failure: dict[str, object] | None = None
        logger.info("SAFE-GPT framework initialized (lazy model load).")

    def _designer_model(self) -> object:
        if self._designer is None:
            logger.info("Loading SAFE-GPT default model...")
            self._designer = load_designer(verbose=False)
            logger.info("SAFE-GPT model ready.")
        return self._designer

    def generate(
        self,
        seed_smiles: str,
        num_molecules: int = 20,
        protein_sequence: str = "",
        binding_model: object = None,
    ) -> list[str]:
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol is None:
            raise ValueError(f"Invalid seed SMILES: {seed_smiles!r}")
        seed_fp = morgan_fp(seed_smiles)

        # Reset the scaffold-failure latch so each new generate() call
        # gets a fresh attempt at scaffold_decoration before falling back.
        reset_scaffold_latch()

        _RDL.DisableLog("rdApp.*")  # type: ignore[attr-defined]
        cfg = self.cfg
        logger.info(
            "[SAFE-GPT] Starting generation — %s rounds x %s samples (mode=%s)",
            cfg.generation.sampling_rounds,
            cfg.generation.samples_per_round,
            cfg.generation.generation_mode,
        )
        logger.info("[SAFE-GPT] Binding model available: %s", bool(binding_model))

        all_generated: dict[str, float] = {}
        t0 = time.perf_counter()

        for i in range(cfg.generation.sampling_rounds):
            try:
                candidates = generate_candidates(
                    self._designer_model(),
                    seed_smiles,
                    cfg.generation.samples_per_round,
                    cfg.generation,
                )
                valid_this_round = 0
                for smi in candidates:
                    if len(smi) < 5:
                        continue
                    if not passes_basic_filters(smi, cfg.filtering):
                        continue
                    valid_this_round += 1
                    r = compute_reward(
                        smi,
                        seed_fp,
                        protein_sequence,
                        binding_model,
                        cfg,
                    )
                    if smi not in all_generated or all_generated[smi] < r:
                        all_generated[smi] = r

                mean_r = sum(all_generated.values()) / len(all_generated) if all_generated else 0.0
                logger.info(
                    "[SAFE-GPT] Round %d/%d sampled=%d valid=%d unique=%d mean_score=%.3f",
                    i + 1,
                    cfg.generation.sampling_rounds,
                    len(candidates),
                    valid_this_round,
                    len(all_generated),
                    mean_r,
                )
                log_round(
                    i + 1,
                    cfg.generation.sampling_rounds,
                    len(candidates),
                    valid_this_round,
                    len(all_generated),
                    mean_r,
                )
            except Exception as exc:
                logger.warning("SAFE-GPT round %d failed: %s", i, exc)
                continue

        logger.info("[SAFE-GPT] Done. %d unique valid molecules collected.", len(all_generated))
        final = select_diverse_top(
            all_generated,
            seed_fp,
            num_molecules * 3,
            cfg.diversity,
            memory=self._memory,
        )
        if not final:
            self.last_failure = {
                "error_code": "SAFEGPT_EMPTY_GENERATION",
                "stage": "safe_gpt_generation",
                "message": "SAFE-GPT produced no drug-like candidates after configured rounds.",
                "hints": [
                    "Increase sampling_rounds",
                    "Increase samples_per_round",
                    "Switch generation_mode to 'denovo'",
                ],
                "retriable": True,
            }
            raise ValueError(
                "SAFEGPT_EMPTY_GENERATION: SAFE-GPT produced no drug-like candidates "
                "after configured rounds."
            )

        stats = benchmark_stats(final, elapsed_s=time.perf_counter() - t0)
        log_summary(stats)
        logger.info("[SAFE-GPT] Returning %d candidates.", min(len(final), num_molecules))
        return final[:num_molecules]