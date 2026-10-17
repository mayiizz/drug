from __future__ import annotations

import logging
from collections import Counter

from models.generation.safe_gpt.config import GenerationConfig
from models.generation.safe_gpt.scaffold import validate_and_encode_scaffold

logger = logging.getLogger(__name__)
_fallback_reason_counter: Counter[str] = Counter()

# Module-level latch: once scaffold encoding fails for this seed, all
# subsequent calls skip straight to de_novo without re-trying validation.
_scaffold_permanently_failed: bool = False


def reset_scaffold_latch() -> None:
    """Call this between generate() invocations so each new seed gets a fresh attempt."""
    global _scaffold_permanently_failed
    _scaffold_permanently_failed = False
    _fallback_reason_counter.clear()


def _normalize_results(results: object) -> list[str]:
    if results is None:
        return []
    if isinstance(results, list):
        return [s for s in results if isinstance(s, str) and s]
    return []


def _log_generation_quality(stage: str, generated: list[str]) -> None:
    valid = _normalize_results(generated)
    unique = list(dict.fromkeys(valid))
    duplicate_count = max(0, len(valid) - len(unique))
    logger.info(
        "[SAFE] %s generated=%d valid=%d invalid=%d duplicate=%d",
        stage,
        len(generated) if isinstance(generated, list) else 0,
        len(valid),
        (len(generated) if isinstance(generated, list) else 0) - len(valid),
        duplicate_count,
    )


def generate_candidates(
    designer: object,
    seed_smiles: str,
    n: int,
    cfg: GenerationConfig,
) -> list[str]:
    global _scaffold_permanently_failed

    kwargs: dict[str, object] = {
        "n_samples_per_trial": n,
        "sanitize": True,
        "temperature": cfg.temperature,
    }

    mode = cfg.generation_mode
    try:
        if mode == "scaffold_decoration":
            # If scaffold encoding already failed this session, skip straight to de_novo.
            if _scaffold_permanently_failed:
                logger.debug("[SAFE] Scaffold permanently failed — using de_novo directly.")
                fallback = _normalize_results(designer.de_novo_generation(**kwargs))  # type: ignore[attr-defined]
                _log_generation_quality("de_novo_latch", fallback)
                return fallback

            logger.info("[SAFE] Raw scaffold seed=%s", seed_smiles)
            scaffold_result = validate_and_encode_scaffold(seed_smiles, slicer=cfg.slicer)

            if not scaffold_result.valid or scaffold_result.safe_scaffold is None:
                reason = scaffold_result.reason or "unknown"
                _fallback_reason_counter[reason] += 1

                # Latch permanently to de_novo so subsequent rounds don't re-validate.
                _scaffold_permanently_failed = True

                logger.warning("[SAFE] Scaffold conditioning unavailable")
                logger.warning("[SAFE] Switching permanently to de_novo generation")
                logger.warning("[SAFE] Raw scaffold   : %s", seed_smiles)
                logger.warning("[SAFE] Sanitized      : %s", scaffold_result.sanitized_smiles)
                logger.warning("[SAFE] Failure reason : %s", reason)
                logger.warning(
                    "[SAFE] scaffold_decoration invalid scaffold (%s) -> de_novo fallback (count=%d)",
                    reason,
                    _fallback_reason_counter[reason],
                )
                fallback = _normalize_results(designer.de_novo_generation(**kwargs))  # type: ignore[attr-defined]
                _log_generation_quality("de_novo_fallback", fallback)
                return fallback

            logger.info(
                "[SAFE] Scaffold encoded successfully with slicer=%s",
                scaffold_result.slicer_used,
            )
            results = _normalize_results(
                designer.scaffold_decoration(scaffold=scaffold_result.safe_scaffold, **kwargs)  # type: ignore[attr-defined]
            )
            _log_generation_quality("scaffold_decoration", results)
            return results

        if mode == "motif_extension":
            results = _normalize_results(
                designer.motif_extension(motif=seed_smiles, **kwargs)  # type: ignore[attr-defined]
            )
            _log_generation_quality("motif_extension", results)
            return results
        if mode == "scaffold_morphing":
            results = _normalize_results(
                designer.scaffold_morphing(scaffold=seed_smiles, **kwargs)  # type: ignore[attr-defined]
            )
            _log_generation_quality("scaffold_morphing", results)
            return results
        if mode == "linker_generation":
            results = _normalize_results(
                designer.linker_generation(fragment=seed_smiles, **kwargs)  # type: ignore[attr-defined]
            )
            _log_generation_quality("linker_generation", results)
            return results

        results = _normalize_results(designer.de_novo_generation(**kwargs))  # type: ignore[attr-defined]
        _log_generation_quality("de_novo_generation", results)
        return results

    except Exception as exc:
        logger.warning("Generation mode %s failed (%s), fallback de_novo.", mode, exc)
        try:
            results = _normalize_results(designer.de_novo_generation(**kwargs))  # type: ignore[attr-defined]
            _log_generation_quality("de_novo_exception_fallback", results)
            return results
        except Exception:
            return []
    finally:
        if _fallback_reason_counter:
            total = sum(_fallback_reason_counter.values())
            reason_parts = ", ".join(f"{k}={v}" for k, v in sorted(_fallback_reason_counter.items()))
            logger.info("[SAFE] %d invalid scaffolds -> de_novo fallback summary (%s)", total, reason_parts)