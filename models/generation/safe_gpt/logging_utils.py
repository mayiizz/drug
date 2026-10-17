from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_round(
    round_idx: int,
    total_rounds: int,
    sampled: int,
    valid_count: int,
    unique_total: int,
    mean_reward: float,
) -> None:
    logger.info(
        "safe_gpt_round=%d/%d sampled=%d valid=%d unique=%d mean_reward=%.4f",
        round_idx,
        total_rounds,
        sampled,
        valid_count,
        unique_total,
        mean_reward,
    )


def log_summary(stats: dict[str, float]) -> None:
    logger.info("safe_gpt_summary=%s", stats)

