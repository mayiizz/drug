"""Docking-guided reward utilities for REINVENT augmented-likelihood RL."""

from __future__ import annotations

import torch
from models.generation.reinvent4.config import DockingRLConfig


def normalize_vina_score(vina_score: float) -> float:
    """
    Convert Vina affinity (kcal/mol, lower/more-negative is better)
    into a bounded, monotonic reward in [0,1].
    """
    try:
        x = -float(vina_score)
    except Exception:
        return 0.0
    # sigmoid(-vina) -> bounded and numerically stable for RL.
    return float(torch.sigmoid(torch.tensor(x, dtype=torch.float32)).item())


def compose_docking_rl_reward(
    *,
    normalized_vina: float,
    qed: float,
    sa_norm: float,
    pharmacophore_bonus: float = 0.0,
    diversity_bonus: float = 0.0,
    pose_stability_bonus: float = 0.0,
    cfg: DockingRLConfig,
) -> float:
    """
    Structure-based multi-objective reward:
      R = w_dock * normalized_vina + w_qed * qed + w_sa * sa_norm
          + w_pharm * pharmacophore_bonus + w_div * diversity_bonus
          + w_pose * pose_stability_bonus
    """
    r = (
        cfg.docking_reward_weight * float(normalized_vina)
        + cfg.qed_reward_weight * float(qed)
        + cfg.sa_reward_weight * float(sa_norm)
        + cfg.pharmacophore_bonus_weight * float(pharmacophore_bonus)
        + cfg.diversity_bonus_weight * float(diversity_bonus)
        + cfg.pose_stability_bonus_weight * float(pose_stability_bonus)
    )
    return float(r)


def percentile_normalize_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Robust normalization:
      (r - median) / (IQR + eps)
    """
    if rewards.numel() == 0:
        return rewards
    med = torch.quantile(rewards, 0.5)
    q25 = torch.quantile(rewards, 0.25)
    q75 = torch.quantile(rewards, 0.75)
    iqr = q75 - q25
    return (rewards - med) / (iqr + eps)


def zscore_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-minibatch normalization before sigma scaling in augmented likelihood:
      (r - mean) / (std + eps)
    """
    if rewards.numel() == 0:
        return rewards
    std = rewards.std()
    if rewards.numel() < 2 or std < 1e-8:
        return rewards - rewards.mean()
    return (rewards - rewards.mean()) / (std + eps)


def zscore_rewards_detached(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same as zscore but without tracking gradients through mean/std."""
    with torch.no_grad():
        return zscore_rewards(rewards, eps)


def apply_reward_normalization(
    rewards: torch.Tensor,
    *,
    mode: str = "zscore",
    eps: float = 1e-8,
) -> torch.Tensor:
    mode_l = (mode or "zscore").lower()
    if mode_l == "none":
        return rewards
    if mode_l == "percentile":
        return percentile_normalize_rewards(rewards, eps)
    return zscore_rewards(rewards, eps)


def sanitize_and_clip_rewards(
    rewards: torch.Tensor,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    out = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_min is not None or clip_max is not None:
        lo = clip_min if clip_min is not None else -float("inf")
        hi = clip_max if clip_max is not None else float("inf")
        out = torch.clamp(out, min=lo, max=hi)
    return out


def apply_ema_smoothing(
    rewards: torch.Tensor,
    *,
    factor: float,
    prev_mean: float | None,
) -> tuple[torch.Tensor, float]:
    """
    Optional scalar EMA smoothing around batch mean for extra RL stability.
    Returns (smoothed_rewards, updated_ema_mean).
    """
    if rewards.numel() == 0:
        return rewards, (0.0 if prev_mean is None else float(prev_mean))
    alpha = max(0.0, min(float(factor), 0.999))
    if alpha <= 0.0:
        m = float(rewards.mean().item())
        return rewards, m
    batch_mean = float(rewards.mean().item())
    ema = batch_mean if prev_mean is None else (alpha * float(prev_mean) + (1.0 - alpha) * batch_mean)
    centered = rewards - rewards.mean()
    smoothed = centered + ema
    return smoothed, ema
