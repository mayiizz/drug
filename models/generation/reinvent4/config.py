"""
Hyperparameters and paths for REINVENT-inspired SMILES-LSTM RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PRIOR_FILENAME = "smiles_lstm_prior.pt"


def _base_dir() -> Path:
    # .../models/generation/reinvent4/config.py -> project root
    return Path(__file__).resolve().parents[3]


def _package_root() -> Path:
    return _base_dir() / "models" / "generation" / "reinvent4"


def _generation_dir() -> Path:
    return _base_dir() / "models" / "generation"


def resolve_prior_paths() -> tuple[Path, Path]:
    """(primary reinvent4 weights dir, legacy generation/weights dir)."""
    primary = _package_root() / "weights"
    legacy = _generation_dir() / "weights"
    return primary, legacy


def resolve_prior_file() -> Path:
    """Resolve canonical REINVENT prior checkpoint or raise clear not-found error."""
    primary, legacy = resolve_prior_paths()
    p_primary = primary / PRIOR_FILENAME
    if p_primary.exists() and p_primary.is_file():
        return p_primary
    p_legacy = legacy / PRIOR_FILENAME
    if p_legacy.exists() and p_legacy.is_file():
        return p_legacy
    raise FileNotFoundError(
        "REINVENT prior checkpoint not found.\n"
        "Checked:\n"
        f"- {primary}\n"
        f"- {legacy}\n"
        "Expected file:\n"
        f"- {PRIOR_FILENAME}"
    )


@dataclass
class ReplayConfig:
    buffer_size: int = 256
    batch_size: int = 32
    prioritized: bool = False
    priority_epsilon: float = 1e-6


@dataclass
class DiversityConfig:
    """Post-RL greedy selection thresholds (aligned with legacy REINVENT generator)."""

    intra_tanimoto_max: float = 0.75
    seed_tanimoto_max: float = 0.99
    mw_min: float = 100.0
    mw_max: float = 700.0
    forbidden_atomic_nums: tuple[int, ...] = (5, 34, 52)
    use_scaffold_diversity: bool = False
    memory_buffer_size: int = 50
    rejection_tanimoto_threshold: float = 0.97
    scaffold_balance_alpha: float = 0.0


@dataclass
class CurriculumConfig:
    """Stage boundaries (iteration index inclusive start). Rewards adapt via CurriculumScheduler."""

    stage_validity_end: int = 5
    stage_qed_end: int = 12
    stage_binding_end: int = 20
    stage_selectivity_end: int = 25
    # After stage_selectivity_end: diversity-weighted stage


@dataclass
class SelectivityConfig:
    """Off-target penalties; gated by enable_selectivity (default off for speed parity)."""

    enable_selectivity: bool = False
    w_herg_penalty: float = 0.15
    w_cyp_penalty: float = 0.15
    w_antitarget_binding: float = 0.2
    anti_target_sequences: tuple[str, ...] = ()
    herg_threshold: float = 0.5
    cyp_threshold: float = 0.7


@dataclass
class CheckpointConfig:
    checkpoint_dir: Path | None = field(default_factory=lambda: Path("checkpoints"))
    auto_save_every: int = 10


@dataclass
class DockingRLConfig:
    """
    Docking-guided reinforcement learning controls.

    DeepPurpose remains a shortlist accelerator; docking reward is the primary
    structure-based optimization signal.
    """

    docking_reward_weight: float = 0.60
    qed_reward_weight: float = 0.15
    sa_reward_weight: float = 0.10
    pharmacophore_bonus_weight: float = 0.10
    diversity_bonus_weight: float = 0.05
    pose_stability_bonus_weight: float = 0.03

    docking_top_k: int = 10
    docking_batch_size: int = 10
    docking_timeout: int = 120
    docking_cache_enabled: bool = True
    docking_frequency: int = 1

    enable_docking_rl: bool = True
    reward_normalization: str = "zscore"  # zscore | percentile | none
    reward_smoothing_factor: float = 0.0  # EMA factor in [0,1)

    reward_clip_min: float = -3.0
    reward_clip_max: float = 3.0

    # Fallback when a sample fails docking. If no successes in batch, use this.
    failed_docking_reward_floor: float = 0.05


@dataclass
class Reinvent4Config:
    fine_tuned_prior_path: Path | None = None
    """Full training / generation hyperparameters."""
    enable_runtime_rl: bool = False

    rl_iterations: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    sigma: float = 120.0
    max_sample_length: int = 72
    sample_temperature: float = 0.6
    greedy_sampling: bool = False

    pretrain_steps: int = 20000
    pretrain_batch_mols: int = 64
    pretrain_lr: float = 1e-3

    replay: ReplayConfig = field(default_factory=ReplayConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    selectivity: SelectivityConfig = field(default_factory=SelectivityConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    docking_rl: DockingRLConfig = field(default_factory=DockingRLConfig)

    binding_sigmoid_centre_nm: float = 100.0
    binding_sigmoid_slope: float = 0.02

    w_binding: float = 0.45
    w_qed: float = 0.25
    w_sa: float = 0.15
    # Deprecated for docking-in-the-loop RL objective (kept for compatibility).
    w_novelty: float = 0.10
    w_validity: float = 0.05
    w_selectivity: float = 0.0
    # Deprecated for docking-in-the-loop RL objective (kept for compatibility).
    w_diversity_bonus: float = 0.0

    novelty_low_sim_cutoff: float = 0.10
    novelty_identical_cutoff: float = 0.99

    reward_clip_min: float | None = None
    reward_clip_max: float | None = None
    normalized_reward_eps: float = 1e-8

    weight_urls: tuple[str, ...] = (
        "https://ndownloader.figshare.com/files/13371586",
        "https://github.com/tblaschke/reinvent-models/releases/download/v1.0/ChEMBL_LSTM_prior.pt",
    )

    lstm_embed_dim: int = 256
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.1


def default_config() -> Reinvent4Config:
    return Reinvent4Config()


# Backward-compat module-level accessors (thin aliases for legacy imports via wrapper)
PRIMARY_WEIGHTS_DIR, LEGACY_WEIGHTS_DIR = resolve_prior_paths()
