from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    sampling_rounds: int = 10
    samples_per_round: int = 64
    temperature: float = 1.0
    max_length: int = 256
    generation_mode: str = "de_novo"
    slicer: str = "brics"  # brics | recap | mmpa | attach | hr


@dataclass
class RewardConfig:
    w_binding: float = 0.45
    w_qed: float = 0.25
    w_sa: float = 0.15
    w_novelty: float = 0.10
    w_validity: float = 0.05
    binding_sigmoid_centre: float = 100.0
    binding_sigmoid_slope: float = 0.02
    novelty_low_sim_cutoff: float = 0.10
    seed_similarity_max: float = 0.99


@dataclass
class DiversityConfig:
    tanimoto_threshold: float = 0.75
    seed_similarity_max: float = 0.99
    mw_min: float = 100.0
    mw_max: float = 700.0
    forbidden_atomic_nums: list[int] = field(default_factory=lambda: [5, 34, 52])
    use_rolling_memory: bool = True
    rolling_memory_size: int = 256


@dataclass
class PharmacophoreConfig:
    enabled: bool = True
    min_ring_count: int = 1
    min_hba: int = 1
    min_hbd: int = 0
    require_aromatic_heterocycle: bool = False


@dataclass
class MedicinalPriorsConfig:
    enabled: bool = True
    prefer_heterocycles: bool = True
    oral_like_soft_bias: bool = True
    cns_like_soft_bias: bool = False
    ring_complexity_penalty_start: int = 5


@dataclass
class FilteringConfig:
    min_ring_count: int = 0
    max_ring_count: int = 10
    require_aromatic_if_multiring: bool = False
    reject_formal_charge: bool = True
    pains_hook_enabled: bool = True
    reactive_group_hook_enabled: bool = True


@dataclass
class DockingConfig:
    docking_backend: str = "deeppurpose"  # deeppurpose | vina
    vina_enabled: bool = False


@dataclass
class SAFEGPTConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    pharmacophore: PharmacophoreConfig = field(default_factory=PharmacophoreConfig)
    medicinal_priors: MedicinalPriorsConfig = field(default_factory=MedicinalPriorsConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)

