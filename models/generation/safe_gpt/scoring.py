from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import QED

from models.generation.safe_gpt.config import RewardConfig, SAFEGPTConfig
from models.generation.safe_gpt.medicinal_priors import medicinal_prior_score
from models.generation.safe_gpt.pharmacophore import pharmacophore_score
from models.generation.safe_gpt.utils import morgan_fp, parse_binding, safe_sa, tanimoto


class BindingScorer:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def score(self, smiles: str, protein_seq: str, binding_model: object) -> float:
        if binding_model is None or not protein_seq:
            return 0.5
        try:
            raw = binding_model.predict(smiles, protein_seq)  # type: ignore[attr-defined]
            nm = parse_binding(raw)
            return 1.0 / (1.0 + math.exp(self.cfg.binding_sigmoid_slope * (nm - self.cfg.binding_sigmoid_centre)))
        except Exception:
            return 0.5


class DockingScorer:
    """Pluggable docking interface; current fallback delegates to DeepPurpose-style score."""

    def score(self, smiles: str, protein_path: str = "", fallback: float = 0.5) -> float:
        _ = (smiles, protein_path)
        return fallback


class QEDScorer:
    def score(self, mol: Chem.Mol | None) -> float:
        if mol is None:
            return 0.0
        try:
            return float(QED.qed(mol))
        except Exception:
            return 0.5


class SAScorer:
    def score(self, mol: Chem.Mol | None) -> float:
        if mol is None:
            return 0.0
        return max(0.0, (10.0 - safe_sa(mol)) / 9.0)


class NoveltyScorer:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def score(self, smiles: str, seed_fp) -> float:
        fp = morgan_fp(smiles)
        sim = tanimoto(fp, seed_fp)
        if sim >= self.cfg.seed_similarity_max:
            return 0.0
        if sim < self.cfg.novelty_low_sim_cutoff:
            return 0.5
        return 1.0 - sim


class DiversityScorer:
    def score(self, smiles: str, selected_fps: list[object], threshold: float = 0.75) -> float:
        fp = morgan_fp(smiles)
        if fp is None:
            return 0.0
        if not selected_fps:
            return 1.0
        max_sim = max(tanimoto(fp, sfp) for sfp in selected_fps)
        if max_sim <= threshold:
            return 1.0
        return max(0.0, 1.0 - (max_sim - threshold) / max(1e-8, 1.0 - threshold))


class PharmacophoreScorer:
    def __init__(self, cfg: SAFEGPTConfig):
        self.cfg = cfg

    def score(self, smiles: str) -> float:
        return pharmacophore_score(smiles, self.cfg.pharmacophore)


def compute_reward(
    smiles: str,
    seed_fp,
    protein_sequence: str,
    binding_model: object,
    cfg: SAFEGPTConfig,
) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    binding_r = BindingScorer(cfg.rewards).score(smiles, protein_sequence, binding_model)
    qed_r = QEDScorer().score(mol)
    sa_r = SAScorer().score(mol)
    novelty_r = NoveltyScorer(cfg.rewards).score(smiles, seed_fp)
    validity_r = 1.0
    pharm_r = pharmacophore_score(smiles, cfg.pharmacophore)
    prior_r = medicinal_prior_score(smiles, cfg.medicinal_priors)

    base = (
        cfg.rewards.w_binding * binding_r
        + cfg.rewards.w_qed * qed_r
        + cfg.rewards.w_sa * sa_r
        + cfg.rewards.w_novelty * novelty_r
        + cfg.rewards.w_validity * validity_r
    )
    return max(0.0, min(1.0, 0.8 * base + 0.1 * pharm_r + 0.1 * prior_r))

