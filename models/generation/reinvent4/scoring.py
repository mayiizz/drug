"""Docking-guided REINVENT scoring components."""

from __future__ import annotations

import logging
import math
from typing import Any, TYPE_CHECKING

import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from models.generation.reinvent4.config import Reinvent4Config
from models.generation.reinvent4.rewards import compose_docking_rl_reward, normalize_vina_score
from models.generation.reinvent4.selectivity import SelectivityScorer
from models.generation.reinvent4.utils import morgan_fp, parse_binding, safe_sa, tanimoto
from models.pharmacophore.config import PharmacophoreConfig
from models.pharmacophore.filtering import evaluate_smiles, select_profile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ValidityScorer:
    def score(self, mol) -> float:
        return 1.0 if mol is not None else 0.0


class QEDScorer:
    def score(self, mol) -> float:
        if mol is None:
            return 0.0
        try:
            return float(QED.qed(mol))
        except Exception:
            return 0.5


class SAScorer:
    def score(self, mol) -> float:
        if mol is None:
            return 0.0
        sa = safe_sa(mol)
        return max(0.0, (10.0 - sa) / 9.0)


class BindingScorer:
    def __init__(self, config: Reinvent4Config):
        self.cfg = config

    def score(
        self,
        smiles: str,
        mol,
        binding_model,
        protein_sequence: str,
    ) -> float:
        if mol is None:
            return 0.0
        if binding_model is None or not protein_sequence:
            return 0.5
        try:
            raw = binding_model.predict(smiles, protein_sequence)
            binding_nm = parse_binding(raw)
            return 1.0 / (
                1.0
                + math.exp(
                    self.cfg.binding_sigmoid_slope
                    * (binding_nm - self.cfg.binding_sigmoid_centre_nm)
                )
            )
        except Exception as e:
            logger.debug("Binding prediction failed for %s: %s", smiles, e)
            return 0.5


class DiversityScorer:
    """Optional intra-batch diversity signal (mean max Tanimoto penalty)."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def score_vs_pool(self, smiles: str, fps: list, current_fp) -> float:
        if current_fp is None:
            return 0.5
        if not fps:
            return 1.0
        sims = [tanimoto(current_fp, fp) for fp in fps if fp is not None]
        if not sims:
            return 1.0
        mx = max(sims)
        if mx > self.threshold:
            return max(0.0, 1.0 - (mx - self.threshold) / max(1e-6, 1.0 - self.threshold))
        return 1.0


class CompositeScorer:
    def __init__(self, config: Reinvent4Config):
        self.cfg = config
        self.validity = ValidityScorer()
        self.qed = QEDScorer()
        self.sa = SAScorer()
        self.binding = BindingScorer(config)
        self.diversity = DiversityScorer(config.diversity.intra_tanimoto_max)
        self.selectivity = SelectivityScorer(config.selectivity)
        self._pharm_cfg = PharmacophoreConfig(
            enabled=True,
            strictness="soft",
            enforce_hard_filters=False,
            target_name="EGFR",
        )
        self._pharm_profile = select_profile("EGFR", self._pharm_cfg)

    def _ring_complexity_bonus(self, mol: Chem.Mol | None) -> float:
        if mol is None:
            return 0.0
        n_rings = mol.GetRingInfo().NumRings()
        hetero_aromatic = 0
        for ring in mol.GetRingInfo().AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and any(
                mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16) for i in ring
            ):
                hetero_aromatic += 1
        aromatic = sum(1 for ring in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        complexity_pref = 1.0 - min(1.0, abs(n_rings - 3) / 3.0)  # moderate ring complexity
        hetero_pref = min(1.0, hetero_aromatic / max(1, aromatic))
        return max(0.0, min(1.0, 0.6 * complexity_pref + 0.4 * hetero_pref))

    def prefilter_shortlist(
        self,
        smiles_list: list[str],
        binding_model,
        protein_sequence: str,
        *,
        top_k: int,
    ) -> tuple[list[str], dict[str, float]]:
        """
        DeepPurpose shortlist accelerator for docking.
        Returns (shortlist_smiles, dp_score_map). Lower score = stronger.
        """
        if not smiles_list:
            return [], {}
        if binding_model is None or not protein_sequence:
            return list(smiles_list[:top_k]), {}
        try:
            batch_scores = binding_model.predict_batch(smiles_list, protein_sequence)
        except Exception as e:
            logger.warning("DeepPurpose prefilter failed: %s", e)
            return list(smiles_list[:top_k]), {}
        score_map: dict[str, float] = {}
        for s, r in zip(smiles_list, batch_scores):
            try:
                score_map[s] = float(parse_binding(r))
            except Exception:
                continue
        shortlist = sorted(
            [s for s in smiles_list if s in score_map],
            key=lambda smi: score_map[smi],
        )[: max(1, min(top_k, len(smiles_list)))]
        if not shortlist:
            shortlist = list(smiles_list[:top_k])
        return shortlist, score_map

    def docking_reward_vector(
        self,
        smiles_list: list[str],
        vina_scores: dict[str, float],
        docking_successes: dict[str, bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float | bool | None]]]:
        """
        Returns:
            raw_composite: (batch,) docking-guided scalar reward
            components: (batch, n_comp) where columns are [vina_norm, qed, sa, validity]
            metadata: per-smiles reward decomposition
        """
        rewards: list[float] = []
        comp_rows: list[list[float]] = []
        metadata: dict[str, dict[str, float | bool | None]] = {}
        scaffold_counts: dict[str, int] = {}
        scaffold_by_smi: dict[str, str] = {}

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                from rdkit.Chem.Scaffolds import MurckoScaffold

                sc = MurckoScaffold.GetScaffoldForMol(mol)
                sc_smi = Chem.MolToSmiles(sc) if sc is not None else ""
            except Exception:
                sc_smi = ""
            scaffold_by_smi[smi] = sc_smi
            if sc_smi:
                scaffold_counts[sc_smi] = scaffold_counts.get(sc_smi, 0) + 1

        # Fallback for docking failures: worst successful reward in batch.
        successful_vina_norms = [
            normalize_vina_score(vs)
            for smi, vs in vina_scores.items()
            if (docking_successes is None or docking_successes.get(smi, True))
        ]
        worst_success = min(successful_vina_norms) if successful_vina_norms else self.cfg.docking_rl.failed_docking_reward_floor

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            validity_val = self.validity.score(mol)

            if mol is None:
                rewards.append(0.0)
                comp_rows.append([0.0, 0.0, 0.0, 0.0])
                metadata[smi] = {
                    "vina_score": None,
                    "pose_energy": None,
                    "vina_norm": 0.0,
                    "qed": 0.0,
                    "sa": 0.0,
                    "validity": 0.0,
                    "docking_success": False,
                    "reward": 0.0,
                }
                continue

            q = self.qed.score(mol)
            s = self.sa.score(mol)
            ring_bonus = self._ring_complexity_bonus(mol)
            vina_raw = vina_scores.get(smi)
            dock_ok = docking_successes.get(smi, True) if docking_successes is not None else (vina_raw is not None)
            if vina_raw is not None and dock_ok:
                vina_norm = normalize_vina_score(vina_raw)
            else:
                vina_norm = float(worst_success)
                dock_ok = False

            pharm_result = evaluate_smiles(smi, self._pharm_profile, self._pharm_cfg)
            pharm_bonus = float(pharm_result.score)

            sc = scaffold_by_smi.get(smi, "")
            sc_freq = float(scaffold_counts.get(sc, 1)) if sc else 1.0
            diversity_bonus = 1.0 / (1.0 + sc_freq - 1.0)

            pose_stability_bonus = 0.0
            if dock_ok and vina_raw is not None:
                pose_stability_bonus = min(1.0, max(0.0, ((-float(vina_raw)) - 5.5) / 4.5))

            R = compose_docking_rl_reward(
                normalized_vina=vina_norm,
                qed=q,
                sa_norm=s,
                pharmacophore_bonus=0.6 * pharm_bonus + 0.4 * ring_bonus,
                diversity_bonus=diversity_bonus,
                pose_stability_bonus=pose_stability_bonus,
                cfg=self.cfg.docking_rl,
            )
            rewards.append(R)
            comp_rows.append([vina_norm, q, s, validity_val])
            metadata[smi] = {
                "vina_score": float(vina_raw) if vina_raw is not None else None,
                "pose_energy": float(vina_raw) if vina_raw is not None else None,
                "vina_norm": float(vina_norm),
                "qed": float(q),
                "sa": float(s),
                "pharmacophore_bonus": float(pharm_bonus),
                "ring_complexity_bonus": float(ring_bonus),
                "diversity_bonus": float(diversity_bonus),
                "pose_stability_bonus": float(pose_stability_bonus),
                "validity": float(validity_val),
                "docking_success": bool(dock_ok),
                "reward": float(R),
            }

        return (
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(comp_rows, dtype=torch.float32),
            metadata,
        )

    # Backward-compat alias used by older callsites.
    def reward_vector(
        self,
        smiles_list: list[str],
        seed_fp,
        binding_model,
        protein_sequence: str,
        weight_override: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Without docking info, this path uses neutral/fallback docking term.
        rewards, comps, _ = self.docking_reward_vector(smiles_list, vina_scores={}, docking_successes={})
        return rewards, comps
