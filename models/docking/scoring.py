from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from models.docking.cache import (
    DockingCache,
    make_docking_cache_key,
    make_ligand_prep_cache_key,
)
from models.docking.config import VinaConfig
from models.docking.pocket import DockingBox
from models.docking.vina_engine import DockingResult, VinaEngine

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def vina_to_reward_sigmoid(vina_score_kcal_mol: float) -> float:
    """
    Convert Vina's (more negative = better) affinity into a reward in [0,1].
    Using: sigmoid(-vina_score)
    """
    # If vina_score = -7.0 => reward ~ sigmoid(7.0) ~ 0.999.
    return sigmoid(-float(vina_score_kcal_mol))


@dataclass(slots=True)
class VinaScoreResult:
    vina_score: Optional[float]
    docking_success: bool
    vina_reward: Optional[float]
    pose_path: Optional[str]
    structure_source: Optional[str] = None
    structure_confidence_label: Optional[str] = None


class VinaScorer:
    """
    Docking + score normalization used for ranking.
    """

    def __init__(
        self,
        cfg: VinaConfig,
        docking_cache: DockingCache,
        vina_engine: Optional[VinaEngine] = None,
    ):
        self.cfg = cfg
        self.cache = docking_cache
        self.vina_engine = vina_engine or VinaEngine(cfg)

    def _ligand_key(self, smiles: str) -> str:
        # Ligand prep key is deterministic on (smiles, cfg).
        return make_ligand_prep_cache_key(smiles, self.cfg)[:32]

    def score_smiles_batch(
        self,
        *,
        protein_id: str,
        protein_pdbqt_path: Path,
        box: DockingBox,
        smiles_list: list[str],
        ligand_preparer,
        structure_source: str | None = None,
        structure_confidence_label: str | None = None,
    ) -> dict[str, VinaScoreResult]:
        """
        Score a list of SMILES using Vina.

        `ligand_preparer` is injected to keep this module independent
        from the prep backend (MGLTools/OpenBabel).
        """
        results: dict[str, VinaScoreResult] = {}
        attempted = len(smiles_list)
        prep_failed = 0
        cached_hits = 0
        failed = 0
        started_at = time.time()

        for smi in smiles_list:
            try:
                smi = ligand_preparer.clean_ligand_smiles(smi)
            except Exception:
                pass
            ligand_key = self._ligand_key(smi)
            ligand_pdbqt_path = self.cache.ligand_path(ligand_key)
            prep_input_smiles = smi

            try:
                # Prepare ligand PDBQT (cached when possible).
                ligand_preparer.prepare_ligand_pdbqt(
                    smiles=prep_input_smiles,
                    ligand_key=ligand_key,
                    output_path=ligand_pdbqt_path,
                )
            except Exception as e:
                # One retry with cleaned/desalted canonical ligand representation.
                retried = False
                try:
                    cleaned = ligand_preparer.clean_ligand_smiles(smi)
                    if cleaned and cleaned != smi:
                        retried = True
                        logger.warning("[VINA] Retrying ligand prep with cleaned SMILES original=%s cleaned=%s", smi, cleaned)
                        ligand_key_retry = self._ligand_key(cleaned)
                        ligand_pdbqt_path = self.cache.ligand_path(ligand_key_retry)
                        ligand_preparer.prepare_ligand_pdbqt(
                            smiles=cleaned,
                            ligand_key=ligand_key_retry,
                            output_path=ligand_pdbqt_path,
                        )
                        prep_input_smiles = cleaned
                except Exception as e2:
                    logger.warning("[VINA] Ligand prep retry failed smiles=%s reason=%s", smi, e2)
                if not retried or not ligand_pdbqt_path.exists():
                    prep_failed += 1
                    failed += 1
                    results[smi] = VinaScoreResult(
                        vina_score=None,
                        docking_success=False,
                        vina_reward=None,
                        pose_path=None,
                        structure_source=structure_source,
                        structure_confidence_label=structure_confidence_label,
                    )
                    continue

            docking_key = make_docking_cache_key(
                protein_id=protein_id,
                smiles=prep_input_smiles,
                box={"center": box.center, "size": box.size},
                cfg=self.cfg,
            )

            pose_path = self.cache.docking_result_path(docking_key).with_suffix(".pdbqt")
            docking_json_path = self.cache.docking_result_path(docking_key)

            cached = self.cache.has_docking_result(docking_key)
            if cached:
                try:
                    cached_payload = docking_json_path.read_text(encoding="utf-8", errors="ignore")
                    # We don't validate JSON strictly here; if it's broken, we'll re-run docking.
                    import json

                    obj = json.loads(cached_payload)
                    vina_score = obj.get("affinity_kcal_mol")
                    success = bool(obj.get("success", False))
                    results[smi] = VinaScoreResult(
                        vina_score=vina_score,
                        docking_success=success,
                        vina_reward=vina_to_reward_sigmoid(vina_score)
                        if isinstance(vina_score, (int, float))
                        else None,
                        pose_path=obj.get("pose_path"),
                        structure_source=structure_source,
                        structure_confidence_label=structure_confidence_label,
                    )
                    cached_hits += 1
                    continue
                except Exception:
                    pass

            docking_result: DockingResult = self.vina_engine.dock(
                receptor_pdbqt=protein_pdbqt_path,
                ligand_pdbqt=ligand_pdbqt_path,
                box=box,
                pose_output_path=pose_path,
            )

            vina_reward = None
            if docking_result.affinity_kcal_mol is not None:
                vina_reward = vina_to_reward_sigmoid(docking_result.affinity_kcal_mol)

            results[smi] = VinaScoreResult(
                vina_score=docking_result.affinity_kcal_mol,
                docking_success=docking_result.success,
                vina_reward=vina_reward,
                pose_path=docking_result.pose_path,
                structure_source=structure_source,
                structure_confidence_label=structure_confidence_label,
            )
            if not docking_result.success:
                failed += 1

            # Persist docking payload for later reuse.
            try:
                docking_json_path.parent.mkdir(parents=True, exist_ok=True)
                import json

                docking_json_path.write_text(
                    json.dumps(
                        {
                            "success": docking_result.success,
                            "affinity_kcal_mol": docking_result.affinity_kcal_mol,
                            "pose_path": docking_result.pose_path,
                            "raw_output": docking_result.raw_output,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                # Cache write failures should not crash ranking.
                pass

        successful = sum(1 for r in results.values() if r.docking_success)
        affinities = [float(r.vina_score) for r in results.values() if r.vina_score is not None]
        mean_affinity = (sum(affinities) / len(affinities)) if affinities else None
        best_affinity = min(affinities) if affinities else None
        vina_failed = max(0, failed - prep_failed)
        runtime_sec = time.time() - started_at
        logger.info("================================================")
        logger.info("DOCKING SUMMARY")
        logger.info("===============")
        logger.info("attempted: %d", attempted)
        logger.info("successful: %d", successful)
        logger.info("failed: %d", failed)
        logger.info("prep_failed: %d", prep_failed)
        logger.info("vina_failed: %d", vina_failed)
        logger.info("mean_affinity: %s", f"{mean_affinity:.4f}" if mean_affinity is not None else "NA")
        logger.info("best_affinity: %s", f"{best_affinity:.4f}" if best_affinity is not None else "NA")
        logger.info("==============")
        logger.info(
            "[VINA] Batch summary attempted=%d successful=%d failed=%d prep_failed=%d cached_hits=%d mean_affinity=%s best_affinity=%s runtime_sec=%.3f",
            attempted,
            successful,
            failed,
            prep_failed,
            cached_hits,
            f"{mean_affinity:.4f}" if mean_affinity is not None else "NA",
            f"{best_affinity:.4f}" if best_affinity is not None else "NA",
            runtime_sec,
        )
        return results

