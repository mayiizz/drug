from __future__ import annotations

import logging

from rdkit import Chem

from benchmarks.utils import murcko_scaffold, preprocess_smiles

logger = logging.getLogger(__name__)


class MosesEvaluator:
    """Compute MOSES-style generative metrics with graceful degradation.

    When the optional ``moses`` (``molsets``) package is unavailable we
    still return validity / uniqueness / novelty / IntDiv1 - they are
    cheap and rdkit-only. FCD is reported as 0.0 in that case. To enable
    full MOSES + FCD support, install the extras::

        pip install -r requirements-benchmarks.txt
    """

    def __init__(self):
        try:
            from moses.metrics import metrics as _moses_metrics  # type: ignore

            self._moses_metrics = _moses_metrics
        except Exception as e:  # noqa: BLE001
            logger.info(
                "[MosesEvaluator] 'moses' (molsets) not installed (%s) - "
                "falling back to rdkit-only metrics; FCD will be 0.0.",
                e,
            )
            self._moses_metrics = None

    def evaluate(
        self,
        generated_smiles: list[str],
        train_smiles: list[str] | None = None,
        test_smiles: list[str] | None = None,
    ) -> dict[str, float]:
        gen = preprocess_smiles(generated_smiles)
        if not gen:
            return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0, "intdiv1": 0.0, "fcd": 0.0}
        valid = [s for s in gen if Chem.MolFromSmiles(s) is not None]
        uniq = len(set(valid)) / max(1, len(valid))
        novelty = 0.0
        if train_smiles:
            novelty = sum(1 for s in valid if s not in set(preprocess_smiles(train_smiles))) / max(1, len(valid))
        out = {
            "validity": len(valid) / max(1, len(gen)),
            "uniqueness": uniq,
            "novelty": float(novelty),
            "intdiv1": self._intdiv1(valid),
            "fcd": 0.0,
        }
        if self._moses_metrics is not None and test_smiles:
            try:
                m = self._moses_metrics.get_all_metrics(gen=valid, k=len(valid), test=test_smiles, train=train_smiles or [])
                out["fcd"] = float(m.get("FCD/Test", 0.0))
                out["intdiv1"] = float(m.get("IntDiv1", out["intdiv1"]))
            except Exception:
                pass
        return out

    def _intdiv1(self, smiles: list[str]) -> float:
        from benchmarks.utils import morgan_fp, tanimoto

        fps = [morgan_fp(s) for s in smiles]
        vals = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                if fps[i] is None or fps[j] is None:
                    continue
                vals.append(1.0 - tanimoto(fps[i], fps[j]))
        return float(sum(vals) / len(vals)) if vals else 0.0


def diversity_summary(smiles: list[str]) -> dict[str, float]:
    from benchmarks.utils import morgan_fp, tanimoto

    fps = [morgan_fp(s) for s in preprocess_smiles(smiles)]
    vals = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i] is None or fps[j] is None:
                continue
            vals.append(tanimoto(fps[i], fps[j]))
    scafs = {murcko_scaffold(s) for s in smiles if s}
    scafs.discard("")
    return {
        "pairwise_similarity_mean": float(sum(vals) / len(vals)) if vals else 0.0,
        "scaffold_diversity": float(len(scafs) / max(1, len(smiles))),
    }

