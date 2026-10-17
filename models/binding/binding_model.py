

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _safe_float(result) -> float:
    """Extract float safely from DeepPurpose output."""
    if result is None:
        raise ValueError("Binding model returned None")

    if isinstance(result, (int, float)):
        return float(result)

    return float(np.array(result).flat[0])


class DeepPurposeBindingPredictor:
    """
    DeepPurpose-based neural affinity approximation model.

    Used only for:
    - fast neural pre-screening
    - shortlist generation
    - approximate affinity prioritization

    Not used as:
    - authoritative binding affinity
    - docking replacement
    - experimental binding predictor
    """

    def __init__(self):
        print("[DeepPurposeBindingPredictor] Loading CNN_CNN_BindingDB on cpu...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from DeepPurpose import DTI as models
            self.model = models.model_pretrained(model="CNN_CNN_BindingDB")
            self._dp_available = True
            print("[DeepPurposeBindingPredictor] Model loaded.")

        except Exception as e:
            self._dp_available = False
            raise RuntimeError(f"DeepPurpose failed to load: {e}")

    # ------------------------------------------------------------------

    def predict(self, smiles: str, protein_seq: str) -> float:
        """
        Predict surrogate affinity approximation for a single molecule.

        Returns:
            raw log10(Kd) score
            lower value = stronger binding
        """

        if not self._dp_available:
            raise RuntimeError("DeepPurpose model not loaded.")

        from rdkit import Chem
        from DeepPurpose.utils import data_process

        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES: '{smiles}'")

        try:
            df = data_process(
                [smiles],
                [protein_seq],
                [0],
                drug_encoding="CNN",
                target_encoding="CNN",
                split_method="no_split",
            )

            raw = self.model.predict(df)
            score = _safe_float(raw)

            logger.debug(f"Binding score {score:.3f} for {smiles[:25]}")

            return score

        except Exception as e:
            raise ValueError(f"DeepPurpose prediction failed: {e}")

    # ------------------------------------------------------------------

    def predict_batch(self, smiles_list: list, protein_seq: str) -> list:
        """
        Batch surrogate affinity approximation.

        Returns list of raw log10(Kd) scores.
        Invalid molecules receive score = 999.
        """

        if not self._dp_available:
            raise RuntimeError("DeepPurpose model not loaded.")

        if not smiles_list:
            return []

        from rdkit import Chem
        from DeepPurpose.utils import data_process

        results = [999.0] * len(smiles_list)

        valid_smiles = []
        valid_idx = []

        # validate molecules
        for i, smi in enumerate(smiles_list):
            if Chem.MolFromSmiles(smi) is not None:
                valid_smiles.append(smi)
                valid_idx.append(i)

        if not valid_smiles:
            return results

        try:
            print(
                f"[DeepPurposeBindingPredictor] Batch predicting {len(valid_smiles)} molecules..."
            )

            df = data_process(
                valid_smiles,
                [protein_seq] * len(valid_smiles),
                [0] * len(valid_smiles),
                drug_encoding="CNN",
                target_encoding="CNN",
                split_method="no_split",
            )

            raw_arr = np.array(self.model.predict(df)).flatten()

            for idx, raw_val in zip(valid_idx, raw_arr):
                results[idx] = float(raw_val)

            valid_scores = [r for r in results if r < 900]

            print(
                f"[DeepPurposeBindingPredictor] Done. Score range: "
                f"{min(valid_scores):.3f} – {max(valid_scores):.3f}"
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")

            # fallback to individual predictions
            for i, smi in zip(valid_idx, valid_smiles):
                try:
                    results[i] = self.predict(smi, protein_seq)
                except Exception:
                    results[i] = 999.0

        return results


# Preferred explicit scientific name.
# Backward compatibility alias (deprecated): use DeepPurposeBindingPredictor.
BindingPredictor = DeepPurposeBindingPredictor

