import random


"""
Development-only placeholder binding model.

This class is NOT used for:
- scientific benchmarking
- docking-guided RL
- publication experiments

Real binding estimation in CURA uses:
- DeepPurpose neural pre-screening
- AutoDock Vina structure-based docking refinement
"""


class MockBindingPredictor:

    def predict(self, smiles, target="COX2"):

        # placeholder for real binding model
        # later you can plug DeepPurpose or docking

        binding_score = random.uniform(0, 10)

        return {
            "target": target,
            "binding_score": binding_score
        }


# Backward compatibility alias (deprecated): use MockBindingPredictor.
BindingPredictor = MockBindingPredictor