# Backward-compatibility shim
# Original: models/binding_predictor.py
# New location: models/binding/binding_predictor.py
from models.binding.binding_predictor import BindingPredictor, MockBindingPredictor

__all__ = ["MockBindingPredictor", "BindingPredictor"]
