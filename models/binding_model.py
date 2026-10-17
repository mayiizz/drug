# Backward-compatibility shim
# Original: models/binding_model.py
# New location: models/binding/binding_model.py
from models.binding.binding_model import BindingPredictor, DeepPurposeBindingPredictor

__all__ = ["DeepPurposeBindingPredictor", "BindingPredictor"]
