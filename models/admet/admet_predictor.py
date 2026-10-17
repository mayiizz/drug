"""
Deprecated compatibility module.

Use `models.admet.local_admet_heuristics.LocalADMETHeuristics` instead.
"""

from models.admet.local_admet_heuristics import ADMETPredictor, LocalADMETHeuristics

__all__ = ["LocalADMETHeuristics", "ADMETPredictor"]