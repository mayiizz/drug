"""
Lightweight local descriptor-based ADMET heuristics.

This module provides descriptor-based filtering and preliminary safety
heuristics for post-generation triage.

It is NOT intended as a replacement for:
- experimental ADMET assays
- PBPK modeling
- comprehensive toxicity prediction

External predictive sources (for example ADMETlab and Tox21) remain the
primary predictive safety signals in CURA.
"""

from __future__ import annotations

import base64
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski
from rdkit.Chem import rdMolDescriptors


class LocalADMETHeuristics:
    """Descriptor-based local heuristics for drug-likeness and safety triage."""

    def predict(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        def _has(smarts: str) -> bool:
            patt = Chem.MolFromSmarts(smarts)
            return bool(patt is not None and mol.HasSubstructMatch(patt))

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)  # type: ignore[attr-defined]

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if h_donors > 5:
            violations += 1
        if h_acceptors > 10:
            violations += 1

        # Simple descriptor heuristics (not predictive ADMET models).
        reactive_alert = any(
            _has(s) for s in (
                "[NX2]=[O]",       # nitroso
                "[CX3H1](=O)[#6]", # aldehyde
                "C=CC(=O)[#6]",    # michael acceptor
            )
        )
        toxicity = 1 if (logp > 5.5 or tpsa > 145 or reactive_alert) else 0
        solubility = 1 if logp < 3 else 0
        absorption = 1 if tpsa < 140 else 0
        metabolism = 1 if mw < 500 else 0
        lipophilicity_risk = 1 if (logp > 5.8 or (aromatic_rings >= 5 and logp > 4.8)) else 0

        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_str = f"data:image/png;base64,{img_str}"
        except Exception:
            image_str = ""

        return {
            "molecular_weight": mw,
            "logp": logp,
            "tpsa": tpsa,
            "h_donors": h_donors,
            "h_acceptors": h_acceptors,
            "rotatable_bonds": rotatable_bonds,
            "lipinski_violations": violations,
            "drug_likeness": 0.85 if violations == 0 else 0.45,
            "toxicity": toxicity,
            "solubility": solubility,
            "absorption": absorption,
            "metabolism": metabolism,
            "lipophilicity_risk": lipophilicity_risk,
            "reactive_alert": int(reactive_alert),
            "image": image_str,
        }


# Backward compatibility alias (deprecated): use LocalADMETHeuristics.
ADMETPredictor = LocalADMETHeuristics

