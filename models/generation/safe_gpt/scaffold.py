from __future__ import annotations

from dataclasses import dataclass
import logging

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

MIN_HEAVY_ATOMS = 8


@dataclass
class ScaffoldStats:
    ring_count: int
    heavy_atoms: int
    scaffold_smiles: str


@dataclass
class ScaffoldValidationResult:
    valid: bool
    sanitized_smiles: str | None
    safe_scaffold: str | None
    slicer_used: str | None = None  # NEW: which slicer succeeded
    reason: str | None = None


def smiles_to_safe_scaffold(smiles: str, slicer: str = "brics") -> str | None:
    try:
        import safe as sf  # type: ignore[import]
        converter = sf.SAFEConverter(slicer=slicer)
        return converter.encoder(smiles)
    except Exception:
        return None


def validate_and_encode_scaffold(smiles: str, slicer: str = "brics") -> ScaffoldValidationResult:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ScaffoldValidationResult(False, None, None, None, "rdkit_parse_failed")
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        return ScaffoldValidationResult(False, None, None, None, "remove_hs_failed")
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    if heavy_atoms < MIN_HEAVY_ATOMS:
        return ScaffoldValidationResult(False, None, None, None, "too_few_heavy_atoms")
    if mol.GetRingInfo().NumRings() <= 0:
        return ScaffoldValidationResult(False, None, None, None, "ring_absent")
    try:
        sanitized = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ScaffoldValidationResult(False, None, None, None, "canonicalization_failed")

    # Single-attempt policy: do not retry multiple slicers.
    safe_scaffold = smiles_to_safe_scaffold(sanitized, slicer=slicer)
    if safe_scaffold:
        logger.debug("[SAFE] Scaffold validated and encoded with slicer=%s", slicer)
        return ScaffoldValidationResult(True, sanitized, safe_scaffold, slicer, None)

    logger.warning(
        "[SAFE] Scaffold encoding failed for scaffold=%s slicer=%s",
        sanitized,
        slicer,
    )
    return ScaffoldValidationResult(False, sanitized, None, None, "safe_tokenizer_precheck_failed")


def encode_with_brics(smiles: str) -> str | None:
    return smiles_to_safe_scaffold(smiles, slicer="brics")


def encode_with_recap(smiles: str) -> str | None:
    return smiles_to_safe_scaffold(smiles, slicer="recap")


def encode_with_mmpa(smiles: str) -> str | None:
    return smiles_to_safe_scaffold(smiles, slicer="mmpa")


def encode_with_attachment_points(smiles: str) -> str | None:
    # SAFE "attach" slicer preserves attachment-site encoding for decoration.
    return smiles_to_safe_scaffold(smiles, slicer="attach")


def murcko_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf)


def scaffold_stats(smiles: str) -> ScaffoldStats:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ScaffoldStats(0, 0, "")
    return ScaffoldStats(
        ring_count=mol.GetRingInfo().NumRings(),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        scaffold_smiles=murcko_scaffold_smiles(smiles),
    )