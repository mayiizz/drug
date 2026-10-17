"""Central SMILES validation helpers used by every featurizer entry point.

Every model that takes generated SMILES as input (Tox21 RF screen, ADMET
descriptors, DeepPurpose binding pre-filter, multi-objective ranker, ...)
should call :func:`validate_smiles_strict` *before* doing any featurization.
This guarantees:

* ``Chem.MolFromSmiles`` returned a valid Mol (parser passed).
* ``Chem.SanitizeMol`` succeeded so downstream calls like
  ``CanonicalRankAtoms``, ``GetSSSR``, or ``MolToSmiles`` cannot blow up on a
  half-initialised RDKit object.

Rejections are logged at WARNING level with the ``[VALIDATION]`` prefix so
they can be grepped from runtime logs without parsing JSON.
"""

from __future__ import annotations

import logging
from typing import Iterable

from rdkit import Chem

logger = logging.getLogger(__name__)

__all__ = [
    "validate_smiles_strict",
    "validate_smiles_detailed",
    "filter_valid_smiles",
    "validate_smiles_batch",
]


def validate_smiles_strict(smi: object) -> "Chem.Mol | None":
    """Return a sanitized RDKit Mol or ``None`` if ``smi`` is unusable.

    Parameters
    ----------
    smi:
        Candidate SMILES. May be ``None``, empty, or non-string; all of those
        are treated as rejections.

    Notes
    -----
    The function never raises. It logs at WARNING level on rejection so the
    caller's tight loop can simply ``continue`` on ``None``.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None

    raw = smi.strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        logger.warning("[VALIDATION] rejected invalid molecule: %r", raw[:120])
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit raises a wide variety
        logger.warning(
            "[VALIDATION] sanitize failed (%s): %r",
            exc.__class__.__name__,
            raw[:120],
        )
        return None

    return mol


def validate_smiles_detailed(smi: object) -> tuple[bool, "Chem.Mol | None", str]:
    """Return ``(ok, mol, reason)`` for callers that need rejection reasons."""
    if not isinstance(smi, str) or not smi.strip():
        return False, None, "empty_or_non_string"

    raw = smi.strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        logger.warning("[VALIDATION] rejected invalid molecule: %r", raw[:120])
        return False, None, "parse_failed"

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit raises a wide variety
        logger.warning(
            "[VALIDATION] sanitize failed (%s): %r",
            exc.__class__.__name__,
            raw[:120],
        )
        return False, None, "sanitize_failed"

    return True, mol, "ok"


def filter_valid_smiles(
    smiles_list: Iterable[str],
    *,
    return_mols: bool = False,
) -> "list[str] | tuple[list[str], list[Chem.Mol]]":
    """Return the subset of ``smiles_list`` that pass strict validation.

    When ``return_mols=True`` returns a paired list of (smiles, mols).
    """
    kept: list[str] = []
    mols: list[Chem.Mol] = []
    for smi in smiles_list:
        mol = validate_smiles_strict(smi)
        if mol is None:
            continue
        kept.append(smi)
        if return_mols:
            mols.append(mol)
    if return_mols:
        return kept, mols
    return kept


def validate_smiles_batch(
    smiles_list: Iterable[str],
    *,
    label: str = "FEATURIZER",
) -> tuple[list[str], list[Chem.Mol]]:
    """Validate a batch and emit a single ``[<label>]`` summary log line.

    Useful right before a DeepChem / DeepPurpose / ADMET batch call where a
    single ``[FEATURIZER] skipped N invalid SMILES`` line is more useful than
    one warning per molecule.
    """
    smiles_list = list(smiles_list)
    kept_smi, kept_mol = [], []
    for smi in smiles_list:
        mol = validate_smiles_strict(smi)
        if mol is None:
            continue
        kept_smi.append(smi)
        kept_mol.append(mol)

    skipped = len(smiles_list) - len(kept_smi)
    if skipped:
        logger.info(
            "[%s] skipped %d/%d invalid SMILES before featurization",
            label,
            skipped,
            len(smiles_list),
        )
    return kept_smi, kept_mol
