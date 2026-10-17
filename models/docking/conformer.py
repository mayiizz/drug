from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from models.docking.config import VinaConfig
from models.utils.smiles_validation import validate_smiles_detailed
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformerMetadata:
    converged: bool
    mmff_energy: float
    energy_per_heavy_atom: float
    embedded: bool
    rejection_reason: str | None = None


def _mmff_energy_for_mol(mol_h) -> float:
    ff = AllChem.MMFFGetMoleculeForceField(mol_h, AllChem.MMFFGetMoleculeProperties(mol_h))
    if ff is None:
        raise RuntimeError("MMFF force field not available for molecule.")
    return float(ff.CalcEnergy())


def generate_conformer(
    smiles: str,
    cfg: VinaConfig,
    random_seed: int = 42,
    max_embed_attempts: int | None = None,
) -> tuple[Chem.Mol, ConformerMetadata]:
    """
    Generate a single 3D conformer (ETKDG + MMFF94 minimize).

    Returns:
      (mol_with_conformer, metadata)
    """
    ok, mol, reason = validate_smiles_detailed(smiles)
    if not ok or mol is None:
        logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Embed using ETKDGv3 + deterministic seed.
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.enforceChirality = True

    embedded = False
    last_err: Optional[Exception] = None
    attempts = int(max_embed_attempts or cfg.max_embed_attempts)
    for _ in range(max(1, attempts)):
        try:
            res = AllChem.EmbedMolecule(mol, params)
            if res == -1:
                continue
            embedded = True
            last_err = None
            break
        except Exception as e:  # pragma: no cover (depends on RDKit behavior)
            last_err = e
            continue

    if not embedded:
        if cfg.conformer_log_failures:
            logger.warning("Conformer embedding failed for %s", smiles)
        raise RuntimeError(f"3D embedding failed for SMILES={smiles}. err={last_err}")

    # MMFF94 energy minimization.
    ff = AllChem.MMFFGetMoleculeForceField(
        mol,
        AllChem.MMFFGetMoleculeProperties(mol),
    )
    if ff is None:
        raise RuntimeError("MMFF parameters not available.")

    mmff_energy_before = float(ff.CalcEnergy())
    # RDKit returns an integer status; we treat “converged” conservatively.
    try:
        status = ff.Minimize(maxIts=cfg.mmff_max_iters)
    except Exception:
        metadata = ConformerMetadata(
            converged=False,
            mmff_energy=mmff_energy_before,
            energy_per_heavy_atom=mmff_energy_before / max(1, mol.GetNumHeavyAtoms()),
            embedded=True,
            rejection_reason="mmff_minimization_failed",
        )
        if cfg.conformer_log_failures:
            logger.warning(
                "Conformer MMFF minimization failed for %s (energy=%.3f)",
                smiles,
                mmff_energy_before,
            )
        return mol, metadata

    mmff_energy_after = float(ff.CalcEnergy())
    n_heavy = max(1, mol.GetNumHeavyAtoms())
    energy_per_atom = mmff_energy_after / n_heavy

    # Convergence heuristic: status==0 is usually “converged”; energy improved is also a good sign.
    converged = bool(status == 0) or (mmff_energy_after <= mmff_energy_before + 1e-6)
    rejection_reason: str | None = None
    if energy_per_atom > cfg.max_mmff_energy_per_atom:
        converged = False
        rejection_reason = "high_strain_energy_per_atom"
    if mmff_energy_after > cfg.max_mmff_total_energy:
        converged = False
        rejection_reason = "high_total_strain_energy"

    # Geometry sanity check: very short heavy-atom distances indicate clashes.
    try:
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            ai = mol.GetAtomWithIdx(i)
            if ai.GetAtomicNum() == 1:
                continue
            pi = conf.GetAtomPosition(i)
            for j in range(i + 1, mol.GetNumAtoms()):
                aj = mol.GetAtomWithIdx(j)
                if aj.GetAtomicNum() == 1:
                    continue
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue
                pj = conf.GetAtomPosition(j)
                d2 = (pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2
                if d2 < 1.0:  # distance < 1.0 A
                    converged = False
                    rejection_reason = "nonbonded_heavy_atom_clash"
                    break
            if not converged and rejection_reason == "nonbonded_heavy_atom_clash":
                break
    except Exception:
        pass

    metadata = ConformerMetadata(
        converged=converged,
        mmff_energy=mmff_energy_after,
        energy_per_heavy_atom=energy_per_atom,
        embedded=True,
        rejection_reason=rejection_reason,
    )
    if cfg.conformer_log_failures:
        logger.info(
            "Conformer result %s: converged=%s energy=%.3f e_per_atom=%.3f reason=%s",
            smiles[:40],
            converged,
            mmff_energy_after,
            energy_per_atom,
            rejection_reason,
        )
    return mol, metadata

