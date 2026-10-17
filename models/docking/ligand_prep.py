from __future__ import annotations
 
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
 
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
import logging
from rdkit.Chem.MolStandardize import rdMolStandardize
 
from models.docking.config import (
    VinaConfig,
    is_mgltools_available,
    is_openbabel_available,
    resolve_openbabel_executable,
)
from models.docking.conformer import generate_conformer
from models.docking.utils import run_subprocess, TempDir
from models.utils.smiles_validation import validate_smiles_detailed
 
logger = logging.getLogger(__name__)
 
 
class LigandPrepError(RuntimeError):
    pass
 
 
@dataclass
class LigandPrepBackend:
    name: str
 
 
class LigandPreparer:
    """
    Prepare ligand PDBQT from SMILES:
      SMILES → 3D conformer → SDF → PDBQT
 
    Prep backend order: MGLTools primary → OpenBabel fallback.
    """
 
    def __init__(self, cfg: VinaConfig):
        self.cfg = cfg
        self._lfc = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger()
 
    def keep_largest_fragment(self, smiles: str) -> str:
        frags = (smiles or "").split(".")
        mols = [(validate_smiles_detailed(f)[1], f) for f in frags]
        mols = [(m, f) for m, f in mols if m is not None]
        if not mols:
            return smiles
        return max(mols, key=lambda x: x[0].GetNumHeavyAtoms())[1]
 
    def clean_ligand_smiles(self, smiles: str) -> str:
        stripped = self.keep_largest_fragment(smiles)
        ok, raw_mol, reason = validate_smiles_detailed(stripped)
        if not ok or raw_mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", stripped, reason)
            raise LigandPrepError("Invalid ligand SMILES before prep.")
        try:
            mol = rdMolStandardize.Cleanup(raw_mol)
        except Exception:
            mol = raw_mol
 
        # Force single largest fragment (desalting/counterion stripping).
        try:
            largest = self._lfc.choose(mol)
        except Exception:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                raise LigandPrepError("No fragments found after parsing ligand.")
            largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
 
        # Neutralize where reasonable.
        try:
            largest = self._uncharger.uncharge(largest)
        except Exception:
            pass
        return Chem.MolToSmiles(largest, canonical=True)
 
    def _validate_ligand_pdbqt(self, path: Path) -> tuple[bool, str]:
        """Returns (is_valid, reason_if_invalid)."""
        if not path.exists() or path.stat().st_size <= 0:
            return False, "file_missing_or_empty"
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if len(txt.splitlines()) < 10:
            return False, "too_few_lines"
        has_root = "ROOT" in txt
        has_endroot = "ENDROOT" in txt
        has_atom = "ATOM" in txt or "HETATM" in txt
        has_torsdof = "TORSDOF" in txt
        if not has_root:
            return False, "missing_ROOT"
        if not has_endroot:
            return False, "missing_ENDROOT"
        if not has_atom:
            return False, "missing_ATOM_records"
        if not has_torsdof:
            return False, "missing_TORSDOF"
        return True, ""
 
    def _write_sdf(self, mol: Chem.Mol, path: Path) -> None:
        """Write mol with 3D conformer to SDF — more reliable than PDB for OpenBabel input."""
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
 
    def prepare_ligand_pdbqt(
        self,
        *,
        smiles: str,
        ligand_key: str,
        output_path: Path,
    ) -> None:
        if output_path.exists() and output_path.stat().st_size > 0:
            valid, _ = self._validate_ligand_pdbqt(output_path)
            if valid:
                return
            # Cached file is stale/corrupt — regenerate.
            output_path.unlink(missing_ok=True)
 
        # Pre-docking desalting + cleanup + neutralization + canonicalization.
        ok, raw_mol, reason = validate_smiles_detailed(smiles)
        if not ok or raw_mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
            raise LigandPrepError("Invalid ligand SMILES before prep.")
        frag_count_before = len(Chem.GetMolFrags(raw_mol, asMols=False))
        clean_smiles = self.clean_ligand_smiles(smiles)
        ok, clean_mol, reason = validate_smiles_detailed(clean_smiles)
        if not ok or clean_mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", clean_smiles, reason)
            raise LigandPrepError("Canonicalized ligand failed RDKit parse.")
        frag_count_after = len(Chem.GetMolFrags(clean_mol, asMols=False))
        if frag_count_after > 1:
            logger.warning("[VINA] Rejected multi-fragment ligand smiles=%s", clean_smiles)
            raise LigandPrepError("Rejected multi-fragment ligand")
        formal_charge = sum(a.GetFormalCharge() for a in clean_mol.GetAtoms())
        logger.info(
            "[VINA] Ligand validation canonical_smiles=%s heavy_atoms=%d fragment_count=%d->%d formal_charge=%d rotatable_bonds=%d",
            clean_smiles,
            clean_mol.GetNumHeavyAtoms(),
            frag_count_before,
            frag_count_after,
            formal_charge,
            int(rdMolDescriptors.CalcNumRotatableBonds(clean_mol)),
        )
 
        with TempDir(base_dir=self.cfg.tmp_dir) as tmpdir:
            ligand_pdb = tmpdir / f"{ligand_key}.pdb"
            ligand_sdf = tmpdir / f"{ligand_key}.sdf"
 
            mol, meta = generate_conformer(
                smiles=clean_smiles,
                cfg=self.cfg,
                random_seed=42,
            )
            if not meta.converged:
                raise LigandPrepError(
                    "Conformer rejected: "
                    f"reason={meta.rejection_reason or 'non_converged'} "
                    f"mmff_energy={meta.mmff_energy:.3f} "
                    f"energy_per_atom={meta.energy_per_heavy_atom:.3f}"
                )
 
            # Write SDF (preferred over PDB — preserves bond order for OpenBabel).
            try:
                self._write_sdf(mol, ligand_sdf)
            except Exception as e:
                raise LigandPrepError(f"SDF write failed: {e}")
 
            # Also write PDB as fallback input for MGLTools.
            try:
                Chem.MolToPDBFile(mol, str(ligand_pdb))
            except Exception as e:
                raise LigandPrepError(f"MolToPDBFile failed: {e}")
 
            pdbqt_ok = False
            last_reason: Optional[str] = None
 
            # ── PRIMARY: MGLTools ────────────────────────────────────────────
            if is_mgltools_available():
                backend = self.cfg.mgltools_prepare_ligand4
                script = backend or shutil.which("prepare_ligand4.py")
                if script:
                    cmd = [
                        sys.executable,
                        script,
                        "-l",
                        str(ligand_pdb),
                        "-o",
                        str(output_path),
                        "-A",
                        "hydrogens",
                    ]
                    proc = run_subprocess(cmd, timeout_sec=self.cfg.docking_timeout, cwd=tmpdir)
                    logger.info("[VINA] MGLTools command=%s", " ".join(cmd))
                    if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                        valid, reason = self._validate_ligand_pdbqt(output_path)
                        if valid:
                            pdbqt_ok = True
                        else:
                            logger.warning("[VINA] MGLTools produced invalid PDBQT reason=%s", reason)
                            last_reason = f"MGLTools PDBQT invalid: {reason}"
                    else:
                        logger.warning("[VINA] MGLTools conversion failed return_code=%s", proc.returncode)
                        logger.warning("[VINA] MGLTools stdout=\n%s", proc.stdout or "")
                        logger.warning("[VINA] MGLTools stderr=\n%s", proc.stderr or "")
                        last_reason = (proc.stdout or "") + "\n" + (proc.stderr or "")
 
            # ── FALLBACK: OpenBabel ──────────────────────────────────────────
            if not pdbqt_ok and is_openbabel_available():
                obabel = str(resolve_openbabel_executable())
 
                # Strategy 1: SDF → PDBQT with full AutoDock torsion-tree generation.
                # --gen3d is NOT needed since SDF already has 3D coords from RDKit.
                # -xr suppresses rigid output; we want the full torsion tree for ligands.
                # -h adds polar hydrogens required by AutoDock atom typing.
                cmd = [
                    obabel,
                    "-isdf", str(ligand_sdf),
                    "-opdbqt",
                    "-O", str(output_path),
                    "-h",
                    "--partialcharge", "gasteiger",
                    "-p", "7.4",
                  ]
                proc = run_subprocess(cmd, timeout_sec=self.cfg.docking_timeout, cwd=tmpdir)
                logger.info("[VINA] OpenBabel SDF->PDBQT command=%s", " ".join(cmd))
                logger.info("[VINA] OpenBabel stdout=\n%s", proc.stdout or "")
                if proc.stderr:
                    logger.info("[VINA] OpenBabel stderr=\n%s", proc.stderr)
 
                if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    valid, reason = self._validate_ligand_pdbqt(output_path)
                    if valid:
                        pdbqt_ok = True
                        logger.info("[VINA] OpenBabel SDF->PDBQT succeeded")
                    else:
                        logger.warning(
                            "[VINA] OpenBabel SDF->PDBQT produced invalid PDBQT reason=%s — trying PDB fallback",
                            reason,
                        )
                        last_reason = f"OpenBabel SDF->PDBQT invalid: {reason}"
                else:
                    logger.warning(
                        "[VINA] OpenBabel SDF->PDBQT failed return_code=%s — trying PDB fallback",
                        proc.returncode,
                    )
                    last_reason = (proc.stdout or "") + "\n" + (proc.stderr or "")
 
                # Strategy 2: PDB → PDBQT fallback (some OpenBabel builds handle PDB better).
                if not pdbqt_ok:
                    output_path.unlink(missing_ok=True)
                    cmd2 = [
                            obabel,
                            "-ipdb", str(ligand_pdb),
                            "-opdbqt",
                            "-O", str(output_path),
                            "-h",
                            "--partialcharge", "gasteiger",
                            "-p", "7.4",
                        ]
                    proc2 = run_subprocess(cmd2, timeout_sec=self.cfg.docking_timeout, cwd=tmpdir)
                    logger.info("[VINA] OpenBabel PDB->PDBQT command=%s", " ".join(cmd2))
                    logger.info("[VINA] OpenBabel stdout=\n%s", proc2.stdout or "")
                    if proc2.stderr:
                        logger.info("[VINA] OpenBabel stderr=\n%s", proc2.stderr)
 
                    if proc2.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                        valid, reason = self._validate_ligand_pdbqt(output_path)
                        if valid:
                            pdbqt_ok = True
                            logger.info("[VINA] OpenBabel PDB->PDBQT succeeded")
                        else:
                            logger.warning(
                                "[VINA] OpenBabel PDB->PDBQT produced invalid PDBQT reason=%s",
                                reason,
                            )
                            last_reason = f"OpenBabel PDB->PDBQT invalid: {reason}\nstdout={proc2.stdout}\nstderr={proc2.stderr}"
                    else:
                        last_reason = (
                            f"OpenBabel PDB->PDBQT failed return_code={proc2.returncode}\n"
                            f"stdout={proc2.stdout}\nstderr={proc2.stderr}"
                        )
 
            if not pdbqt_ok:
                logger.error("[VINA] All ligand prep strategies failed for smiles=%s reason=%s", smiles, last_reason)
                raise LigandPrepError(f"Failed to produce ligand PDBQT. reason={last_reason}")
 
            logger.info("[VINA] Ligand prepared successfully path=%s", output_path)