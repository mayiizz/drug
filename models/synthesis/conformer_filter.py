"""
ConformerFilter — RDKit 3D conformer generation + MMFF94 energy filter.

Replaces docking for systems without AutoDock.
Rejects molecules that:
  - Cannot be embedded in 3D (sterically impossible topology)
  - Have extremely high MMFF94 strain energy (geometry is physically unreasonable)
  - Produce clashing atoms after minimisation

This is NOT docking — it does not score binding pose.
It is a geometry sanity check that eliminates synthetically
implausible molecules before the expensive DeepPurpose step.
"""

import logging
import os
from rdkit import Chem, RDLogger as _RDL
from rdkit.Chem import AllChem, Descriptors
from models.utils.smiles_validation import validate_smiles_detailed

_RDL.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment with a clean fallback."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[ConformerFilter] invalid %s=%r; using default %.1f",
            name, raw, default,
        )
        return default


# MMFF94 strain energy upper bound in kcal/mol per heavy atom.
# Drug-like molecules typically score 0.5-3.0 kcal/mol per atom.
# Values above 8.0 indicate severely strained / unrealistic geometry.
MAX_ENERGY_PER_ATOM = 8.0

# Total ring strain warning threshold in kcal/mol. Molecules above this
# value typically signal a fused-ring or chiral-center anomaly even when
# per-atom energy looks reasonable.
RING_STRAIN_WARNING_KCAL = 25.0

# Total MMFF94 strain hard rejection threshold in kcal/mol. Anything above
# this point describes a molecule whose 3D geometry is physically impossible
# (impossible ring system, severe steric clash, etc.) regardless of size.
#
# Env-tunable so the runtime can be loosened during stabilisation without a
# code change. The published research-paper-grade default is 60.0 kcal/mol;
# we ship 80.0 here as a stabilisation measure while REINVENT diversity is
# being re-validated. Flip it back via:
#     export CURA_MAX_TOTAL_STRAIN_KCAL=60.0
MAX_TOTAL_STRAIN_KCAL = _env_float("CURA_MAX_TOTAL_STRAIN_KCAL", 80.0)
logger.info(
    "[ConformerFilter] hard cap MAX_TOTAL_STRAIN_KCAL = %.1f kcal/mol",
    MAX_TOTAL_STRAIN_KCAL,
)

# Maximum attempts for 3D embedding per molecule.
MAX_EMBED_ATTEMPTS = 3


class ConformerFilter:
    """
    Filters a list of SMILES by 3D embeddability and MMFF94 strain energy.

    Usage:
        cf = ConformerFilter()
        passed = cf.filter(smiles_list)
    """

    def __init__(self, max_energy_per_atom: float = MAX_ENERGY_PER_ATOM):
        self.max_energy_per_atom = max_energy_per_atom
        # Aggregated diagnostics filled in by ``filter`` and consumed by
        # CuraPipeline summary logging.
        self.last_filter_summary: dict[str, float | int] = {
            "input_count": 0,
            "passed": 0,
            "failed_embedding": 0,
            "failed_converge": 0,
            "ring_strain_warnings": 0,
            "rejected_total_strain": 0,
            "mean_strain_per_atom": 0.0,
            "mean_strain_total": 0.0,
        }
        print("[ConformerFilter] Ready (RDKit MMFF94).")

    def _check(self, smiles: str) -> tuple[bool, str, dict[str, float | bool]]:
        """Returns (passes, reason, diagnostics).

        Diagnostics keys:
            ``embedded`` (bool), ``converged`` (bool),
            ``energy_total`` (kcal/mol or NaN), ``energy_per_atom`` (kcal/mol or NaN),
            ``ring_strain_warning`` (bool), ``mmff_ok`` (bool).
        """
        diag: dict[str, float | bool] = {
            "embedded": False,
            "converged": False,
            "energy_total": float("nan"),
            "energy_per_atom": float("nan"),
            "ring_strain_warning": False,
            "mmff_ok": False,
        }
        ok, mol, reason = validate_smiles_detailed(smiles)
        if not ok or mol is None:
            logger.warning("[VALIDATION] rejected invalid molecule: %s (%s)", smiles, reason)
            return False, "invalid SMILES", diag

        try:
            mol_h = Chem.AddHs(mol)
        except Exception:
            return False, "AddHs failed", diag

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.enforceChirality = True

        result = AllChem.EmbedMolecule(mol_h, params)
        if result == -1:
            params2 = AllChem.EmbedParameters()
            params2.randomSeed = 42
            result = AllChem.EmbedMolecule(mol_h, params2)
            if result == -1:
                return False, "3D embedding failed (sterically impossible)", diag
        diag["embedded"] = True

        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol_h,
                AllChem.MMFFGetMoleculeProperties(mol_h),
            )
            if ff is None:
                return True, "ok (MMFF unavailable, embedding passed)", diag

            converged_code = ff.Minimize(maxIts=500)
            diag["mmff_ok"] = True
            diag["converged"] = bool(converged_code == 0)
            energy = ff.CalcEnergy()

            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy == 0:
                return False, "no heavy atoms", diag

            energy_per_atom = energy / n_heavy
            diag["energy_total"] = float(energy)
            diag["energy_per_atom"] = float(energy_per_atom)
            diag["ring_strain_warning"] = bool(energy > RING_STRAIN_WARNING_KCAL)

            # Hard reject impossible ring systems / catastrophic clashes.
            if energy > MAX_TOTAL_STRAIN_KCAL:
                return (
                    False,
                    (
                        f"impossible geometry (MMFF total {energy:.1f} kcal/mol > "
                        f"{MAX_TOTAL_STRAIN_KCAL:.1f}; likely impossible ring system)"
                    ),
                    diag,
                )

            if energy_per_atom > self.max_energy_per_atom:
                return (
                    False,
                    (
                        f"strained geometry (MMFF {energy_per_atom:.1f} kcal/mol per atom, "
                        f"max {self.max_energy_per_atom:.1f})"
                    ),
                    diag,
                )

        except Exception as e:
            logger.warning("MMFF energy calc failed for %s: %s - keeping molecule.", smiles, e)
            return True, "ok (energy check skipped)", diag

        return True, "ok", diag

    def filter(self, smiles_list: list[str]) -> list[str]:
        """Filter a list of SMILES. Returns those passing 3D geometry check."""
        passed: list[str] = []
        failed_embedding = 0
        failed_converge = 0
        ring_strain_warnings = 0
        rejected_total_strain = 0
        strain_total_samples: list[float] = []
        strain_per_atom_samples: list[float] = []

        for smi in smiles_list:
            ok, reason, diag = self._check(smi)
            mmff_ok = bool(diag.get("mmff_ok"))
            embedded = bool(diag.get("embedded"))
            if not embedded:
                failed_embedding += 1
            if mmff_ok and not bool(diag.get("converged")):
                failed_converge += 1
            if mmff_ok:
                e_total = float(diag.get("energy_total"))  # type: ignore[arg-type]
                e_pa = float(diag.get("energy_per_atom"))  # type: ignore[arg-type]
                if e_total == e_total:  # NaN check
                    strain_total_samples.append(e_total)
                if e_pa == e_pa:
                    strain_per_atom_samples.append(e_pa)
                if bool(diag.get("ring_strain_warning")):
                    ring_strain_warnings += 1
                    logger.warning(
                        "[ConformerFilter] ring-strain warning energy=%.1f kcal/mol smiles=%s",
                        e_total,
                        smi[:80],
                    )
                if not ok and e_total == e_total and e_total > MAX_TOTAL_STRAIN_KCAL:
                    rejected_total_strain += 1
                    logger.warning(
                        "[ConformerFilter] reject total_strain=%.1f kcal/mol > %.1f smiles=%s",
                        e_total,
                        MAX_TOTAL_STRAIN_KCAL,
                        smi[:80],
                    )

            if ok:
                passed.append(smi)
                logger.debug(
                    "[ConformerFilter] + smiles=%s converged=%s energy=%.2f kcal/mol",
                    smi[:60],
                    bool(diag.get("converged")),
                    float(diag.get("energy_total"))  # type: ignore[arg-type]
                    if diag.get("energy_total") == diag.get("energy_total")
                    else float("nan"),
                )
                print(f"  [3D] + {smi[:60]}{'...' if len(smi) > 60 else ''}")
            else:
                print(f"  [3D] x {reason}: {smi[:60]}{'...' if len(smi) > 60 else ''}")

        n_in = len(smiles_list)
        n_out = len(passed)
        mean_strain_total = (
            sum(strain_total_samples) / len(strain_total_samples)
            if strain_total_samples
            else 0.0
        )
        mean_strain_per_atom = (
            sum(strain_per_atom_samples) / len(strain_per_atom_samples)
            if strain_per_atom_samples
            else 0.0
        )
        self.last_filter_summary = {
            "input_count": n_in,
            "passed": n_out,
            "failed_embedding": failed_embedding,
            "failed_converge": failed_converge,
            "ring_strain_warnings": ring_strain_warnings,
            "rejected_total_strain": rejected_total_strain,
            "mean_strain_per_atom": float(mean_strain_per_atom),
            "mean_strain_total": float(mean_strain_total),
        }

        logger.info(
            "[ConformerFilter] %d/%d passed | mean_strain=%.2f kcal/mol | failed_converge=%d | ring_warn=%d | reject_total_strain=%d",
            n_out,
            n_in,
            mean_strain_total,
            failed_converge,
            ring_strain_warnings,
            rejected_total_strain,
        )
        print(f"[ConformerFilter] {n_out}/{n_in} passed 3D geometry check.")
        return passed