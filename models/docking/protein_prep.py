from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.docking.config import (
    VinaConfig,
    is_mgltools_available,
    is_openbabel_available,
    resolve_openbabel_executable,
)
from models.docking.utils import TempDir, run_subprocess

logger = logging.getLogger(__name__)


class ProteinPrepError(RuntimeError):
    pass


@dataclass
class ProteinPrepBackend:
    name: str


class ProteinPreparer:
    """
    Prepare rigid receptor PDBQT from protein PDB.

    Backend order:
        1. MGLTools prepare_receptor4.py
        2. OpenBabel fallback (rigid-cleaned)

    IMPORTANT:
    Receptors MUST NOT contain:
        ROOT
        BRANCH
        ENDBRANCH
        TORSDOF

    Those are ligand-only torsion-tree tags.
    """

    def __init__(self, cfg: VinaConfig):
        self.cfg = cfg

    # ============================================================
    # Validation
    # ============================================================

    def _validate_receptor_pdbqt(self, path: Path) -> bool:

        if not path.exists():
            logger.error("[VINA] receptor.pdbqt missing: %s", path)
            return False

        if path.stat().st_size <= 0:
            logger.error("[VINA] receptor.pdbqt empty: %s", path)
            return False

        txt = path.read_text(encoding="utf-8", errors="ignore")

        # --------------------------------------------------------
        # CRITICAL:
        # receptor MUST NOT contain ligand torsion tags
        # --------------------------------------------------------

        forbidden = [
            "ROOT",
            "BRANCH",
            "ENDBRANCH",
            "TORSDOF",
        ]

        for tag in forbidden:
            if tag in txt:
                logger.error(
                    "[VINA] Invalid receptor PDBQT contains forbidden tag: %s",
                    tag,
                )
                return False

        lines = txt.splitlines()

        atom_lines = [
            ln
            for ln in lines
            if ln.startswith("ATOM") or ln.startswith("HETATM")
        ]

        if not atom_lines:
            logger.error("[VINA] receptor.pdbqt contains no ATOM/HETATM records")
            return False

        # Basic AutoDock atom typing heuristic
        typed = 0

        for ln in atom_lines:
            toks = ln.split()

            if toks and toks[-1].isalpha():
                typed += 1

        if typed == 0:
            logger.error("[VINA] receptor.pdbqt lacks AutoDock atom typing")
            return False

        return True

    # ============================================================
    # Cleanup invalid ligand-style tags from receptor
    # ============================================================

    def _cleanup_openbabel_receptor_pdbqt(self, path: Path) -> None:

        txt = path.read_text(encoding="utf-8", errors="ignore")

        cleaned = []

        forbidden_prefixes = (
            "ROOT",
            "BRANCH",
            "ENDBRANCH",
            "TORSDOF",
        )

        for line in txt.splitlines():

            stripped = line.strip()

            if any(stripped.startswith(p) for p in forbidden_prefixes):
                continue

            cleaned.append(line)

        path.write_text(
            "\n".join(cleaned) + "\n",
            encoding="utf-8",
        )

    # ============================================================
    # Main preparation
    # ============================================================

    def prepare_receptor_pdbqt(
        self,
        *,
        pdb_string: str,
        protein_key: str,
        output_path: Path,
    ) -> None:

        if output_path.exists() and output_path.stat().st_size > 0:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with TempDir(base_dir=self.cfg.tmp_dir) as tmpdir:

            receptor_pdb = tmpdir / f"{protein_key}.pdb"

            receptor_pdb.write_text(
                pdb_string,
                encoding="utf-8",
            )

            # ----------------------------------------------------
            # Preserve raw receptor source
            # ----------------------------------------------------

            output_path.with_suffix(".source.pdb").write_text(
                pdb_string,
                encoding="utf-8",
            )

            pdbqt_ok = False
            last_reason: Optional[str] = None

            # ====================================================
            # PRIMARY: MGLTools
            # ====================================================

            if is_mgltools_available():

                script = (
                    self.cfg.mgltools_prepare_receptor4
                    or shutil.which("prepare_receptor4.py")
                )

                if script:

                    cmd = [
                        sys.executable,
                        script,
                        "-r",
                        str(receptor_pdb),
                        "-o",
                        str(output_path),
                        "-A",
                        "hydrogens",
                    ]

                    logger.info(
                        "[VINA] Running MGLTools receptor prep: %s",
                        " ".join(cmd),
                    )

                    proc = run_subprocess(
                        cmd,
                        timeout_sec=self.cfg.docking_timeout,
                        cwd=tmpdir,
                    )

                    logger.info(
                        "[VINA] receptor prep stdout=\n%s",
                        proc.stdout,
                    )

                    logger.info(
                        "[VINA] receptor prep stderr=\n%s",
                        proc.stderr,
                    )

                    if (
                        proc.returncode == 0
                        and output_path.exists()
                        and output_path.stat().st_size > 0
                    ):

                        if self._validate_receptor_pdbqt(output_path):

                            pdbqt_ok = True

                            logger.info(
                                "[VINA] Receptor prepared successfully via MGLTools"
                            )

                        else:
                            last_reason = (
                                "MGLTools produced invalid receptor PDBQT"
                            )

                    else:
                        last_reason = (
                            (proc.stdout or "")
                            + "\n"
                            + (proc.stderr or "")
                        )

            # ====================================================
            # FALLBACK: OpenBabel
            # ====================================================

            if not pdbqt_ok and is_openbabel_available():

                cmd = [
                    str(resolve_openbabel_executable()),
                    str(receptor_pdb),
                    "-xr",          # rigid receptor mode
                    "-p",           # preserve hydrogens
                    "-O",
                    str(output_path),
                ]

                logger.warning(
                    "[VINA] Falling back to OpenBabel receptor prep"
                )

                logger.info(
                    "[VINA] OpenBabel command: %s",
                    " ".join(cmd),
                )

                proc = run_subprocess(
                    cmd,
                    timeout_sec=self.cfg.docking_timeout,
                    cwd=tmpdir,
                )

                logger.info(
                    "[VINA] OpenBabel stdout=\n%s",
                    proc.stdout,
                )

                logger.info(
                    "[VINA] OpenBabel stderr=\n%s",
                    proc.stderr,
                )

                if (
                    proc.returncode == 0
                    and output_path.exists()
                    and output_path.stat().st_size > 0
                ):

                    # --------------------------------------------
                    # CRITICAL FIX:
                    # remove ligand torsion tags
                    # --------------------------------------------

                    self._cleanup_openbabel_receptor_pdbqt(
                        output_path
                    )

                    if self._validate_receptor_pdbqt(output_path):

                        pdbqt_ok = True

                        logger.info(
                            "[VINA] Receptor prepared successfully via OpenBabel"
                        )

                    else:
                        last_reason = (
                            "OpenBabel produced invalid receptor PDBQT"
                        )

                else:
                    last_reason = (
                        (proc.stdout or "")
                        + "\n"
                        + (proc.stderr or "")
                    )

            # ====================================================
            # FINAL FAILURE
            # ====================================================

            if not pdbqt_ok:

                logger.error(
                    "[VINA] Failed receptor preparation. reason=%s",
                    last_reason,
                )

                raise ProteinPrepError(
                    f"Failed to produce valid receptor PDBQT. "
                    f"reason={last_reason}"
                )

            logger.info(
                "[VINA] Final receptor PDBQT ready: %s",
                output_path,
            )