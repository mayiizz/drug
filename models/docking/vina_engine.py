from __future__ import annotations

import re
import logging
import subprocess
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.docking.config import VinaConfig, resolve_vina_executable
from models.docking.utils import parse_vina_stdout_best_affinity, run_subprocess

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DockingResult:
    affinity_kcal_mol: Optional[float]
    pose_path: Optional[str]
    success: bool
    raw_output: str


class VinaEngine:
    """
    AutoDock Vina execution wrapper.

    This wrapper is designed to fail gracefully:
      - missing vina executable
      - timeouts
      - parse failures
    """

    def __init__(self, cfg: VinaConfig):
        self.cfg = cfg
        self._vina_executable = resolve_vina_executable()
        logger.info("[VINA] Executable resolved: %s", self._vina_executable)
        self._checked_version = False

    def _ensure_vina_runtime_ready(self) -> None:
        if self._checked_version:
            return
        try:
            proc = subprocess.run(
                [str(self._vina_executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception as exc:
            raise RuntimeError(f"[VINA] Version check failed: {exc}") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"[VINA] Version check failed for {self._vina_executable}. "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        self._checked_version = True

    def _write_vina_config(
        self,
        config_path: Path,
        *,
        receptor_path: Path,
        ligand_path: Path,
        pose_path: Path,
        cx: float,
        cy: float,
        cz: float,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        lines = [
            f"receptor = {receptor_path}",
            f"ligand = {ligand_path}",
            f"center_x = {cx}",
            f"center_y = {cy}",
            f"center_z = {cz}",
            f"size_x = {sx}",
            f"size_y = {sy}",
            f"size_z = {sz}",
            f"exhaustiveness = {self.cfg.exhaustiveness}",
            f"num_modes = {self.cfg.num_modes}",
            f"energy_range = {self.cfg.energy_range}",
            f"cpu = {self.cfg.cpu_threads}",
            f"out = {pose_path}",
        ]
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _persist_failed_run(
        self,
        *,
        working_dir: Path,
        cmd: list[str],
        receptor_path: Path,
        ligand_path: Path,
        stdout: str,
        stderr: str,
        return_code: int | None,
    ) -> None:
        debug_root = Path("result/debug_failed_docking").resolve()
        debug_root.mkdir(parents=True, exist_ok=True)
        tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        dest = debug_root / tag
        shutil.copytree(working_dir, dest, dirs_exist_ok=True)
        (dest / "vina_stdout.log").write_text(stdout or "", encoding="utf-8", errors="ignore")
        (dest / "vina_error.log").write_text(stderr or "", encoding="utf-8", errors="ignore")
        (dest / "failure_meta.txt").write_text(
            "\n".join(
                [
                    f"return_code={return_code}",
                    f"command={' '.join(cmd)}",
                    f"receptor={receptor_path}",
                    f"ligand={ligand_path}",
                    f"working_dir={working_dir}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        logger.warning("[VINA] Failed docking artifacts saved to %s", dest)

    def _log_failure_context(
        self,
        *,
        cmd: list[str],
        receptor: Path,
        ligand: Path,
        config_path: Path,
        cwd: Path,
        stdout: str,
        stderr: str,
        return_code: int | None,
    ) -> None:
        logger.warning("[VINA] Docking failed")
        logger.warning("[VINA] return_code=%s", return_code)
        logger.warning("[VINA] command=%s", " ".join(cmd) if cmd else "<empty>")
        logger.warning("[VINA] receptor=%s", receptor)
        logger.warning("[VINA] ligand=%s", ligand)
        logger.warning("[VINA] config=%s", config_path)
        logger.warning("[VINA] cwd=%s", cwd)
        logger.warning("[VINA] stdout=\n%s", stdout or "")
        logger.warning("[VINA] stderr=\n%s", stderr or "")

    def dock(
        self,
        receptor_pdbqt: Path,
        ligand_pdbqt: Path,
        box,
        pose_output_path: Optional[Path] = None,
    ) -> DockingResult:
        self._ensure_vina_runtime_ready()
        Path(self.cfg.tmp_dir).mkdir(parents=True, exist_ok=True)
        working_dir = Path(self.cfg.tmp_dir) / f"cura_vina_run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[VINA] Docking started receptor=%s ligand=%s",
            receptor_pdbqt,
            ligand_pdbqt,
        )
        if not receptor_pdbqt.exists() or not ligand_pdbqt.exists():
            msg = (
                f"Missing receptor or ligand pdbqt. receptor={receptor_pdbqt}, "
                f"ligand={ligand_pdbqt}"
            )
            self._log_failure_context(
                cmd=[],
                receptor=receptor_pdbqt,
                ligand=ligand_pdbqt,
                config_path=working_dir / "vina_config.txt",
                cwd=working_dir,
                stdout="",
                stderr=msg,
                return_code=None,
            )
            return DockingResult(
                affinity_kcal_mol=None,
                pose_path=None,
                success=False,
                raw_output=msg,
            )

        local_receptor = working_dir / "receptor.pdbqt"
        local_ligand = working_dir / "ligand.pdbqt"
        shutil.copy2(receptor_pdbqt, local_receptor)
        shutil.copy2(ligand_pdbqt, local_ligand)
        receptor_source_pdb = receptor_pdbqt.with_suffix(".source.pdb")
        if receptor_source_pdb.exists():
            shutil.copy2(receptor_source_pdb, working_dir / "receptor_raw.pdb")

        pose_output_path = pose_output_path or (Path(self.cfg.tmp_dir) / "vina_out_pose.pdbqt")
        pose_output_path.parent.mkdir(parents=True, exist_ok=True)
        local_pose_output_path = working_dir / "vina_out_pose.pdbqt"
        if pose_output_path.exists():
            try:
                pose_output_path.unlink()
            except Exception:
                pass

        cx, cy, cz = box.center
        sx, sy, sz = box.size
        logger.info("[VINA] Receptor: %s", receptor_pdbqt)
        logger.info("[VINA] Ligand: %s", ligand_pdbqt)
        logger.info("[VINA] Working dir: %s", working_dir)
        logger.info("[VINA] Grid center: (%.4f, %.4f, %.4f)", float(cx), float(cy), float(cz))
        logger.info("[VINA] Grid size: (%.4f, %.4f, %.4f)", float(sx), float(sy), float(sz))
        logger.info(
            "[VINA] Config center_x=%.4f center_y=%.4f center_z=%.4f size_x=%.4f size_y=%.4f size_z=%.4f exhaustiveness=%s",
            float(cx),
            float(cy),
            float(cz),
            float(sx),
            float(sy),
            float(sz),
            self.cfg.exhaustiveness,
        )

        cmd = [
            str(self._vina_executable),
            "--receptor",
            str(local_receptor),
            "--ligand",
            str(local_ligand),
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(sx),
            "--size_y",
            str(sy),
            "--size_z",
            str(sz),
            "--exhaustiveness",
            str(self.cfg.exhaustiveness),
            "--num_modes",
            str(self.cfg.num_modes),
            "--energy_range",
            str(self.cfg.energy_range),
            "--cpu",
            str(self.cfg.cpu_threads),
            "--out",
            str(local_pose_output_path),
        ]
        self._write_vina_config(
            working_dir / "vina_config.txt",
            receptor_path=local_receptor,
            ligand_path=local_ligand,
            pose_path=local_pose_output_path,
            cx=float(cx),
            cy=float(cy),
            cz=float(cz),
            sx=float(sx),
            sy=float(sy),
            sz=float(sz),
        )
        vina_config_path = working_dir / "vina_config.txt"
        logger.info("[VINA] command=%s", " ".join(cmd))

        try:
            proc = run_subprocess(cmd, timeout_sec=self.cfg.docking_timeout, cwd=working_dir)
        except Exception as e:  # pragma: no cover
            self._log_failure_context(
                cmd=cmd,
                receptor=local_receptor,
                ligand=local_ligand,
                config_path=vina_config_path,
                cwd=working_dir,
                stdout="",
                stderr=str(e),
                return_code=None,
            )
            self._persist_failed_run(
                working_dir=working_dir,
                cmd=cmd,
                receptor_path=local_receptor,
                ligand_path=local_ligand,
                stdout="",
                stderr=str(e),
                return_code=None,
            )
            return DockingResult(
                affinity_kcal_mol=None,
                pose_path=None,
                success=False,
                raw_output=f"Failed to execute vina: {e}",
            )

        raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
        logger.warning("[VINA RAW OUTPUT]\n%s", raw)
        (working_dir / "vina_stdout.log").write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
        (working_dir / "vina_stderr.log").write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            self._log_failure_context(
                cmd=cmd,
                receptor=local_receptor,
                ligand=local_ligand,
                config_path=vina_config_path,
                cwd=working_dir,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                return_code=proc.returncode,
            )
            self._persist_failed_run(
                working_dir=working_dir,
                cmd=cmd,
                receptor_path=local_receptor,
                ligand_path=local_ligand,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                return_code=proc.returncode,
            )
            return DockingResult(
                affinity_kcal_mol=None,
                pose_path=str(local_pose_output_path) if local_pose_output_path.exists() else None,
                success=False,
                raw_output=raw,
            )

        affinity = parse_vina_stdout_best_affinity(raw)
        if affinity is None:
            # As a fallback, try to locate the first “-x.x” after “affinity” text.
            m = re.search(r"affinity\s*[:=]?\s*(-?\d+\.\d+)", raw, re.IGNORECASE)
            affinity = float(m.group(1)) if m else None

        success = affinity is not None and local_pose_output_path.exists()
        logger.info("[VINA] Ligand prepared ligand=%s", ligand_pdbqt)
        if success and affinity is not None:
            shutil.copy2(local_pose_output_path, pose_output_path)
            logger.info("[VINA] Docking completed pose=%s", pose_output_path)
            logger.info("[VINA] Affinity=%.4f kcal/mol", affinity)
        else:
            logger.warning("[VINA] Docking failed: affinity_parse_or_pose_missing")
            self._log_failure_context(
                cmd=cmd,
                receptor=local_receptor,
                ligand=local_ligand,
                config_path=vina_config_path,
                cwd=working_dir,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                return_code=proc.returncode,
            )
            self._persist_failed_run(
                working_dir=working_dir,
                cmd=cmd,
                receptor_path=local_receptor,
                ligand_path=local_ligand,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                return_code=proc.returncode,
            )
        return DockingResult(
            affinity_kcal_mol=affinity,
            pose_path=str(pose_output_path) if pose_output_path.exists() else None,
            success=bool(success),
            raw_output=raw,
        )

