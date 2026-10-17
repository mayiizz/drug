from __future__ import annotations
 
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
 
 
def run_subprocess(
    cmd: list[str],
    timeout_sec: int,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with stdout/stderr capture and a timeout.
    Does NOT raise on non-zero return codes.
    """
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
 
 
class TempDir:
    """Context manager for temp directories with safe cleanup."""
 
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir
        self.path: Optional[Path] = None
 
    def __enter__(self) -> Path:
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(prefix="cura_vina_", dir=str(self.base_dir) if self.base_dir else None)
        )
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path
 
    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
 
 
# Vina stdout result table row.
# Actual Vina output looks like:
#    1         -7.2      0.000      0.000
# columns: mode | affinity (kcal/mol) | dist from best mode | rmsd l.b.
# The affinity is the SECOND column (index 1), always negative for good binders.
#
# We match lines that start with an integer rank followed by a negative float.
# This is intentionally broad to handle slight formatting variations across
# Vina versions (1.1.2, 1.2.x, gnina, smina).
_VINA_RESULT_ROW_RE = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<aff>-\d+\.\d+)\s+",
    re.MULTILINE,
)
 
# Fallback: match "affinity: -7.2" style summary lines some Vina builds emit.
_VINA_AFFINITY_LABEL_RE = re.compile(
    r"affinity\s*[:\s]+(?P<aff>-\d+\.\d+)",
    re.IGNORECASE,
)
 
 
def parse_vina_stdout_best_affinity(stdout: str) -> Optional[float]:
    """
    Parse Vina stdout for the best (rank 1) affinity (kcal/mol).
 
    Vina always outputs negative affinities for valid poses.
    Returns None if no valid affinity found.
    """
    if not stdout:
        return None
 
    # Primary: find rank-1 row in the result table.
    best: Optional[float] = None
    for m in _VINA_RESULT_ROW_RE.finditer(stdout):
        try:
            rank = int(m.group("rank"))
            aff = float(m.group("aff"))
            if rank == 1:
                return aff
            # Track the most negative (best) affinity seen as a fallback
            # in case rank column is missing/malformed.
            if best is None or aff < best:
                best = aff
        except Exception:
            continue
 
    if best is not None:
        return best
 
    # Secondary fallback: "affinity: -7.2" label anywhere in output.
    m2 = _VINA_AFFINITY_LABEL_RE.search(stdout)
    if m2:
        try:
            return float(m2.group("aff"))
        except Exception:
            pass
 
    return None
 
 
def read_text_if_exists(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return None
 
 
def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
 
 
def read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
