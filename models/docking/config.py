from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_VINA_EXECUTABLE = Path(r"D:\tools\vina\vina.exe")
VINA_EXECUTABLE = Path(
    os.getenv(
        "CURA_VINA_EXECUTABLE",
        str(DEFAULT_VINA_EXECUTABLE),
    )
)
OPENBABEL_EXECUTABLE = Path(
    os.getenv(
        "CURA_OPENBABEL_EXECUTABLE",
        "obabel",
    )
)


def resolve_vina_executable() -> Path:
    p = Path(VINA_EXECUTABLE).expanduser()
    if os.name == "nt" and (not p.exists()) and p.suffix.lower() != ".exe":
     alt = p.with_suffix(".exe")
     if alt.exists():
        p = alt
    if not p.exists() or not p.is_file():
        raise RuntimeError(
            f"[VINA] Executable missing. expected={p}. "
            "Set CURA_VINA_EXECUTABLE to a valid vina.exe path."
        )
    if os.name == "nt" and p.suffix.lower() != ".exe":
        raise RuntimeError(
            f"[VINA] Invalid executable for Windows. expected .exe path, got={p}"
        )
    return p.resolve()


def resolve_openbabel_executable() -> Path:
    candidate = Path(OPENBABEL_EXECUTABLE).expanduser()
    if candidate.exists():
        exe = candidate.resolve()
    else:
        which_match = shutil.which(str(candidate))
        if not which_match:
            raise RuntimeError(
                f"[OpenBabel] Executable missing. expected={candidate}. "
                "Set CURA_OPENBABEL_EXECUTABLE to a valid obabel executable."
            )
        exe = Path(which_match).resolve()

    if os.name == "nt" and exe.suffix and exe.suffix.lower() != ".exe":
        raise RuntimeError(f"[OpenBabel] Invalid Windows executable path: {exe}")
    try:
        proc = subprocess.run(
            [str(exe), "-V"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(f"[OpenBabel] Version check failed for {exe}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"[OpenBabel] Version check failed for {exe}. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return exe


def is_vina_available(vina_executable: str | None = None) -> bool:
    """Deterministic check for AutoDock Vina executable."""
    try:
        if vina_executable:
            p = Path(vina_executable).expanduser()
            return p.exists() and p.is_file()
        resolve_vina_executable()
        return True
    except Exception:
        return False


def is_mgltools_available() -> bool:
    """
    Best-effort check for AutoDockTools/MGLTools prep scripts.

    We look for common script names in PATH or typical environment variables.
    """
    # These script names are the typical MGLTools/ADT scripts used by vina workflows.
    candidates = [
        "prepare_receptor4.py",
        "prepare_ligand4.py",
    ]

    for c in candidates:
        if shutil.which(c) is not None:
            return True

    mgltools_home = os.environ.get("MGLTOOLS_HOME") or os.environ.get("AUTODOCKTOOLS_HOME")
    if mgltools_home:
        home = Path(mgltools_home)
        # Common location: .../bin
        for sub in ("bin", "scripts", ""):
            if (home / sub / "prepare_receptor4.py").exists() and (home / sub / "prepare_ligand4.py").exists():
                return True
    return False


def is_openbabel_available() -> bool:
    """Deterministic check for OpenBabel executable and version."""
    try:
        resolve_openbabel_executable()
        return True
    except Exception:
        return False


def _default_cache_dir() -> Path:
    # Prefer an on-disk cache to avoid redoing expensive prep.
    return Path(os.environ.get("CURA_DOCKING_CACHE", "result/docking_cache")).resolve()


def _default_tmp_dir() -> Path:
    return Path(os.environ.get("CURA_DOCKING_TMP", "result/docking_tmp")).resolve()

def _default_prefilter_cache_dir() -> Path:
    # Share the same root by default, but keep prefilter artifacts separate.
    return Path(os.environ.get("CURA_PREFILTER_CACHE", "result/docking_cache/prefilter")).resolve()


@dataclass(slots=True)
class VinaConfig:
    vina_executable: str = field(default_factory=lambda: str(VINA_EXECUTABLE))
    openbabel_executable: str = field(default_factory=lambda: str(OPENBABEL_EXECUTABLE))
    exhaustiveness: int = field(default_factory=lambda: int(os.environ.get("VINA_EXHAUSTIVENESS", "8")))
    num_modes: int = field(default_factory=lambda: int(os.environ.get("VINA_NUM_MODES", "1")))
    energy_range: float = field(default_factory=lambda: float(os.environ.get("VINA_ENERGY_RANGE", "3.0")))
    docking_timeout: int = field(default_factory=lambda: int(os.environ.get("VINA_TIMEOUT_SEC", "60")))
    cpu_threads: int = field(default_factory=lambda: int(os.environ.get("VINA_THREADS", str(os.cpu_count() or 1))))

    # Shortlist size for top-k docking refinement
    top_k_docking: int = field(default_factory=lambda: int(os.environ.get("VINA_TOP_K_DOCKING", "20")))

    # ── DeepPurpose prefilter controls (shortlisting only) ──────────────────
    prefilter_enabled: bool = field(default_factory=lambda: os.environ.get("CURA_PREFILTER_ENABLED", "1").strip() not in ("0", "false", "False"))
    prefilter_top_k: int = field(default_factory=lambda: int(os.environ.get("CURA_PREFILTER_TOP_K", "50")))
    # Accept either 0.10 or 10 (meaning 10%) from env.
    prefilter_top_percent: float | None = field(
        default_factory=lambda: (
            None
            if os.environ.get("CURA_PREFILTER_TOP_PERCENT", "").strip() == ""
            else float(os.environ.get("CURA_PREFILTER_TOP_PERCENT", "0.1"))
        )
    )
    # Threshold in DeepPurpose *raw score* space (log10(Kd)-like); optional.
    # Lower raw score = stronger binding surrogate.
    prefilter_affinity_threshold: float | None = field(
        default_factory=lambda: (
            None
            if os.environ.get("CURA_PREFILTER_THRESHOLD", "").strip() == ""
            else float(os.environ.get("CURA_PREFILTER_THRESHOLD", "0"))
        )
    )

    prefilter_cache_enabled: bool = field(default_factory=lambda: os.environ.get("CURA_PREFILTER_CACHE_ENABLED", "0").strip() in ("1", "true", "True"))
    prefilter_cache_dir: Path = field(default_factory=_default_prefilter_cache_dir)

    # If prefilter is unavailable, optionally dock a small random subset instead of everything.
    dock_random_subset_when_no_prefilter: int = field(default_factory=lambda: int(os.environ.get("CURA_DOCK_RANDOM_WHEN_NO_PREFILTER", "20")))

    # Ablation switches
    mode: str = field(default_factory=lambda: os.environ.get("CURA_BINDING_MODE", "hybrid").strip().lower())

    # Auto-enable only when tools are present (no hard dependency).
    docking_enabled: bool = True
    strict_benchmark_mode: bool = field(
        default_factory=lambda: os.environ.get("CURA_STRICT_BENCHMARK_MODE", "0").strip() in ("1", "true", "True")
    )

    # Docking box size (side length, Angstroms) used when pocket extraction gives only a center.
    # Default 22A balances kinase ATP-pocket coverage with search efficiency.
    # Override at runtime via the ``VINA_BOX_SIZE`` environment variable.
    box_size: float = field(default_factory=lambda: float(os.environ.get("VINA_BOX_SIZE", "22.0")))

    # Prep scripts (optional overrides)
    mgltools_prepare_receptor4: Optional[str] = field(default_factory=lambda: os.environ.get("MGLTOOLS_PREP_RECEPTOR4"))
    mgltools_prepare_ligand4: Optional[str] = field(default_factory=lambda: os.environ.get("MGLTOOLS_PREP_LIGAND4"))

    # Caching/temp controls
    cache_dir: Path = field(default_factory=_default_cache_dir)
    tmp_dir: Path = field(default_factory=_default_tmp_dir)

    # Conformer filtering thresholds (RDKit side)
    max_mmff_energy_per_atom: float = field(default_factory=lambda: float(os.environ.get("CURA_MMFF_MAX_E_PER_ATOM", "8.0")))
    mmff_max_iters: int = field(default_factory=lambda: int(os.environ.get("CURA_MMFF_MAX_ITERS", "500")))
    max_mmff_total_energy: float = field(default_factory=lambda: float(os.environ.get("CURA_MMFF_MAX_TOTAL_E", "800.0")))
    max_embed_attempts: int = field(default_factory=lambda: int(os.environ.get("CURA_EMBED_MAX_ATTEMPTS", "5")))
    conformer_log_failures: bool = field(default_factory=lambda: os.environ.get("CURA_CONFORMER_LOG_FAILURES", "1").strip() not in ("0", "false", "False"))

    def effective_docking_enabled(self) -> bool:
        """Enable only if caller requested it and tools are detected."""
        if self.mode == "deeppurpose_only":
            return False
        if not self.docking_enabled:
            return False

        # We consider docking "available" if vina is present AND at least one conversion tool exists.
        # (MGLTools is preferred, but OpenBabel provides a fallback.)
        try:
            _ = resolve_vina_executable()
        except Exception:
            return False

        # If neither prep backend exists, we can't reliably create PDBQT.
        if not is_mgltools_available() and not is_openbabel_available():
            return False

        return True

    def __post_init__(self) -> None:
        # Ensure filesystem roots exist before any docking/prep subprocess cwd usage.
        self.cache_dir = Path(self.cache_dir)
        self.tmp_dir = Path(self.tmp_dir)
        self.prefilter_cache_dir = Path(self.prefilter_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.prefilter_cache_dir.mkdir(parents=True, exist_ok=True)

    def effective_prefilter_enabled(self) -> bool:
        """
        Prefilter can be used even when docking is unavailable.
        It is disabled for vina-only ablations.
        """
        if self.mode == "vina_only":
            return False
        return bool(self.prefilter_enabled)

    def resolved_prefilter_top_percent(self) -> float | None:
        """
        Normalise prefilter_top_percent to [0,1] if set.
        Accepts values like 10 (meaning 10%) or 0.10.
        """
        if self.prefilter_top_percent is None:
            return None
        p = float(self.prefilter_top_percent)
        if p > 1.0:
            p = p / 100.0
        return max(0.0, min(1.0, p))

