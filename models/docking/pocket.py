from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from models.docking.config import VinaConfig

logger = logging.getLogger(__name__)
MAX_RL_BOX_SIDE = 24.0


@dataclass(frozen=True, slots=True)
class DockingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def _radius_from_volume_est(volume_est: float) -> float:
    # volume = 4/3*pi*r^3  => r = (3*volume/(4*pi))^(1/3)
    return float((3.0 * volume_est / (4.0 * math.pi)) ** (1.0 / 3.0))


def box_from_pocket(pocket: Dict[str, Any], cfg: VinaConfig) -> DockingBox:
    """
    Convert PocketExtractor output → docking box.

    PocketExtractor returns:
      - "centre": (x, y, z)
      - optional "volume_est"
    """
    center = pocket.get("centre")
    if not center or len(center) != 3:
        raise ValueError("Pocket has no valid 'centre'.")

    volume_est = pocket.get("volume_est")
    if isinstance(volume_est, (int, float)) and volume_est > 0:
        r = _radius_from_volume_est(float(volume_est))
        # Expand slightly to ensure the full binding region is covered.
        side = min(MAX_RL_BOX_SIDE, max(cfg.box_size, 2.0 * r * 1.5))
    else:
        side = cfg.box_size

    size = (float(side), float(side), float(side))
    return DockingBox(center=(float(center[0]), float(center[1]), float(center[2])), size=size)


def _ca_centroid_for_residues(
    pdb_string: str,
    residues: Iterable[Any],
) -> Optional[Tuple[float, float, float]]:
    """Return mean CA coordinate for the supplied residue identifiers.

    ``residues`` is loose-typed because PocketExtractor may emit any of:
        - integer residue numbers (4)
        - ``"A:120"`` chain:resnum strings
        - dicts ``{"chain": "A", "resnum": 120}``
        - ``(chain, resnum)`` tuples
    Anything we cannot parse is silently skipped.
    """
    targets: set[tuple[str | None, int]] = set()
    for r in residues or ():
        if r is None:
            continue
        chain: str | None = None
        resnum: int | None = None
        if isinstance(r, int):
            resnum = r
        elif isinstance(r, str):
            s = r.strip()
            if ":" in s:
                head, tail = s.split(":", 1)
                chain = head.strip() or None
                try:
                    resnum = int(tail.strip())
                except ValueError:
                    continue
            else:
                try:
                    resnum = int(s)
                except ValueError:
                    continue
        elif isinstance(r, dict):
            chain = r.get("chain") or None
            try:
                resnum = int(r.get("resnum") or r.get("resi") or r.get("id"))
            except Exception:
                continue
        elif isinstance(r, (tuple, list)) and len(r) >= 2:
            chain = str(r[0]) if r[0] else None
            try:
                resnum = int(r[1])
            except Exception:
                continue
        if resnum is None:
            continue
        targets.add((chain, resnum))

    if not targets:
        return None

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if " CA " not in line[12:16]:
            continue
        try:
            chain_id = line[21:22].strip() or None
            resnum = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except Exception:
            continue
        if (chain_id, resnum) in targets or (None, resnum) in targets:
            xs.append(x)
            ys.append(y)
            zs.append(z)

    if not xs:
        return None
    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )


def box_from_kinase_pocket(
    pocket: Dict[str, Any],
    pdb_string: Optional[str],
    cfg: VinaConfig,
) -> DockingBox:
    """Build a docking box centered on kinase ATP-pocket landmarks when known.

    Prefers a centroid computed from the union of hinge / DFG / catalytic
    residue CA coordinates in ``pdb_string`` when the pocket dict provides
    them. Falls back to ``box_from_pocket`` otherwise so the helper is safe
    to call unconditionally from ``CuraPipeline._dock``.
    """
    if pocket is None:
        raise ValueError("kinase pocket dict is required")

    landmark_keys = ("hinge_residues", "dfg_residues", "catalytic_residues")
    landmark_residues: list[Any] = []
    for key in landmark_keys:
        v = pocket.get(key)
        if not v:
            continue
        if isinstance(v, (list, tuple, set)):
            landmark_residues.extend(v)
        else:
            landmark_residues.append(v)

    if pdb_string and landmark_residues:
        centroid = _ca_centroid_for_residues(pdb_string, landmark_residues)
        if centroid is not None:
            volume_est = pocket.get("volume_est")
            if isinstance(volume_est, (int, float)) and volume_est > 0:
                r = _radius_from_volume_est(float(volume_est))
                side = min(MAX_RL_BOX_SIDE, max(cfg.box_size, 2.0 * r * 1.5))
            else:
                side = cfg.box_size
            logger.info(
                "[VINA] kinase-aware box center=(%.2f,%.2f,%.2f) side=%.1fA "
                "(%d hinge/DFG/cat landmarks)",
                centroid[0],
                centroid[1],
                centroid[2],
                side,
                len(landmark_residues),
            )
            return DockingBox(
                center=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                size=(float(side), float(side), float(side)),
            )

    return box_from_pocket(pocket, cfg)


def box_from_pdb_whole_protein(pdb_string: str, cfg: VinaConfig) -> DockingBox:
    """
    Approximate a whole-protein docking box by bounding CA coordinates.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        # CA atom lines contain " CA " in the atom name column.
        if len(line) < 54:
            continue
        if " CA " not in line[12:16]:
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except Exception:
            continue
        xs.append(x)
        ys.append(y)
        zs.append(z)

    if not xs:
        # Ultimate fallback: a fixed box around origin.
        return DockingBox(center=(0.0, 0.0, 0.0), size=(cfg.box_size, cfg.box_size, cfg.box_size))

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0

    margin = 6.0  # Angstroms; gives vina room
    sx = min(MAX_RL_BOX_SIDE, max(cfg.box_size, (xmax - xmin) + margin))
    sy = min(MAX_RL_BOX_SIDE, max(cfg.box_size, (ymax - ymin) + margin))
    sz = min(MAX_RL_BOX_SIDE, max(cfg.box_size, (zmax - zmin) + margin))

    return DockingBox(center=(cx, cy, cz), size=(sx, sy, sz))

