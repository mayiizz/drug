from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen, Lipinski

try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
except Exception:  # pragma: no cover
    FilterCatalog = None  # type: ignore[assignment]
    FilterCatalogParams = None  # type: ignore[assignment]

from models.generation.safe_gpt.config import FilteringConfig


def has_reactive_group(mol: Chem.Mol) -> bool:
    # Medicinally problematic reactive groups.
    smarts_list = [
        "[NX2]=[O]",  # nitroso
        "[N]=[N]",  # azo
        "[NX3]-[N]=[N]",  # triazene
        "[CX3H1](=O)[#6]",  # aldehyde
        "C=CC(=O)[#6]",  # michael acceptor (enone-like)
        "C(=O)Cl",  # acid chloride
        "[S](=O)(=O)Cl",  # sulfonyl chloride
    ]
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return True
    return False


_STRUCTURAL_ALERT_CATALOG = None


def _build_structural_alert_catalog():
    """Lazy-build a combined PAINS A/B/C + Brenk filter catalog.

    Cached at module level so the SMARTS sets are compiled once. Returns
    ``None`` when ``rdkit.Chem.FilterCatalog`` is unavailable, in which case
    callers fall back to the lightweight ``has_reactive_group`` heuristic.
    """
    global _STRUCTURAL_ALERT_CATALOG
    if _STRUCTURAL_ALERT_CATALOG is not None:
        return _STRUCTURAL_ALERT_CATALOG
    if FilterCatalog is None or FilterCatalogParams is None:
        return None
    params = FilterCatalogParams()
    for name in ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK"):
        cat = getattr(FilterCatalogParams.FilterCatalogs, name, None)
        if cat is not None:
            try:
                params.AddCatalog(cat)
            except Exception:
                continue
    _STRUCTURAL_ALERT_CATALOG = FilterCatalog(params)
    return _STRUCTURAL_ALERT_CATALOG


def has_structural_alert(mol: Chem.Mol) -> bool:
    """True if ``mol`` matches any PAINS A/B/C or Brenk SMARTS alert.

    Falls back to ``False`` when RDKit's FilterCatalog is unavailable so
    we never silently drop a whole batch on a missing optional dep; the
    lightweight ``has_reactive_group`` covers the most dangerous motifs.
    """
    catalog = _build_structural_alert_catalog()
    if catalog is None:
        return False
    try:
        return bool(catalog.HasMatch(mol))
    except Exception:
        return False


def has_pains_alert(mol: Chem.Mol) -> bool:
    """Backward-compatible alias - now PAINS+Brenk."""
    return has_structural_alert(mol)


def _is_single_fragment(mol: Chem.Mol) -> bool:
    try:
        return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) == 1
    except Exception:
        return True


def passes_basic_filters(smiles: str, cfg: FilteringConfig) -> bool:
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if not _is_single_fragment(mol):
        return False
    if mol.GetNumHeavyAtoms() < 2:
        return False
    if cfg.reject_formal_charge:
        if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
            return False
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < cfg.min_ring_count or ring_count > cfg.max_ring_count:
        return False
    if cfg.require_aromatic_if_multiring and ring_count > 1:
        aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        if aromatic == 0:
            return False
    if cfg.reactive_group_hook_enabled and has_reactive_group(mol):
        return False
    if cfg.pains_hook_enabled and has_structural_alert(mol):
        return False
    logp = Crippen.MolLogP(mol)  # type: ignore[attr-defined]
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    if logp > 5.8:
        return False
    if arom >= 5 and logp > 4.8:
        return False
    if rot > 11:
        return False
    return True

