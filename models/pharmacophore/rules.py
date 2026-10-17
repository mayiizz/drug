from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Lipinski


def aromatic_ring_count(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    n = 0
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n += 1
    return n


def hetero_aromatic_ring_count(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    n = 0
    for ring in rings:
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16) for i in ring):
            n += 1
    return n


def aromatic_n_count(mol: Chem.Mol) -> int:
    return sum(
        1
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetIsAromatic()
    )


def hba_count(mol: Chem.Mol) -> int:
    return int(Lipinski.NumHAcceptors(mol))


def hbd_count(mol: Chem.Mol) -> int:
    return int(Lipinski.NumHDonors(mol))


def rotatable_bonds(mol: Chem.Mol) -> int:
    return int(Lipinski.NumRotatableBonds(mol))


def has_substructure(mol: Chem.Mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return False
    return bool(mol.HasSubstructMatch(patt))

