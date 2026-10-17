import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

# Generators are created once (efficient)
_morgan_gen_2048 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)
_morgan_gen_1024 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024,
)


def get_morgan_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_gen_2048.GetFingerprint(mol)


def mols_to_morgan_fingerprint_matrix(mols: list, *, fp_size: int = 1024) -> np.ndarray:
    """
    RDKit Morgan count simulation → fixed-length bit matrix for sklearn / RF.

    fp_size 1024 or 2048 must match the generator used to build training data.
    Invalid / None mols become zero rows.
    """
    if fp_size == 1024:
        gen = _morgan_gen_1024
    elif fp_size == 2048:
        gen = _morgan_gen_2048
    else:
        raise ValueError(f"fp_size must be 1024 or 2048, got {fp_size}")

    n = len(mols)
    X = np.zeros((n, fp_size), dtype=np.float32)
    buf = np.zeros((fp_size,), dtype=np.int8)
    for i, m in enumerate(mols):
        if m is None:
            continue
        fp = gen.GetFingerprint(m)
        ConvertToNumpyArray(fp, buf)
        X[i] = buf.astype(np.float32)
    return X
