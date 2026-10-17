from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem


class MoleculeVisualizer:
    @staticmethod
    def validate_smiles(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return mol

    @staticmethod
    def generate_2d_coordinates(mol):
        AllChem.Compute2DCoords(mol)
        return mol

    @staticmethod
    def render_molecule(smiles: str):
        mol = MoleculeVisualizer.validate_smiles(smiles)
        if mol is None:
            return None

        mol = MoleculeVisualizer.generate_2d_coordinates(mol)
        img = Draw.MolToImage(mol, size=(400, 400))
        return img