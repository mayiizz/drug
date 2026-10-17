from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64


def smiles_to_base64(smiles: str):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(300,200))

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{encoded}"