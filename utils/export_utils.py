import csv
from rdkit import Chem


def export_csv(molecules, filename="generated_molecules.csv"):

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["SMILES"])

        for m in molecules:
            writer.writerow([m])

    return filename


def export_sdf(molecules, filename="generated_molecules.sdf"):

    writer = Chem.SDWriter(filename)

    for sm in molecules:

        mol = Chem.MolFromSmiles(sm)

        if mol:
            writer.write(mol)

    writer.close()

    return filename