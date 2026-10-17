import pandas as pd
import numpy as np
from models.chemberta_encoder import ChemBERTaEncoder

print("Loading dataset...")

df = pd.read_csv("data/chembl_subset.csv")

encoder = ChemBERTaEncoder()

embeddings = []

print("Generating embeddings...")

for i, smiles in enumerate(df["smiles"]):

    vector = encoder.encode(smiles)

    embeddings.append(vector)

    if i % 500 == 0:
        print(f"Processed {i} molecules")

embeddings = np.array(embeddings).astype("float32")

np.save("data/chembl_embeddings.npy", embeddings)

print("Embeddings saved to data/chembl_embeddings.npy")
print("Shape:", embeddings.shape)