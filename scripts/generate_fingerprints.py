import pandas as pd
import numpy as np

from models.fingerprint_engine import FingerprintEngine

print("Loading dataset...")

df = pd.read_csv("data/chembl_100k.csv")

engine = FingerprintEngine()

fingerprints = []

print("Generating fingerprints...")

for i, smiles in enumerate(df["smiles"]):

    fp = engine.smiles_to_fingerprint(smiles)

    if fp is not None:
        fingerprints.append(fp)
    else:
        fingerprints.append(np.zeros(2048))

    if i % 500 == 0:
        print(f"Processed {i} molecules")

fingerprints = np.array(fingerprints).astype("float32")

np.save("data/chembl_fingerprints.npy", fingerprints)

print("Fingerprints saved.")
print("Shape:", fingerprints.shape)