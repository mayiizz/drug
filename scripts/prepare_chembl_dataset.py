from datasets import load_dataset
import pandas as pd

print("Loading ChEMBL dataset...")

ds = load_dataset("antoinebcx/smiles-molecules-chembl")

data = ds["train"]

print("Total molecules:", len(data))

# Convert to pandas
df = data.to_pandas()

# Keep only SMILES column
df = df[["smiles"]]

# Take subset for now (10k molecules)
subset = df.head(100000)

subset.to_csv("data/chembl_100k.csv", index=False)

print("Subset saved: data/chembl_subset.csv")
print("Total rows:", len(subset))