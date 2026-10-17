import requests
import pandas as pd
import os

# Ensure output directory exists
os.makedirs("../data/raw", exist_ok=True)

url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

params = {
    "target_chembl_id": "CHEMBL203",
    "standard_type": "IC50",
    "standard_relation": "<",
    "standard_value": 100,
    "limit": 1000
}

all_data = []

print("📥 Fetching EGFR data from ChEMBL...")

while url:
    response = requests.get(url, params=params).json()
    all_data.extend(response['activities'])
    url = response['page_meta']['next']
    params = {}

df = pd.DataFrame(all_data)

# Keep only required columns
df = df[['canonical_smiles', 'standard_value']].dropna()

# Save file
output_path = "../data/raw/egfr_raw.csv"
df.to_csv(output_path, index=False)

print(f"✅ Saved {len(df)} molecules to {output_path}")