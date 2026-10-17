import pandas as pd
import json
import numpy as np
from models.similarity.similarity_engine import FingerprintEngine

class SideEffectEngine:
    def __init__(self):
        # Load dataset
        self.data = pd.read_csv("data/sider_side_effects.csv")
        self.data["drugname"] = self.data["drugname"].str.lower().str.strip()

        # Load pre-fetched SMILES mapping
        try:
            with open("data/sider_smiles.json", "r") as f:
                self.sider_smiles = json.load(f)
        except:
            self.sider_smiles = {}

        # Fingerprint engine
        self.fp_engine = FingerprintEngine()
        
        # Pre-compute fingerprints for SIDER drugs
        self.sider_fps = {}
        for name, smiles in self.sider_smiles.items():
            try:
                fp = self.fp_engine.smiles_to_fingerprint(smiles)
                if fp is not None:
                    self.sider_fps[name] = fp
            except:
                continue

    def tanimoto_similarity(self, fp1, fp2):
        intersection = np.logical_and(fp1, fp2).sum()
        union = np.logical_or(fp1, fp2).sum()
        return intersection / union if union > 0 else 0

    def predict_side_effects(self, query_smiles):
        try:
            query_fp = self.fp_engine.smiles_to_fingerprint(query_smiles)
            if query_fp is None:
                return {}
        except:
            return {}

        # Compute similarity with all known SIDER drugs
        similarities = []
        for name, fp in self.sider_fps.items():
            sim = self.tanimoto_similarity(query_fp, fp)
            similarities.append((name, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:3] # Use top 3 similar drugs

        predicted_effects = {}
        for name, score in top_matches:
            # Boost score if very similar
            weight = score * 2 if score > 0.8 else score
            
            # Lookup effects for this drug
            effects = self.data[self.data["drugname"] == name]["pt"].unique()
            for effect in effects:
                if effect not in predicted_effects:
                    predicted_effects[effect] = 0
                predicted_effects[effect] += weight

        return predicted_effects

    def get_side_effects(self, drug_name):
        drug_name = drug_name.lower().strip()
        rows = self.data[self.data["drugname"] == drug_name]
        return rows["pt"].unique().tolist()
