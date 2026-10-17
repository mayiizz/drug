import pubchempy as pcp


class DrugResolver:

    def name_to_smiles(self, drug_name):

        try:
            compounds = pcp.get_compounds(drug_name, "name")

            if len(compounds) == 0:
                return None

            compound = compounds[0]

            return compound.connectivity_smiles

        except Exception as e:
            print("PubChem lookup failed:", e)
            return None


    def resolve(self, drug_name):

        try:
            compounds = pcp.get_compounds(drug_name, "name")

            if len(compounds) == 0:
                return None

            compound = compounds[0]

            result = {
                "name": compound.iupac_name,
                "cid": compound.cid,
                "smiles": compound.connectivity_smiles,
                "synonyms": compound.synonyms[:10] if compound.synonyms else []
            }

            return result

        except Exception as e:
            print("PubChem lookup failed:", e)
            return None