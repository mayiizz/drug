"""
PocketExtractor — extracts binding pocket residues and physicochemical
properties from a PDB structure using pure BioPython + RDKit.

No Java, no fpocket, no external tools required.

Method:
  1. Parse PDB with BioPython
  2. Find the geometric centre of the protein
  3. For kinase targets: identify the likely ATP-binding pocket using
     conserved hinge residues (Glu-Met/Leu pair in helix C + hinge loop)
  4. Extract all residues within POCKET_RADIUS Å of the pocket centre
  5. Compute physicochemical profile of the pocket
     (charge, H-bond donors/acceptors, hydrophobicity, volume estimate)
  6. Map profile → SMILES fragment bias for REINVENT reward shaping

Reference for pocket-biased generation approach:
  Ragoza et al. (2022) "Generating 3D molecules conditional on receptor
  binding sites with deep generative models." Chem. Sci. 13, 2701.
"""

import io
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Pocket extraction radius in Angstroms
POCKET_RADIUS = 10.0

# Residue physicochemical classifications
CHARGED_POS  = {"ARG", "LYS", "HIS"}
CHARGED_NEG  = {"ASP", "GLU"}
HBOND_DONOR  = {"SER", "THR", "TYR", "TRP", "ASN", "GLN", "LYS", "ARG", "HIS"}
HBOND_ACCEPT = {"ASP", "GLU", "SER", "THR", "TYR", "ASN", "GLN", "HIS"}
HYDROPHOBIC  = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "CYS"}

# Conserved kinase hinge residues used to locate ATP pocket
# Format: (residue_name, search_window_start, search_window_end)
# These are the Glu (helix αC) and Met/Leu (hinge) conserved in most kinases
KINASE_HINGE_RESIDUES = {"GLU", "MET", "LEU", "THR", "ALA"}


class PocketExtractor:
    """
    Extracts binding pocket properties from a PDB string.

    Usage:
        pe = PocketExtractor()
        pocket = pe.extract(pdb_string)
        # pocket is a dict with keys:
        #   centre       : (x, y, z) tuple
        #   residues     : list of residue names in pocket
        #   n_hbond_don  : int
        #   n_hbond_acc  : int
        #   n_hydrophobic: int
        #   n_charged    : int
        #   volume_est   : float (Å³)
        #   smiles_bias  : list[str] of fragment SMILES biasing generation
    """

    def extract(self, pdb_string: str) -> dict:
        """
        Extract pocket properties from a PDB-format string.

        Returns a dict describing the binding pocket's physicochemical profile.
        """
        try:
            from Bio import PDB as _PDB
            from Bio.PDB.PDBParser import PDBParser
        except ImportError:
            raise RuntimeError(
                "BioPython is required for pocket extraction. "
                "Install with: pip install biopython"
            )

        # Parse PDB from string
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(
                "protein", io.StringIO(pdb_string)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDB structure: {e}")

        # Collect all CA atoms (alpha carbons — one per residue)
        ca_atoms  = []
        residues  = []
        residue_ids: list[tuple[str, int]] = []  # (chain, resnum) for landmarks
        for model in structure:
            for chain in model:
                chain_id = chain.id if hasattr(chain, "id") else ""
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue   # skip HETATMs
                    if "CA" in residue:
                        ca_atoms.append(residue["CA"].get_vector())
                        residues.append(residue)
                        try:
                            resnum = int(residue.get_id()[1])
                        except Exception:
                            resnum = -1
                        residue_ids.append((str(chain_id), resnum))

        if not ca_atoms:
            raise RuntimeError("No CA atoms found in PDB structure.")

        print(f"[PocketExtractor] {len(ca_atoms)} residues parsed from PDB.")

        # Find pocket centre
        # Strategy: use a predicted kinase hinge region if detectable,
        # otherwise fall back to geometric centre of the full structure.
        pocket_centre, kinase_landmarks = self._find_pocket_centre(
            ca_atoms, residues, residue_ids
        )

        # Extract residues within POCKET_RADIUS of pocket centre
        pocket_residues = []
        for atom, residue in zip(ca_atoms, residues):
            dist = self._distance(atom, pocket_centre)
            if dist <= POCKET_RADIUS:
                pocket_residues.append(residue.get_resname())

        if len(pocket_residues) < 3:
            # Fallback: use a larger radius
            print(f"[PocketExtractor] Pocket too small at {POCKET_RADIUS}Å — "
                  f"expanding to {POCKET_RADIUS * 1.5}Å...")
            for atom, residue in zip(ca_atoms, residues):
                dist = self._distance(atom, pocket_centre)
                if dist <= POCKET_RADIUS * 1.5:
                    pocket_residues.append(residue.get_resname())

        print(f"[PocketExtractor] {len(pocket_residues)} residues in binding pocket.")

        # Compute physicochemical profile
        n_hbond_don   = sum(1 for r in pocket_residues if r in HBOND_DONOR)
        n_hbond_acc   = sum(1 for r in pocket_residues if r in HBOND_ACCEPT)
        n_hydrophobic = sum(1 for r in pocket_residues if r in HYDROPHOBIC)
        n_charged_pos = sum(1 for r in pocket_residues if r in CHARGED_POS)
        n_charged_neg = sum(1 for r in pocket_residues if r in CHARGED_NEG)
        n_charged     = n_charged_pos + n_charged_neg
        volume_est    = (4 / 3) * math.pi * (POCKET_RADIUS ** 3)

        # Map pocket profile → fragment SMILES bias
        # These fragments will be used as seeds for REINVENT
        # to bias generation toward pocket-complementary chemistry
        smiles_bias = self._profile_to_fragments(
            n_hbond_don, n_hbond_acc, n_hydrophobic,
            n_charged_pos, n_charged_neg, len(pocket_residues)
        )

        pocket = {
            "centre":        (pocket_centre[0], pocket_centre[1], pocket_centre[2]),
            "residues":      pocket_residues,
            "n_hbond_don":   n_hbond_don,
            "n_hbond_acc":   n_hbond_acc,
            "n_hydrophobic": n_hydrophobic,
            "n_charged":     n_charged,
            "n_charged_pos": n_charged_pos,
            "n_charged_neg": n_charged_neg,
            "volume_est":    round(volume_est, 1),
            "smiles_bias":   smiles_bias,
            "pocket_size":   len(pocket_residues),
        }

        # Optional kinase landmarks for kinase-aware docking box centering.
        # Each list contains "<chain>:<resnum>" strings. Omitted entirely
        # when no landmarks were confidently detected so downstream code can
        # treat absence as "not a kinase / unknown topology".
        for key in ("hinge_residues", "dfg_residues", "catalytic_residues"):
            ids = kinase_landmarks.get(key) if kinase_landmarks else None
            if ids:
                pocket[key] = ids
        if kinase_landmarks.get("is_kinase_like"):
            pocket["is_kinase_like"] = True

        print(f"[PocketExtractor] Profile — "
              f"HBD:{n_hbond_don} HBA:{n_hbond_acc} "
              f"Hydrophobic:{n_hydrophobic} Charged:{n_charged}")
        print(f"[PocketExtractor] {len(smiles_bias)} fragment seeds generated.")

        return pocket

    def _find_pocket_centre(
        self,
        ca_atoms,
        residues,
        residue_ids: list[tuple[str, int]] | None = None,
    ) -> tuple[tuple[float, float, float], dict]:
        """
        Find the most likely binding pocket centre and any kinase landmarks.

        Returns
        -------
        (centre, landmarks) where ``centre`` is the (x, y, z) pocket centre
        and ``landmarks`` is a dict possibly containing
        ``hinge_residues`` / ``dfg_residues`` / ``catalytic_residues`` lists
        of ``"<chain>:<resnum>"`` strings plus an ``is_kinase_like`` flag.
        """
        res_names = [r.get_resname() for r in residues]
        landmarks: dict[str, list[str] | bool] = {}

        def _id(idx: int) -> str | None:
            if not residue_ids or idx < 0 or idx >= len(residue_ids):
                return None
            chain, resnum = residue_ids[idx]
            if resnum < 0:
                return None
            return f"{chain}:{resnum}"

        # ── Hinge: Glu (αC) ... Met/Leu/Ala (hinge backbone) window ─────
        hinge_idx = None
        for i in range(len(res_names) - 3):
            if (
                res_names[i] == "GLU"
                and res_names[i + 3] in {"MET", "LEU", "ALA"}
            ):
                hinge_idx = i + 1
                break

        if hinge_idx is not None:
            print(
                f"[PocketExtractor] Kinase hinge detected at residue index {hinge_idx}."
            )
            hinge_ids = [
                _id(j)
                for j in range(max(0, hinge_idx - 1), min(len(res_names), hinge_idx + 3))
            ]
            hinge_ids = [s for s in hinge_ids if s]
            if hinge_ids:
                landmarks["hinge_residues"] = hinge_ids

            # ── DFG motif scan (Asp-Phe-Gly), prefer windows downstream of hinge
            dfg_idx = None
            for i in range(max(0, hinge_idx), len(res_names) - 2):
                if (
                    res_names[i] == "ASP"
                    and res_names[i + 1] == "PHE"
                    and res_names[i + 2] == "GLY"
                ):
                    dfg_idx = i
                    break
            if dfg_idx is None:
                # Allow upstream DFG too (unusual numbering)
                for i in range(0, len(res_names) - 2):
                    if (
                        res_names[i] == "ASP"
                        and res_names[i + 1] == "PHE"
                        and res_names[i + 2] == "GLY"
                    ):
                        dfg_idx = i
                        break
            if dfg_idx is not None:
                dfg_ids = [_id(dfg_idx + k) for k in range(3)]
                dfg_ids = [s for s in dfg_ids if s]
                if dfg_ids:
                    landmarks["dfg_residues"] = dfg_ids

            # ── Catalytic K-E-D triad heuristic ─────────────────────────
            #   K is the catalytic Lys upstream of αC-Glu, E is αC-Glu (already
            #   located at hinge_idx-1 in this heuristic), D is from DFG.
            cat_ids: list[str] = []
            # nearest LYS upstream of αC Glu (within 30 residues)
            ac_glu_idx = hinge_idx - 1
            for k_idx in range(ac_glu_idx - 1, max(-1, ac_glu_idx - 30), -1):
                if res_names[k_idx] == "LYS":
                    s = _id(k_idx)
                    if s:
                        cat_ids.append(s)
                    break
            s_glu = _id(ac_glu_idx)
            if s_glu:
                cat_ids.append(s_glu)
            if dfg_idx is not None:
                s_d = _id(dfg_idx)
                if s_d:
                    cat_ids.append(s_d)
            if cat_ids:
                landmarks["catalytic_residues"] = cat_ids

            landmarks["is_kinase_like"] = bool(
                landmarks.get("hinge_residues") and landmarks.get("dfg_residues")
            )

            v = ca_atoms[hinge_idx]
            return ((v[0], v[1], v[2]), landmarks)

        # Fallback: geometric centre of middle third of the protein
        n = len(ca_atoms)
        start = n // 3
        end   = 2 * n // 3
        core  = ca_atoms[start:end]

        cx = sum(v[0] for v in core) / len(core)
        cy = sum(v[1] for v in core) / len(core)
        cz = sum(v[2] for v in core) / len(core)

        print("[PocketExtractor] Using geometric centre of protein core.")
        return ((cx, cy, cz), landmarks)

    def _distance(self, atom_vector, centre: tuple) -> float:
        """Euclidean distance between a BioPython Vector and an (x,y,z) tuple."""
        dx = atom_vector[0] - centre[0]
        dy = atom_vector[1] - centre[1]
        dz = atom_vector[2] - centre[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _profile_to_fragments(
        self,
        n_hbd: int, n_hba: int, n_hydro: int,
        n_pos: int, n_neg: int, pocket_size: int
    ) -> list:
        """
        Map pocket physicochemical profile to a list of SMILES fragments
        that will bias REINVENT generation toward pocket-complementary
        chemical space.

        Logic:
          - H-bond rich pocket → include NH-donors and C=O acceptors
          - Hydrophobic pocket  → include aromatic rings and alkyl chains
          - Charged pocket      → include amines (pos) or carboxylates (neg)
          - Large pocket        → prefer larger scaffolds
        """
        from rdkit import Chem

        # Base scaffold pool — heteroaromatic cores found in kinase inhibitors
        base_scaffolds = [
            "c1cnc2ccccc2n1",                                   # quinazoline
            "c1ccc2[nH]cnc2c1",                                 # benzimidazole
            "c1cnc2[nH]ccc2c1",                                 # 7-azaindole
            "c1ccc2ncccc2c1",                                   # quinoline
            "c1ccc2c(c1)cnn2",                                  # indazole
            "Cc1nc2ccccc2[nH]1",                                # methylbenzimidazole
            "c1cnc(N)nc1N",                                     # diaminopyrimidine
            "c1cc2ccccc2nc1N",                                  # aminoquinoline
        ]

        # H-bond donor pocket → add NH-containing fragments
        hbd_fragments = [
            "Nc1ccncc1",          # aminopyridine (H-bond donor, ATP mimic)
            "Nc1ncnc2ccccc12",    # aminoquinazoline
            "O=C(N)c1ccncc1",     # nicotinamide
            "Nc1ccc(O)cc1",       # aminophenol
        ] if n_hbd >= 3 else []

        # H-bond acceptor pocket → add carbonyl/ether acceptors
        hba_fragments = [
            "O=C1CCCO1",          # gamma-butyrolactone
            "COc1ccncc1",         # methoxypyridine
            "O=c1cc[nH]cc1=O",   # pyrimidinedione
            "O=C(O)c1ccncc1",    # nicotinic acid
        ] if n_hba >= 3 else []

        # Hydrophobic pocket → add aromatic/lipophilic fragments
        hydro_fragments = [
            "c1ccc(-c2ccccc2)cc1",   # biphenyl
            "c1ccc2ccccc2c1",         # naphthalene
            "C1CCCCC1",               # cyclohexane
            "c1ccsc1",                # thiophene
            "c1ccoc1",                # furan
        ] if n_hydro >= 4 else []

        # Positively charged pocket → add basic nitrogen fragments
        pos_fragments = [
            "C1CCN(CC1)c1ccncc1",    # piperidyl-pyridine
            "CN1CCNCC1",              # methylpiperazine
            "c1cnc(N2CCNCC2)nc1",    # pyrimidyl-piperazine
        ] if n_pos >= 2 else []

        # Negatively charged pocket → add H-bond donors
        neg_fragments = [
            "Nc1ccc(S(=O)(=O)N)cc1",  # sulfonamide
            "OC(=O)c1ccncc1",          # nicotinic acid
        ] if n_neg >= 2 else []

        # Large pocket → prefer larger drug-like scaffolds
        large_fragments = [
            "C(#N)c1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",    # erlotinib-like
            "COc1cc2ncnc(N)c2cc1OC",                             # QZ scaffold
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",  # celecoxib-like
        ] if pocket_size >= 10 else []

        # Combine all fragments
        all_frags = (
            base_scaffolds +
            hbd_fragments  +
            hba_fragments  +
            hydro_fragments +
            pos_fragments  +
            neg_fragments  +
            large_fragments
        )

        # Validate with RDKit — only return parseable SMILES
        valid = []
        for smi in all_frags:
            if Chem.MolFromSmiles(smi) is not None:
                valid.append(smi)

        # Deduplicate
        seen = set()
        unique = []
        for smi in valid:
            if smi not in seen:
                seen.add(smi)
                unique.append(smi)

        return unique