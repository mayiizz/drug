"""
ESMFoldClient — calls Meta's free ESMFold REST API to fold a protein sequence
into a 3D structure (.pdb format).

No API key required. Rate limit: ~1 request/second.
API endpoint: https://api.esmatlas.com/foldSequence/v1/pdb/

Reference:
  Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein
  structure with a language model." Science 379, 1123-1130.
  https://doi.org/10.1126/science.ade2574
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

ESMFOLD_URL     = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_SEQ_LEN     = 400    # ESMFold API hard limit
REQUEST_TIMEOUT = 15     # seconds — fail fast, pipeline has fallback
MAX_RETRIES     = 1      # only 1 attempt — fallback seeds handle failure


class ESMFoldClient:
    """
    Folds a protein sequence into a 3D .pdb structure using ESMFold.

    Usage:
        client = ESMFoldClient()
        pdb_string = client.fold(sequence)
    """

    def fold(self, sequence: str) -> str:
        """
        Fold a protein sequence into a PDB-format string.

        Args:
            sequence: Amino acid sequence (single-letter codes).
                      Truncated to MAX_SEQ_LEN if longer.

        Returns:
            PDB-format string of the predicted structure.

        Raises:
            RuntimeError: If the API call fails after retries.
        """
        # ESMFold API has a 400 aa limit — truncate to the N-terminal domain
        # which typically contains the kinase/binding domain for most targets.
        # For EGFR (1210 aa) we take the first 400 aa which covers the
        # extracellular L1 domain. The intracellular kinase domain starts ~712.
        # We preferentially use aa 700-1100 for kinase targets if seq > 400.
        seq = self._select_domain(sequence)

        print(f"[ESMFold] Folding {len(seq)} aa (truncated from {len(sequence)} aa)...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    ESMFOLD_URL,
                    data=seq,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    pdb_str = response.text
                    if "ATOM" in pdb_str:
                        print(f"[ESMFold] Structure predicted successfully "
                              f"({len(pdb_str)} chars, {pdb_str.count('ATOM')} atoms).")
                        return pdb_str
                    else:
                        raise RuntimeError(f"ESMFold returned invalid PDB: {pdb_str[:200]}")

                elif response.status_code == 429:
                    wait = 10 * attempt
                    print(f"[ESMFold] Rate limited — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)

                else:
                    raise RuntimeError(
                        f"ESMFold API error {response.status_code}: {response.text[:200]}"
                    )

            except requests.exceptions.Timeout:
                print(f"[ESMFold] Timeout on attempt {attempt}/{MAX_RETRIES}...")
                if attempt == MAX_RETRIES:
                    raise RuntimeError("ESMFold API timed out after all retries.")
                time.sleep(5 * attempt)

            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot reach ESMFold API. Check internet connection: {e}"
                )

        raise RuntimeError("ESMFold API failed after all retries.")

    def _select_domain(self, sequence: str) -> str:
        """
        Select the most informative domain segment for structure prediction.
        For sequences longer than MAX_SEQ_LEN:
          - If seq > 700 aa (likely a multi-domain kinase), take the
            C-terminal kinase domain region (last 400 aa) which contains
            the ATP binding pocket targeted by most kinase inhibitors.
          - Otherwise take the first 400 aa.
        """
        if len(sequence) <= MAX_SEQ_LEN:
            return sequence

        if len(sequence) > 700:
            # Kinase domain is typically in the C-terminal half
            # Take the last MAX_SEQ_LEN residues
            domain = sequence[-MAX_SEQ_LEN:]
            print(f"[ESMFold] Long sequence ({len(sequence)} aa) — "
                  f"using C-terminal kinase domain (last {MAX_SEQ_LEN} aa).")
        else:
            domain = sequence[:MAX_SEQ_LEN]
            print(f"[ESMFold] Sequence truncated to first {MAX_SEQ_LEN} aa.")

        return domain

    def save_pdb(self, pdb_string: str, path: str) -> None:
        """Save a PDB string to a file."""
        with open(path, "w") as f:
            f.write(pdb_string)
        print(f"[ESMFold] PDB saved to {path}")