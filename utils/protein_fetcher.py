import re
import requests
import logging

logger = logging.getLogger(__name__)

# Valid gene name: 1–10 uppercase letters/digits, must start with a letter
GENE_NAME_RE       = re.compile(r'^[A-Z][A-Z0-9]{1,9}$')
# UniProt accession: e.g. P00533, Q9Y6K9
UNIPROT_ACCESSION_RE = re.compile(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$')


def _is_valid_query(query: str) -> bool:
    q = query.strip().upper()
    if GENE_NAME_RE.match(q):
        return True
    if UNIPROT_ACCESSION_RE.match(q):
        return True
    return False


def fetch_protein_sequence(protein_name: str) -> dict | None:
    """
    Fetch protein data from UniProt.

    Returns dict with keys: name, uniprot_id, sequence
    Returns {"error": "..."} on failure.
    Returns None if query format is clearly invalid.
    """
    query = protein_name.strip()

    # Layer 1: Format validation
    if not _is_valid_query(query):
        logger.warning(f"Protein query rejected (invalid format): '{query}'")
        return {
            "error": f"'{query}' is not a valid gene name or UniProt accession. "
                     "Please use a gene name like EGFR, HER2, TP53, or ACE2."
        }

    # Layer 2: UniProt API — reviewed Swiss-Prot + human first
    # reviewed:true  = manually curated canonical entries only (no isoforms/fragments)
    # organism_id:9606 = Homo sapiens
    data = None
    for search_q in [
        f"gene_exact:{query}+AND+reviewed:true+AND+organism_id:9606",
        f"{query}+AND+reviewed:true+AND+organism_id:9606",
        f"{query}+AND+reviewed:true",
    ]:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/search?query={search_q}&format=json&size=5"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            candidate = response.json()
            if candidate.get("results"):
                data = candidate
                break
        except Exception as e:
            logger.error(f"UniProt API error: {e}")
            return {"error": "Could not reach UniProt API. Check network connection."}

    if data is None or not data.get("results"):
        return {"error": f"Protein '{query}' not found in UniProt reviewed entries. "
                         "Please use an official gene name like EGFR, HER2, TP53, BRCA1."}

    # Layer 3: STRICT result matching
    # The returned protein must EXACTLY match the query as:
    #   - a UniProt accession (e.g. P00533)
    #   - an official gene symbol (e.g. EGFR, exact match only)
    #   - a gene synonym (exact match only)
    # Substring / partial matches are NOT accepted.
    # This prevents "RUDRA" from matching a protein that happens to contain
    # those letters somewhere in its name.
    query_upper = query.upper()
    matched = None

    for entry in data["results"]:
        accession = entry.get("primaryAccession", "").upper()

        # Collect all gene symbols and synonyms for this entry
        gene_names = set()
        for genes in entry.get("genes", []):
            primary = genes.get("geneName", {}).get("value", "").upper()
            if primary:
                gene_names.add(primary)
            for syn in genes.get("synonyms", []):
                s = syn.get("value", "").upper()
                if s:
                    gene_names.add(s)

        # Exact match only — accession or gene symbol
        if query_upper == accession or query_upper in gene_names:
            matched = entry
            break

    if matched is None:
        # NO fallback — if the query doesn't exactly match any returned gene name
        # or accession, reject it. This prevents "RUDRA", "SMILE", "CANCER" etc.
        # from returning random unrelated proteins.
        return {
            "error": (
                f"'{query}' did not match any known human protein gene name. "
                "UniProt returned results but none matched your query exactly. "
                "Please use an official HGNC gene symbol like EGFR, HER2, TP53, "                "BRCA1, ACE2, KRAS, or a UniProt accession like P00533."
            )
        }

    try:
        sequence   = matched["sequence"]["value"]
        accession  = matched["primaryAccession"]
        gene_name  = (
            matched.get("genes", [{}])[0]
                   .get("geneName", {})
                   .get("value", query)
        )
        full_name  = (
            matched.get("proteinDescription", {})
                   .get("recommendedName", {})
                   .get("fullName", {})
                   .get("value", gene_name)
        )
    except (KeyError, IndexError) as e:
        logger.error(f"Parsing UniProt response failed: {e}")
        return {"error": "Unexpected UniProt response format."}

    return {
        "name":       full_name,          # e.g. "Epidermal growth factor receptor"
        "gene":       gene_name,          # e.g. "EGFR"
        "uniprot_id": accession,          # e.g. "P00533"
        "sequence":   sequence,
    }