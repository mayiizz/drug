import httpx
from langchain_core.tools import tool

@tool
async def search_pubmed(query: str) -> str:
    """Search PubMed for scientific literature on a molecule or target."""
    async with httpx.AsyncClient() as client:
        # Step 1: Get paper IDs
        search_res = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": 5, "retmode": "json"},
            timeout=10.0
        )
        ids = search_res.json()["esearchresult"]["idlist"]
        if not ids:
            return "No papers found on PubMed for this query."

        # Step 2: Fetch abstracts
        fetch_res = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "text", "rettype": "abstract"},
            timeout=10.0
        )
        return fetch_res.text[:4000]