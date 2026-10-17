import httpx
from langchain_core.tools import tool

@tool
async def search_semantic_scholar(query: str) -> str:
    """Search Semantic Scholar for papers with citation counts and abstracts."""
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": 5,
                "fields": "title,abstract,year,citationCount,externalIds"
            },
            timeout=10.0
        )
        papers = res.json().get("data", [])
        if not papers:
            return "No results found on Semantic Scholar."

        output = []
        for p in papers:
            output.append(
                f"Title: {p.get('title', 'N/A')}\n"
                f"Year: {p.get('year', 'N/A')} | Citations: {p.get('citationCount', 0)}\n"
                f"Abstract: {(p.get('abstract') or 'No abstract available.')[:300]}"
            )
        return "\n---\n".join(output)