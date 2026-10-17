import httpx
from langchain_core.tools import tool

@tool
async def search_web(query: str) -> str:
    """Search the web for recent drug discovery news, preprints, and clinical updates.
    Uses DuckDuckGo instant answer API — no API key required."""

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                },
                timeout=10.0,
                headers={"User-Agent": "CURA-AI-Research-Agent/1.0"}
            )

        data = res.json()
        output_parts = []

        # Instant answer / abstract
        abstract = data.get("AbstractText", "").strip()
        abstract_source = data.get("AbstractSource", "")
        if abstract:
            output_parts.append(f"Summary ({abstract_source}): {abstract}")

        # Related topics
        topics = data.get("RelatedTopics", [])
        for topic in topics[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                url  = topic.get("FirstURL", "")
                text = topic.get("Text", "")[:300]
                output_parts.append(f"• {text}\n  URL: {url}")

        return "\n\n".join(output_parts) if output_parts else (
            "No instant results found. PubMed and Semantic Scholar results should suffice."
        )

    except Exception as e:
        return f"Web search failed: {str(e)}. Relying on PubMed and Semantic Scholar only."