# ================================
# research_agent.py — query-first literature survey
# ================================

import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage

from models.research.rag_agent import run_rag, fetch_pubmed


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    error: str
    literature: list


def parse_input(state):
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if isinstance(content, str):
        query = content.strip()
    else:
        query = str(content).strip()

    if not query:
        return {
            "query": "",
            "error": "Please provide a valid molecule or use a natural language query.",
            "literature": [],
        }

    return {"query": query, "error": "", "literature": []}


async def fetch_literature(state):
    if state.get("error"):
        return {"literature": []}

    query = state["query"]
    papers = await fetch_pubmed(query)
    return {"literature": papers}


def synthesize(state):
    if state.get("error"):
        return {"messages": [AIMessage(content=state["error"])]}

    result = run_rag(
        state.get("literature", []),
        query=state["query"],
    )

    return {"messages": [AIMessage(content=result)]}


graph = StateGraph(AgentState)

graph.add_node("parse_input", parse_input)
graph.add_node("fetch_literature", fetch_literature)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("parse_input")

graph.add_edge("parse_input", "fetch_literature")
graph.add_edge("fetch_literature", "synthesize")
graph.add_edge("synthesize", END)

research_agent = graph.compile()
