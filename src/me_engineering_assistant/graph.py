"""LangGraph orchestration for the ME Engineering Assistant.

Defines the agent workflow graph (routing -> retrieval -> generation) and exposes a simple
runner entrypoint for executing the graph on a user question.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .router import route_question
from .rag_chain import rag_answer, FALLBACK_ANSWER
from .vectorstore import build_vectorstores


class AgentState(TypedDict):
    """State container flowing through the LangGraph workflow."""

    question: str
    routes: List[str]
    answer: str
    metadata: Dict


# In-memory cache for the vectorstores. This is built lazily on first access
# and reused for all subsequent queries within the same process.
_VECTORSTORES: Dict[str, object] | None = None


def get_vectorstores() -> Dict[str, object]:
    """
    Lazily build and cache the vectorstores.

    For the small ECU manuals used in this challenge it is reasonable to keep
    the in-memory vectorstores for the lifetime of the process.
    """
    global _VECTORSTORES

    if _VECTORSTORES is None:
        _VECTORSTORES = build_vectorstores()

    return _VECTORSTORES


def router_node(state: AgentState) -> AgentState:
    """LangGraph node: classify which ECU manuals are relevant for the query."""
    decision = route_question(state["question"])
    state["routes"] = decision["routes"]
    state.setdefault("metadata", {})
    state["metadata"]["routing_reason"] = decision["reason"]
    return state


def rag_node(state: AgentState) -> AgentState:
    """LangGraph node: perform RAG based on routing to generate responses."""
    vs_dict = get_vectorstores()

    routes = state.get("routes", None)

    # IMPORTANT:
    # - None  => no routing decision; we may fall back to all docs
    # - []    => explicitly no match (e.g., unknown ECU model); return fallback
    if routes is None:
        routes = list(vs_dict.keys())

    if isinstance(routes, list) and len(routes) == 0:
        state["answer"] = FALLBACK_ANSWER
        state.setdefault("metadata", {})
        state["metadata"]["rag_reason"] = "No matching manuals for the requested ECU model."
        return state

    answer = rag_answer(state["question"], vs_dict, routes)
    state["answer"] = answer
    return state


def build_agent_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "rag")
    graph.add_edge("rag", END)

    return graph.compile()


# Compile the workflow once at module import time and reuse it.
WORKFLOW = build_agent_graph()


def run_agent(question: str) -> AgentState:
    """
    Convenience entrypoint for running the agent.

    The returned state contains:
    - "answer": final response string
    - "routes": which document collections were used
    - "metadata": additional information such as routing reasons
    """
    final_state: AgentState = WORKFLOW.invoke(
        {
            "question": question,
            "routes": [],
            "answer": "",
            "metadata": {},
        },
    )
    return final_state
