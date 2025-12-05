from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from .vectorstore import build_vectorstores
from .router import route_question
from .rag_chain import rag_answer


class AgentState(TypedDict):
    question: str
    routes: List[str]
    answer: str
    metadata: Dict


# Build the vector library once during module loading (sufficient for small data scenarios)
VECTORSTORES = build_vectorstores()


def router_node(state: AgentState) -> AgentState:
    """LangGraph Node: Route based on the query."""
    decision = route_question(state["question"])
    state["routes"] = decision["routes"]
    state.setdefault("metadata", {})
    state["metadata"]["routing_reason"] = decision["reason"]
    return state


def rag_node(state: AgentState) -> AgentState:
    """LangGraph Node: Performs RAG based on routing to generate responses."""
    routes = state.get("routes") or list(VECTORSTORES.keys())
    answer = rag_answer(state["question"], VECTORSTORES, routes)
    state["answer"] = answer
    return state


def build_agent_graph():
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "rag")
    graph.add_edge("rag", END)

    return graph.compile()


def run_agent(question: str) -> Dict:
    """
    Exposed Simple Interface:
    Given a query, return the final state.
    The state contains:
    - “answer”: The final response
    - “routes”: Which document collections were used
    - “metadata”: Such as routing reasons
    """
    workflow = build_agent_graph()
    final_state: AgentState = workflow.invoke(
        {"question": question, "routes": [], "answer": "", "metadata": {}}
    )
    return final_state
