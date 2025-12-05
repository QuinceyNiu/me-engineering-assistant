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


# 在模块加载时构建一次向量库（小数据场景足够）
VECTORSTORES = build_vectorstores()


def router_node(state: AgentState) -> AgentState:
    """LangGraph 节点：根据问题做路由。"""
    decision = route_question(state["question"])
    state["routes"] = decision["routes"]
    state.setdefault("metadata", {})
    state["metadata"]["routing_reason"] = decision["reason"]
    return state


def rag_node(state: AgentState) -> AgentState:
    """LangGraph 节点：根据路由做 RAG，生成回答。"""
    routes = state.get("routes") or list(VECTORSTORES.keys())
    answer = rag_answer(state["question"], VECTORSTORES, routes)
    state["answer"] = answer
    return state


def build_agent_graph():
    """构建 LangGraph workflow。"""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "rag")
    graph.add_edge("rag", END)

    return graph.compile()


def run_agent(question: str) -> Dict:
    """
    对外暴露的简易接口：给一个问题，返回最终 state。
    state 包含：
    - "answer": 最终回答
    - "routes": 使用了哪些文档集合
    - "metadata": 例如路由原因
    """
    workflow = build_agent_graph()
    final_state: AgentState = workflow.invoke(
        {"question": question, "routes": [], "answer": "", "metadata": {}}
    )
    return final_state
