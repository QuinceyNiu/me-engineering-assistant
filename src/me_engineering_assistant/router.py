# src/me_engineering_assistant/router.py

from typing import List, Literal, TypedDict

RouteName = Literal["ECU-700", "ECU-800-base", "ECU-800-plus"]


class RoutingDecision(TypedDict):
    routes: List[RouteName]
    need_both: bool
    reason: str


# 根据文件内容的直觉做简单关键词匹配：
# - ECU-700 系列：ECU-750、ECU-700
# - ECU-800-base：ECU-850、ECU-800
# - ECU-800-plus：ECU-850b、plus 版
DOC_KEYWORDS = {
    "ECU-700": ["750", "700", "ecu 750", "ecu-750", "ecu700"],
    "ECU-800-base": ["850", "800", "ecu 850", "ecu-850", "ecu800"],
    "ECU-800-plus": ["850b", "850-b", "800 plus", "ecu 850b", "ecu-850b"],
}


def route_question(question: str) -> RoutingDecision:
    """
    不用 LLM，只用关键词做朴素路由：
    - 命中哪个关键词，就选哪个文档集合
    - 如果一个都没命中，就默认选全部，交给 RAG 去过滤
    """
    q = question.lower()
    selected: List[RouteName] = []

    for route, keywords in DOC_KEYWORDS.items():
        if any(k in q for k in keywords):
            selected.append(route)  # type: ignore[arg-type]

    if not selected:
        # 完全没匹配到，就全部兜底
        selected = ["ECU-700", "ECU-800-base", "ECU-800-plus"]  # type: ignore[assignment]
        reason = "No strong keyword match, falling back to all docs."
    else:
        reason = "Matched keywords for: " + ", ".join(selected)

    return {
        "routes": selected,
        "need_both": len(selected) > 1,
        "reason": reason,
    }
