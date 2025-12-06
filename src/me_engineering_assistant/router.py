# src/me_engineering_assistant/router.py

from typing import List, Literal, TypedDict

RouteName = Literal["ECU-700", "ECU-800-base", "ECU-800-plus"]


class RoutingDecision(TypedDict):
    routes: List[RouteName]
    need_both: bool
    reason: str


# Perform simple keyword matching based on file content:
# - ECU-700 Series: ECU-750, ECU-700
# - ECU-800-base: ECU-850, ECU-800
# - ECU-800-plus: ECU-850b, plus version
DOC_KEYWORDS = {
    "ECU-700": ["750", "700", "ecu 750", "ecu-750", "ecu700"],
    "ECU-800-base": ["850", "800", "ecu 850", "ecu-850", "ecu800"],
    "ECU-800-plus": ["850b", "850-b", "800 plus", "ecu 850b", "ecu-850b"],
}


def route_question(question: str) -> RoutingDecision:
    """
    No LLM required—use keywords for simple routing:
    - Match a keyword to select the corresponding document collection
    - If no keywords match, default to all collections and let RAG handle filtering
    """
    q = question.lower()
    selected: List[RouteName] = []

    for route, keywords in DOC_KEYWORDS.items():
        if any(k in q for k in keywords):
            selected.append(route)  # type: ignore[arg-type]

    if not selected:
        # No matches found at all, so catch all
        selected = ["ECU-700", "ECU-800-base", "ECU-800-plus"]  # type: ignore[assignment]
        reason = "No strong keyword match, falling back to all docs."
    else:
        reason = "Matched keywords for: " + ", ".join(selected)

    return {
        "routes": selected,
        "need_both": len(selected) > 1,
        "reason": reason,
    }
