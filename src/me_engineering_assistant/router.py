# src/me_engineering_assistant/router.py
from __future__ import annotations

from typing import List, Literal, TypedDict

RouteName = Literal["ECU-700", "ECU-800-base", "ECU-800-plus"]


class RoutingDecision(TypedDict):
    """
    Result of the routing step.

    - routes: list of document collections that should be queried
    - reason: human-readable explanation that can be logged or inspected
    """

    routes: List[RouteName]
    reason: str


# Perform simple keyword matching based on known ECU model mentions.
# - ECU-700 Series: ECU-750, ECU-700
# - ECU-800-base:  ECU-850, ECU-800
# - ECU-800-plus:  ECU-850b, "plus" variant
DOC_KEYWORDS = {
    "ECU-700": ["750", "700", "ecu 750", "ecu-750", "ecu700"],
    "ECU-800-base": ["850", "800", "ecu 850", "ecu-850", "ecu800"],
    "ECU-800-plus": ["850b", "850-b", "800 plus", "ecu 850b", "ecu-850b"],
}


def route_question(question: str) -> RoutingDecision:
    """
    Perform rule-based routing without an LLM.

    - Match a keyword to select the corresponding document collection.
    - If no keywords match, default to all collections and let RAG handle
      filtering based on embeddings and similarity search.
    """
    q = question.lower()
    selected: List[RouteName] = []

    for route, keywords in DOC_KEYWORDS.items():
        if any(k in q for k in keywords):
            selected.append(route)  # type: ignore[arg-type]

    if not selected:
        # No matches found at all, so fall back to all manuals.
        selected = ["ECU-700", "ECU-800-base", "ECU-800-plus"]  # type: ignore[assignment]
        reason = "No strong keyword match, falling back to all docs."
    else:
        reason = "Matched keywords for: " + ", ".join(selected)

    return {
        "routes": selected,
        "reason": reason,
    }
