# src/me_engineering_assistant/router.py
from __future__ import annotations

import re
from typing import List, Literal, TypedDict

RouteName = Literal["ECU-700", "ECU-800-base", "ECU-800-plus"]


class RoutingDecision(TypedDict):
    routes: List[RouteName]
    reason: str


DOC_KEYWORDS = {
    "ECU-700": ["750", "700", "ecu 750", "ecu-750", "ecu700"],
    "ECU-800-base": ["850", "800", "ecu 850", "ecu-850", "ecu800"],
    "ECU-800-plus": ["850b", "850-b", "800 plus", "ecu 850b", "ecu-850b"],
}


def route_question(question: str) -> RoutingDecision:
    """
    Perform rule-based routing without an LLM.

    Fix:
    - If the question explicitly mentions an ECU model (e.g., ECU-550) and it is unknown,
      return routes=[] to avoid hallucinated answers (do NOT fall back to all docs).
    - If no explicit model is mentioned, keep the original behavior.
    """
    q = question.lower()

    # Detect patterns like "ECU-750", "ECU 850b", etc.
    m = re.search(r"\becu[-\s]?(\d{3})([a-z])?\b", q)
    if m:
        model = f"{m.group(1)}{m.group(2) or ''}".lower()  # e.g. "750", "850b"
        known_models = {"700", "750", "800", "850", "850b"}
        if model not in known_models:
            return {
                "routes": [],
                "reason": f"Unknown ECU model ECU-{model.upper()}, refusing broad fallback.",
            }

    selected: List[RouteName] = []
    for route, keywords in DOC_KEYWORDS.items():
        if any(k in q for k in keywords):
            selected.append(route)  # type: ignore[arg-type]

    if not selected:
        selected = ["ECU-700", "ECU-800-base", "ECU-800-plus"]  # type: ignore[assignment]
        reason = "No strong keyword match, falling back to all docs."
    else:
        reason = "Matched keywords for: " + ", ".join(selected)

    return {"routes": selected, "reason": reason}
