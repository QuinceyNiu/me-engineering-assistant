"""Rule-based routing for selecting the appropriate ECU manual collection.

Inspects a user question (e.g., model identifiers/keywords) and chooses the best manual(s)
to query, keeping retrieval focused and reducing cross-manual noise.
"""

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
    matches = re.findall(r"\becu[-\s]?(\d{3})([a-z])?\b", q)

    if matches:
        # Normalize all mentioned models, e.g. [("850",""),("850","b")] -> ["850","850b"]
        mentioned_models = [f"{num}{suffix or ''}".lower() for (num, suffix) in matches]

        model_to_routes: dict[str, List[RouteName]] = {
            "700": ["ECU-700"],
            "750": ["ECU-700"],
            "800": ["ECU-800-base"],
            "850": ["ECU-800-base"],
            "850b": ["ECU-800-plus"],
        }

        unknown = sorted({m for m in mentioned_models if m not in model_to_routes})
        if unknown:
            return {
                "routes": [],
                "reason": "Unknown ECU model(s) "
                          + ", ".join(f"ECU-{m.upper()}" for m in unknown)
                          + ", refusing broad fallback.",
            }

        # Union routes across all mentioned models, keep stable order
        routes: List[RouteName] = []
        for m in mentioned_models:
            for r in model_to_routes[m]:
                if r not in routes:
                    routes.append(r)

        return {
            "routes": routes,
            "reason": "Explicit ECU model(s) "
                      + ", ".join(f"ECU-{m.upper()}" for m in sorted(set(mentioned_models)))
                      + " -> "
                      + ", ".join(routes),
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
