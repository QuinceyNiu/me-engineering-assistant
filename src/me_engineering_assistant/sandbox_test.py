"""
Debugging script: Run directly within the IDE to quickly validate the Agent's functionality.
"""

from .graph import run_agent

def main() -> None:
    """Simple CLI smoke test for the agent using a hard-coded example question."""

    question = "How much RAM does the ECU-550 have?"
    result = run_agent(question)

    print("Question:", question)
    print("Routes:", result.get("routes"))
    print("Routing reason:", result.get("metadata", {}).get("routing_reason"))
    print("Answer:\n", result["answer"])


if __name__ == "__main__":
    main()
