"""
Debugging script: Run directly within the IDE to quickly validate the Agent's functionality.
"""

from .graph import run_agent


def main() -> None:
    question = "What is the maximum operating temperature for the ECU-850b?"
    result = run_agent(question)

    print("Question:", question)
    print("Routes:", result.get("routes"))
    print("Routing reason:", result.get("metadata", {}).get("routing_reason"))
    print("Answer:\n", result["answer"])


if __name__ == "__main__":
    main()
