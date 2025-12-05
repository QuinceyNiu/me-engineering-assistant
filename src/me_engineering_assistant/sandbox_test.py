"""
调试脚本：直接在 IDE 里运行，用来快速验证 Agent 的效果。
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
