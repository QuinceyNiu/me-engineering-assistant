"""
Standalone benchmark script for the ME Engineering Assistant.

This script is intentionally placed under `tests/` but is NOT a pytest test.
Run it directly with Python to inspect:
- Per-question latency
- Per-question answer
- Overall accuracy and timing statistics

Usage:
    python -m tests.benchmark
or, if your PYTHONPATH is set:
    python tests/benchmark.py
"""

import csv
import time
from pathlib import Path

from me_engineering_assistant.config import TEST_QUESTIONS_PATH
from me_engineering_assistant.graph import run_agent


def load_questions() -> list[str]:
    """Load all questions from the configured CSV file."""
    path = Path(TEST_QUESTIONS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"test-questions.csv not found at: {path}")

    questions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in reader.fieldnames or []]

        if "Question" in fieldnames:
            col = "Question"
        elif "question" in fieldnames:
            col = "question"
        else:
            raise ValueError(
                f"CSV header must contain a Question column, got: {fieldnames}"
            )

        for row in reader:
            q = row.get(col)
            if q:
                questions.append(q.strip())

    if not questions:
        raise ValueError("No questions loaded from test-questions.csv")

    return questions


def run_benchmark() -> None:
    """Run the agent on all questions and print a human-readable benchmark."""
    questions = load_questions()

    results: list[dict] = []
    answered = 0
    total_start = time.perf_counter()

    for q in questions:
        start = time.perf_counter()
        state = run_agent(q)
        elapsed = time.perf_counter() - start

        ans = state["answer"]
        ok = bool(
            ans and "The manual does not provide this information." not in ans
        )

        if ok:
            answered += 1

        results.append(
            {
                "question": q,
                "answer": ans or "",
                "ok": ok,
                "time": elapsed,
            }
        )

    total_elapsed = time.perf_counter() - total_start
    total = len(results)
    ratio = answered / total
    avg_time = sum(r["time"] for r in results) / total
    max_time = max(r["time"] for r in results)

    # Pretty print results
    print("\n=== ME Engineering Assistant: Benchmark ===")
    for idx, r in enumerate(results, start=1):
        status = "OK " if r["ok"] else "MISS"
        # Truncate long answers for readability
        ans_snippet = r["answer"].replace("\n", " ")
        if len(ans_snippet) > 300:
            ans_snippet = ans_snippet[:297] + "..."

        print(f"{idx:02d}. [{status}] {r['time']:.2f}s")
        print(f"    Q: {r['question']}")
        print(f"    A: {ans_snippet}")

    print("\nSummary:")
    print(f"- Questions         : {total}")
    print(f"- Answered          : {answered} ({ratio:.0%})")
    print(f"- Avg time / q      : {avg_time:.2f}s")
    print(f"- Max time / q      : {max_time:.2f}s")
    print(f"- Total runtime     : {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    run_benchmark()
