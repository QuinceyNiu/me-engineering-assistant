import csv
import time
from pathlib import Path

from me_engineering_assistant.config import TEST_QUESTIONS_PATH
from me_engineering_assistant.graph import run_agent


def test_agent_answers_most_questions():
    """
    End-to-end test:
    - Load questions from test-questions.csv
    - Run the full agent pipeline on each question
    - Print a small benchmark summary (per-question latency)
    - Ensure at least 80% questions receive a non-fallback answer
    """
    path = Path(TEST_QUESTIONS_PATH)
    assert path.exists(), "test-questions.csv not found"

    # 1) Load questions from CSV
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in reader.fieldnames or []]

        if "Question" in fieldnames:
            col = "Question"
        elif "question" in fieldnames:
            col = "question"
        else:
            raise AssertionError(
                f"CSV header must contain a Question column, got: {fieldnames}"
            )

        for row in reader:
            q = row.get(col)
            if q:
                questions.append(q.strip())

    assert questions, "No questions loaded from test-questions.csv"

    # 2) Run agent on each question, record correctness & latency
    results: list[dict] = []
    answered = 0

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
                "answer": ans,
                "ok": ok,
                "time": elapsed,
            }
        )

    total = len(results)
    ratio = answered / total
    avg_time = sum(r["time"] for r in results) / total
    max_time = max(r["time"] for r in results)

    # 3) Print benchmark summary (visible with `pytest -q -s`)
    print("\n=== ME Engineering Assistant: E2E Benchmark ===")
    for idx, r in enumerate(results, start=1):
        status = "OK " if r["ok"] else "MISS"
        print(
            f"{idx:02d}. [{status}] {r['time']:.2f}s - {r['question']}"
        )

    print("\nSummary:")
    print(f"- Questions      : {total}")
    print(f"- Answered       : {answered} ({ratio:.0%})")
    print(f"- Avg time / q   : {avg_time:.2f}s")
    print(f"- Max time / q   : {max_time:.2f}s\n")

    # 4) Main success criterion from the challenge:
    # at least 80% of questions should be answered successfully.
    assert ratio >= 0.8, f"Answer ratio too low: {ratio:.2f}"
