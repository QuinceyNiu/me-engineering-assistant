import csv
from pathlib import Path

from me_engineering_assistant.config import TEST_QUESTIONS_PATH
from me_engineering_assistant.graph import run_agent


def test_agent_answers_most_questions():
    path = Path(TEST_QUESTIONS_PATH)
    assert path.exists(), "test-questions.csv not found"

    questions = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in reader.fieldnames or []]

        if "Question" in fieldnames:
            col = "Question"
        elif "question" in fieldnames:
            col = "question"
        else:
            raise AssertionError(f"CSV header must contain a Question column, got: {fieldnames}")

        for row in reader:
            q = row.get(col)
            if q:
                questions.append(q.strip())

    assert questions, "No questions loaded from test-questions.csv"

    answered = 0
    for q in questions:
        state = run_agent(q)
        ans = state["answer"]
        if ans and "The manual does not provide this information." not in ans:
            answered += 1

    ratio = answered / len(questions)
    assert ratio >= 0.8, f"Answer ratio too low: {ratio:.2f}"
