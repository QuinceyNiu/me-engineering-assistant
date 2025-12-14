"""
Standalone benchmark script for the ME Engineering Assistant.

Run:
    python -m tests.benchmark
"""

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from me_engineering_assistant.config import TEST_QUESTIONS_PATH
from me_engineering_assistant.graph import run_agent


# ----------------------------
# Helpers: parsing + scoring
# ----------------------------

MODEL_RE = re.compile(r"\bECU-\d{3}[a-z]?\b", re.IGNORECASE)
# numbers with units we care about (temperature, capacity, current, speed)
UNIT_PATTERNS: list[re.Pattern] = [
    re.compile(r"[-+]\s?\d+\s?°\s?C", re.IGNORECASE),            # +85°C, -40°C
    re.compile(r"\b\d+(\.\d+)?\s?(GB|MB)\b", re.IGNORECASE),     # 2 GB, 16GB, 2 MB
    re.compile(r"\b\d+(\.\d+)?\s?(mA|A)\b", re.IGNORECASE),      # 550mA, 1.7A
    re.compile(r"\b\d+(\.\d+)?\s?(TOPS)\b", re.IGNORECASE),      # 5 TOPS
    re.compile(r"\b\d+(\.\d+)?\s?(Mbps|Gbps)\b", re.IGNORECASE), # 1 Mbps
]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_models(text: str) -> set[str]:
    return {m.upper() for m in MODEL_RE.findall(text or "")}


def extract_unit_tokens(text: str) -> set[str]:
    text = text or ""
    tokens: set[str] = set()
    for pat in UNIT_PATTERNS:
        for m in pat.findall(text):
            # m can be tuple depending on pattern, use match object instead
            pass
    # Use finditer for robust capture
    for pat in UNIT_PATTERNS:
        for mo in pat.finditer(text):
            tok = mo.group(0)
            # normalize common formatting
            tok = tok.replace(" ", "")
            tok = tok.replace("°C", "°C")  # no-op, clarity
            tok = tok.upper()
            tokens.add(tok)
    return tokens


def has_any_contradiction(expected_tokens: set[str], answer_tokens: set[str]) -> bool:
    """
    Very simple contradiction heuristic:
    If expected contains '16GB' but answer contains '32GB' and expected doesn't contain 32GB,
    that's a contradiction for capacity questions, etc.
    """
    # If the expected includes any token with unit, we try to detect a "wrong other value" in answer.
    # Only apply for GB/MB/°C/A/mA/Mbps/TOPS tokens.
    expected = expected_tokens
    answer = answer_tokens

    # Group by unit type
    def unit_type(tok: str) -> str:
        if "°C" in tok:
            return "TEMP"
        if tok.endswith("GB") or tok.endswith("MB"):
            return "CAP"
        if tok.endswith("MA") or tok.endswith("A"):
            return "CURR"
        if tok.endswith("MBPS") or tok.endswith("GBPS"):
            return "SPEED"
        if tok.endswith("TOPS"):
            return "TOPS"
        return "OTHER"

    exp_by_unit: dict[str, set[str]] = {}
    ans_by_unit: dict[str, set[str]] = {}
    for t in expected:
        exp_by_unit.setdefault(unit_type(t), set()).add(t)
    for t in answer:
        ans_by_unit.setdefault(unit_type(t), set()).add(t)

    # Contradiction if answer includes a unit token of same type that is NOT in expected
    # AND expected had at least one token of that type.
    for u, exp_set in exp_by_unit.items():
        if u == "OTHER":
            continue
        ans_set = ans_by_unit.get(u, set())
        extra = ans_set - exp_set
        if extra and exp_set:
            return True

    return False


@dataclass
class QAItem:
    question: str
    expected: str


def load_items() -> list[QAItem]:
    path = Path(TEST_QUESTIONS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"test-questions.csv not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        # Question column
        if "Question" in headers:
            q_col = "Question"
        elif "question" in headers:
            q_col = "question"
        else:
            raise ValueError(f"CSV header must contain a Question column, got: {headers}")

        # Expected column
        exp_col = None
        for cand in ("Expected_Answer", "expected_answer", "Expected Answer", "expected"):
            if cand in headers:
                exp_col = cand
                break
        if not exp_col:
            raise ValueError(
                f"CSV header must contain an Expected_Answer column, got: {headers}"
            )

        items: list[QAItem] = []
        for row in reader:
            q = normalize(row.get(q_col, ""))
            exp = normalize(row.get(exp_col, ""))
            if q:
                items.append(QAItem(question=q, expected=exp))

    if not items:
        raise ValueError("No questions loaded from test-questions.csv")

    return items


def grade_answer(expected: str, answer: str) -> tuple[str, float, list[str]]:
    """
    Returns: (status, score_0_1, reasons)
    status in {OK, PARTIAL, MISS}
    """
    expected_n = normalize(expected)
    answer_n = normalize(answer)

    exp_models = extract_models(expected_n)
    ans_models = extract_models(answer_n)

    exp_tokens = extract_unit_tokens(expected_n)
    ans_tokens = extract_unit_tokens(answer_n)

    reasons: list[str] = []

    # Model coverage (only if expected contains models)
    model_hit = 1.0
    if exp_models:
        hit = len(exp_models & ans_models)
        model_hit = hit / max(1, len(exp_models))
        if model_hit < 1.0:
            missing = sorted(exp_models - ans_models)
            if missing:
                reasons.append(f"missing_models={missing}")

    # Numeric/unit token coverage (only if expected has tokens)
    token_hit = 1.0
    if exp_tokens:
        hit = len(exp_tokens & ans_tokens)
        token_hit = hit / max(1, len(exp_tokens))
        if token_hit < 1.0:
            missing = sorted(exp_tokens - ans_tokens)
            if missing:
                reasons.append(f"missing_values={missing}")

    contradiction = has_any_contradiction(exp_tokens, ans_tokens)
    if contradiction:
        reasons.append("possible_contradiction_in_values")

    # Weighted score (values matter more than model names)
    score = 0.75 * token_hit + 0.25 * model_hit
    if contradiction:
        score = max(0.0, score - 0.35)

    if score >= 0.8 and not contradiction:
        status = "OK "
    elif score >= 0.3:
        status = "PART"
    else:
        status = "MISS"

    return status, score, reasons


# ----------------------------
# Benchmark runner
# ----------------------------

def run_benchmark() -> None:
    items = load_items()

    results: list[dict] = []
    ok_cnt = 0
    part_cnt = 0

    total_start = time.perf_counter()

    for item in items:
        start = time.perf_counter()
        state = run_agent(item.question)
        elapsed = time.perf_counter() - start

        ans = (state.get("answer") or "").strip()

        # If the system explicitly says it doesn't know, that is a MISS
        if not ans or "The manual does not provide this information." in ans:
            status, score, reasons = "MISS", 0.0, ["empty_or_no_info"]
        else:
            status, score, reasons = grade_answer(item.expected, ans)

        if status == "OK ":
            ok_cnt += 1
        elif status == "PART":
            part_cnt += 1

        results.append(
            {
                "question": item.question,
                "expected": item.expected,
                "answer": ans,
                "status": status,
                "score": score,
                "reasons": reasons,
                "time": elapsed,
            }
        )

    total_elapsed = time.perf_counter() - total_start
    total = len(results)

    avg_time = sum(r["time"] for r in results) / total
    max_time = max(r["time"] for r in results)

    print("\n=== ME Engineering Assistant: Benchmark (with expected grading) ===")
    for idx, r in enumerate(results, start=1):
        ans_snippet = r["answer"].replace("\n", " ")
        if len(ans_snippet) > 260:
            ans_snippet = ans_snippet[:257] + "..."

        reason_str = ""
        if r["reasons"]:
            reason_str = " | " + ", ".join(r["reasons"])

        print(f"{idx:02d}. [{r['status']}] {r['time']:.2f}s | score={r['score']:.2f}{reason_str}")
        print(f"    Q: {r['question']}")
        print(f"    A: {ans_snippet}")

    print("\nSummary:")
    print(f"- Questions         : {total}")
    print(f"- OK                : {ok_cnt} ({ok_cnt/total:.0%})")
    print(f"- Partial           : {part_cnt} ({part_cnt/total:.0%})")
    print(f"- Miss              : {total - ok_cnt - part_cnt} ({(total-ok_cnt-part_cnt)/total:.0%})")
    print(f"- Avg time / q      : {avg_time:.2f}s")
    print(f"- Max time / q      : {max_time:.2f}s")
    print(f"- Total runtime     : {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    run_benchmark()
