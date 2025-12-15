"""
Standalone benchmark script for the ME Engineering Assistant.

Run:
    python -m tests.benchmark

What this benchmark does:
- Loads questions + expected answers from test-questions.csv
- Runs the agent to get answers
- Grades answers against expected answers with question-aware heuristics
  (so we don't incorrectly require extra details for certain question types)
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

from me_engineering_assistant.config import TEST_QUESTIONS_PATH
from me_engineering_assistant.graph import run_agent


# ----------------------------
# Regex + parsing helpers
# ----------------------------

MODEL_RE = re.compile(r"\bECU-\d{3}[a-z]?\b", re.IGNORECASE)

# Sentence split (good enough for our short expected answers)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Negation words (for "does not support" etc.)
NEGATION_RE = re.compile(r"\b(do not|does not|don't|doesn't|not)\b", re.IGNORECASE)

# Units we want to score (tokens like +85°C, 16GB, 550mA, 5 TOPS, 2 Mbps, 1.5 GHz)
UNIT_PATTERNS: list[re.Pattern] = [
    re.compile(r"[-+]\s?\d+\s?°\s?C", re.IGNORECASE),              # +85°C, -40°C
    re.compile(r"\b\d+(\.\d+)?\s?(GB|MB)\b", re.IGNORECASE),       # 2 GB, 16GB, 2 MB
    re.compile(r"\b\d+(\.\d+)?\s?(mA|A)\b", re.IGNORECASE),        # 550mA, 1.7A
    re.compile(r"\b\d+(\.\d+)?\s?(TOPS)\b", re.IGNORECASE),        # 5 TOPS
    re.compile(r"\b\d+(\.\d+)?\s?(Mbps|Gbps)\b", re.IGNORECASE),   # 1 Mbps
    re.compile(r"\b\d+(\.\d+)?\s?GHz\b", re.IGNORECASE),           # 1.5 GHz
]


def normalize(text: str) -> str:
    """Normalize whitespace for more stable matching."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_models(text: str) -> set[str]:
    """Extract ECU model IDs like ECU-850b."""
    return {m.upper() for m in MODEL_RE.findall(text or "")}


def extract_unit_tokens(text: str) -> set[str]:
    """
    Extract unit tokens and normalize them into compact uppercase forms:
      '+85°C' -> '+85°C'
      '2 GB'  -> '2GB'
      '550mA' -> '550MA'
      '1.7A'  -> '1.7A'
      '1.5 GHz' -> '1.5GHZ'
    """
    tokens: set[str] = set()
    for pat in UNIT_PATTERNS:
        for mo in pat.finditer(text or ""):
            tok = mo.group(0)
            tok = tok.replace(" ", "")
            tok = tok.upper()
            tokens.add(tok)
    return tokens


def extract_models_with_polarity(expected_text: str) -> tuple[set[str], set[str]]:
    """
    Extract (positive_models, negative_models) from expected answer text.

    This is mainly for "support" questions like OTA:
    - Models in a negated sentence ("do not support") are treated as negative.
    - We do NOT require the answer to mention negative models, but if it does,
      we can penalize it as a likely mistake.
    """
    pos: set[str] = set()
    neg: set[str] = set()

    for sent in SENT_SPLIT_RE.split(expected_text or ""):
        sent_l = sent.lower()
        models = extract_models(sent)
        if not models:
            continue

        # Only treat negation specially when the sentence looks like a "support" claim.
        support_context = ("support" in sent_l) or ("ota" in sent_l) or ("over-the-air" in sent_l)
        if support_context and NEGATION_RE.search(sent):
            neg |= models
        else:
            pos |= models

    return pos, neg


def parse_temp_c(token: str) -> float | None:
    """Parse '+85°C' / '-40°C' into float degrees C."""
    m = re.search(r"([-+]?\d+)\s*°\s*C", token.replace(" ", ""), re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_current_ma(token: str) -> float | None:
    """
    Parse '1.7A' -> 1700 mA, '550MA' -> 550 mA.
    Return value in mA.
    """
    tok = token.replace(" ", "").upper()
    m = re.search(r"(\d+(\.\d+)?)(MA|A)\b", tok)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(3)
    return val if unit == "MA" else val * 1000.0


def pick_max_temp_token(tokens: set[str]) -> set[str]:
    """Pick only the maximum temperature token from a token set (if any)."""
    temps = [(t, parse_temp_c(t)) for t in tokens if "°C" in t.upper()]
    temps = [(t, v) for (t, v) in temps if v is not None]
    if not temps:
        return set()
    max_v = max(v for _, v in temps)
    return {t for (t, v) in temps if v == max_v}


def pick_extreme_temp_tokens(tokens: set[str]) -> set[str]:
    """Pick min and max temperature tokens (if any)."""
    temps = [(t, parse_temp_c(t)) for t in tokens if "°C" in t.upper()]
    temps = [(t, v) for (t, v) in temps if v is not None]
    if not temps:
        return set()
    vs = [v for _, v in temps]
    mn, mx = min(vs), max(vs)
    return {t for (t, v) in temps if v in (mn, mx)}


def pick_max_current_token(tokens: set[str]) -> set[str]:
    """Pick only the maximum current token from a token set (if any)."""
    curr = [(t, parse_current_ma(t)) for t in tokens if t.upper().endswith(("MA", "A"))]
    curr = [(t, v) for (t, v) in curr if v is not None]
    if not curr:
        return set()
    max_v = max(v for _, v in curr)
    return {t for (t, v) in curr if v == max_v}


def is_enable_command_question(question: str) -> bool:
    """Detect command-like questions where matching the command is enough."""
    q = (question or "").lower()
    return (("how do you" in q) or ("how to" in q)) and ("enable" in q)


def expected_command_snippet(expected: str) -> str | None:
    """
    Extract the most important command snippet from expected text.
    We intentionally keep this simple and robust.
    """
    exp = expected or ""

    # Prefer content inside a code fence if present.
    m = re.search(r"```(?:bash)?\s*([\s\S]*?)```", exp, re.IGNORECASE)
    if m:
        cmd = normalize(m.group(1))
        return cmd if cmd else None

    # Fallback: find a command-like substring.
    m2 = re.search(r"(me-driver-ctl[^\n`]+)", exp, re.IGNORECASE)
    return normalize(m2.group(1)) if m2 else None


def wants_only_max_temperature(question: str) -> bool:
    q = (question or "").lower()
    return ("maximum" in q) and ("temperature" in q)


def wants_temperature_extremes(question: str) -> bool:
    q = (question or "").lower()
    return (("harshest" in q) or ("widest" in q)) and ("temperature" in q)


def wants_only_load_current(question: str) -> bool:
    q = (question or "").lower()
    return ("under load" in q) or ("load" in q)


def is_differences_question(question: str) -> bool:
    return "differences between" in (question or "").lower()


def allows_any_of_expected_models(question: str) -> bool:
    """
    For some questions, any correct model among the expected set is acceptable.
    Example: "Which ECU can operate in the harshest temperature conditions?"
    If both ECU-850 and ECU-850b qualify, mentioning either is acceptable.
    """
    q = (question or "").lower()
    if ("which ecu" in q) and ("harshest" in q) and ("temperature" in q):
        return True
    return False


def detect_value_contradiction(
    question: str,
    expected_tokens: set[str],
    answer_tokens: set[str],
    temp_mode: str,
    curr_mode: str,
) -> bool:
    """
    Conservative contradiction detection.

    Important: For questions like "maximum temperature" we do NOT treat extra
    non-conflicting temperature tokens as a contradiction (e.g., including -40°C
    in addition to +85°C should be fine).
    """
    # Temperature contradiction
    exp_temps = [parse_temp_c(t) for t in expected_tokens if "°C" in t.upper()]
    ans_temps = [parse_temp_c(t) for t in answer_tokens if "°C" in t.upper()]
    exp_temps = [v for v in exp_temps if v is not None]
    ans_temps = [v for v in ans_temps if v is not None]

    if exp_temps and ans_temps:
        exp_min, exp_max = min(exp_temps), max(exp_temps)
        ans_min, ans_max = min(ans_temps), max(ans_temps)

        if temp_mode == "max":
            # For "maximum temperature" we only care that the max matches.
            if ans_max != exp_max:
                return True
        elif temp_mode == "extremes":
            # For "harshest/widest", any token outside expected range is suspicious.
            if ans_min < exp_min or ans_max > exp_max:
                return True
        else:
            # Generic: any temp token not in expected is suspicious.
            exp_set = {t for t in expected_tokens if "°C" in t.upper()}
            ans_set = {t for t in answer_tokens if "°C" in t.upper()}
            if exp_set and (ans_set - exp_set):
                return True

    # Current contradiction
    exp_curr = [parse_current_ma(t) for t in expected_tokens if t.upper().endswith(("MA", "A"))]
    ans_curr = [parse_current_ma(t) for t in answer_tokens if t.upper().endswith(("MA", "A"))]
    exp_curr = [v for v in exp_curr if v is not None]
    ans_curr = [v for v in ans_curr if v is not None]

    if exp_curr and ans_curr:
        exp_max = max(exp_curr)
        ans_max = max(ans_curr)

        if curr_mode == "max":
            if ans_max != exp_max:
                return True
        else:
            exp_set = {t for t in expected_tokens if t.upper().endswith(("MA", "A"))}
            ans_set = {t for t in answer_tokens if t.upper().endswith(("MA", "A"))}
            if exp_set and (ans_set - exp_set):
                return True

    # Other unit types: if answer includes a unit token type not present in expected, treat as suspicious.
    # (This helps catch cases like "ECU-850 has 32GB" when expected is 16GB.)
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
        if tok.endswith("GHZ"):
            return "FREQ"
        return "OTHER"

    exp_by_type: dict[str, set[str]] = {}
    ans_by_type: dict[str, set[str]] = {}
    for t in expected_tokens:
        exp_by_type.setdefault(unit_type(t), set()).add(t)
    for t in answer_tokens:
        ans_by_type.setdefault(unit_type(t), set()).add(t)

    for t_type, exp_set in exp_by_type.items():
        if t_type in ("TEMP", "CURR", "OTHER"):
            continue  # handled above or irrelevant
        ans_set = ans_by_type.get(t_type, set())
        extra = ans_set - exp_set
        if exp_set and extra:
            return True

    return False


# ----------------------------
# Data model + CSV loading
# ----------------------------

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


# ----------------------------
# Scoring logic
# ----------------------------

def grade_answer(question: str, expected: str, answer: str) -> tuple[str, float, list[str]]:
    """
    Returns: (status, score_0_1, reasons)
    status in {OK, PART, MISS}

    Key idea:
    - Score unit tokens (values) + models
    - Apply question-aware filters so we don't require irrelevant details
    - Detect obvious contradictions
    """
    expected_n = normalize(expected)
    answer_n = normalize(answer)
    qn = normalize(question)

    reasons: list[str] = []

    # Command questions: if the command matches, treat as OK without needing model mentions.
    if is_enable_command_question(qn):
        cmd = expected_command_snippet(expected_n)
        if cmd and cmd.lower() in answer_n.lower():
            return "OK ", 1.0, []
        reasons.append("command_not_matched")

    exp_tokens = extract_unit_tokens(expected_n)
    ans_tokens = extract_unit_tokens(answer_n)

    # Decide per-question "modes" for contradiction logic.
    temp_mode = "all"
    curr_mode = "all"

    # Temperature filters
    if wants_only_max_temperature(qn):
        temp_mode = "max"
        max_tok = pick_max_temp_token(exp_tokens)
        if max_tok:
            # Replace ALL expected temp tokens with the max temp token(s).
            exp_tokens = {t for t in exp_tokens if "°C" not in t.upper()} | max_tok

    if wants_temperature_extremes(qn):
        temp_mode = "extremes"
        extreme = pick_extreme_temp_tokens(exp_tokens)
        if extreme:
            exp_tokens = {t for t in exp_tokens if "°C" not in t.upper()} | extreme

    # Current filters
    if wants_only_load_current(qn):
        curr_mode = "max"
        max_curr = pick_max_current_token(exp_tokens)
        if max_curr:
            exp_tokens = {t for t in exp_tokens if not t.upper().endswith(("MA", "A"))} | max_curr

    # Models with polarity (mainly for support/OTA questions)
    exp_pos_models, exp_neg_models = extract_models_with_polarity(expected_n)
    ans_models = extract_models(answer_n)

    # Penalize mentioning models that are explicitly "not supported" in expected.
    unsupported_mentioned = exp_neg_models & ans_models
    if unsupported_mentioned:
        reasons.append(f"includes_unsupported_models={sorted(unsupported_mentioned)}")

    # Model scoring mode: "all" vs "any"
    require_any_model = allows_any_of_expected_models(qn)

    model_hit = 1.0
    if exp_pos_models:
        if require_any_model:
            model_hit = 1.0 if (exp_pos_models & ans_models) else 0.0
            if model_hit == 0.0:
                reasons.append(f"missing_any_of_models={sorted(exp_pos_models)}")
        else:
            hit = len(exp_pos_models & ans_models)
            model_hit = hit / max(1, len(exp_pos_models))
            if model_hit < 1.0:
                reasons.append(f"missing_models={sorted(exp_pos_models - ans_models)}")

    # Value scoring
    token_hit = 1.0
    if exp_tokens:
        hit = len(exp_tokens & ans_tokens)
        token_hit = hit / max(1, len(exp_tokens))
        if token_hit < 1.0:
            reasons.append(f"missing_values={sorted(exp_tokens - ans_tokens)}")

    contradiction = detect_value_contradiction(
        question=qn,
        expected_tokens=exp_tokens,
        answer_tokens=ans_tokens,
        temp_mode=temp_mode,
        curr_mode=curr_mode,
    )
    if contradiction:
        reasons.append("possible_contradiction_in_values")

    # Weighting: values matter more than model names.
    score = 0.80 * token_hit + 0.20 * model_hit

    # Penalties
    if contradiction:
        score = max(0.0, score - 0.35)
    if unsupported_mentioned:
        score = max(0.0, score - 0.35)

    # Thresholds
    ok_th = 0.80
    part_th = 0.30

    # "Differences between" questions: allow PART more easily if at least something matches.
    if is_differences_question(qn):
        part_th = 0.20

    if score >= ok_th and not contradiction:
        status = "OK "
    elif score >= part_th:
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

        # If the system explicitly says it doesn't know, that is a MISS.
        if not ans or "The manual does not provide this information." in ans:
            status, score, reasons = "MISS", 0.0, ["empty_or_no_info"]
        else:
            status, score, reasons = grade_answer(item.question, item.expected, ans)

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
