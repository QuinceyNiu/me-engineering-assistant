# src/me_engineering_assistant/rag_chain.py
from __future__ import annotations

from typing import Dict, List, Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TOP_K, LLM_MODEL_NAME, MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Lazy-loaded local LLM
# ---------------------------------------------------------------------------

_DEVICE: Optional[str] = None
_DTYPE: Optional[torch.dtype] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None


def _select_device_and_dtype() -> tuple[str, torch.dtype]:
    """
    Select the best available device and an appropriate dtype.

    - Prefer Apple Silicon (MPS) when available
    - Otherwise try CUDA
    - Fallback to CPU
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # BF16 works well on MPS/CUDA; fall back to FP32 on CPU
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    return device, dtype


def _ensure_model_loaded() -> None:
    """
    Lazily load the tokenizer and model on first use.

    This avoids heavy initialization at module import time and is friendlier
    for environments where the module may be imported without ever running
    inference (e.g., static analysis, certain test setups).
    """
    global _DEVICE, _DTYPE, _TOKENIZER, _MODEL

    if _MODEL is not None and _TOKENIZER is not None:
        return

    device, dtype = _select_device_and_dtype()
    _DEVICE, _DTYPE = device, dtype

    print(f"[RAG] Loading local LLM '{LLM_MODEL_NAME}' on device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=dtype)

    model.to(device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model


# ---------------------------------------------------------------------------
# Prompt construction and answer generation
# ---------------------------------------------------------------------------


def _build_prompt(question: str, context: str) -> str:
    """
    Build the prompt for the local LLM.

    For models that support chat templates (such as Phi-3-mini-instruct),
    prioritize constructing prompts using `apply_chat_template`. If that fails,
    fall back to a simple manually written text template.
    """
    _ensure_model_loaded()
    assert _TOKENIZER is not None  # for type checkers

    tokenizer = _TOKENIZER

    system_msg = (
        'You are the "ME Engineering Assistant", an ECU technical expert. '
        "Answer strictly based on the provided ECU manual context. "
        'If the answer is not present in the context, reply exactly: '
        '"The manual does not provide this information."'
    )
    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer in concise, professional English for an engineer."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Prefer a chat template when available
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:  # pylint: disable=broad-exception-caught
        # Some model/tokenizer combinations may not support chat templates.
        # In that case, fall back to a simple, explicit text prompt.
        return system_msg + "\n\n" + user_msg + "\n\nAnswer:"


def _generate_llm_answer(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Invoke the local LLM to generate the answer.

    From the full output text, extract the most answer-like sentence and
    return it. This keeps the response concise and easier to evaluate
    automatically.
    """
    _ensure_model_loaded()
    assert _TOKENIZER is not None
    assert _MODEL is not None
    assert _DEVICE is not None

    tokenizer = _TOKENIZER
    model = _MODEL
    device = _DEVICE

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # If an assistant tag is present, prioritize the portion after it.
    for key in ["", "Assistant:", "assistant:"]:
        if key and key in full_text:
            full_text = full_text.split(key, 1)[-1].strip()
            break

    # Split into sentences using ., ?, ! while preserving delimiters.
    parts = re.split(r"([\.?!])", full_text)
    sentences: List[str] = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        return "The manual does not provide this information."

    # Prefer sentences that contain numbers (often specifications / direct facts).
    candidate_sentences = [s for s in sentences if re.search(r"\d", s)]
    if not candidate_sentences:
        # Fallback: take the last sentence that does not look like prompt boilerplate.
        for s in reversed(sentences):
            low = s.lower()
            if any(
                x in low
                for x in [
                    "context:",
                    "question:",
                    "you are the",
                    "answer in concise",
                ]
            ):
                continue
            return s.strip()
        # If everything looks like boilerplate, just return the last sentence.
        return sentences[-1].strip()

    # From candidate sentences, try to eliminate the question itself.
    filtered: List[str] = []
    for s in candidate_sentences:
        low = s.lower()
        if "question" in low:
            continue
        if low.endswith("?"):
            continue
        filtered.append(s)

    if filtered:
        answer = filtered[-1].strip()
    else:
        answer = candidate_sentences[-1].strip()

    return answer if answer else "The manual does not provide this information."


# ---------------------------------------------------------------------------
# Public RAG API
# ---------------------------------------------------------------------------


def rag_answer(
    question: str,
    vs_dict: Dict[str, object],
    routes: List[str],
) -> str:
    """
    Perform retrieval-augmented generation:

    - Retrieve similar fragments from the selected vector libraries
    - Concatenate the context
    - Invoke the local LLM to generate the final answer
    """
    docs = []
    for route in routes:
        vs = vs_dict[route]
        docs.extend(vs.similarity_search(question, k=TOP_K))

    if not docs:
        return "The manual does not provide this information."

    # Concatenate the context while avoiding duplicated fragments and
    # overly long prompts. The current limit is a pragmatic choice based
    # on the small size of the manuals and Phi-3 context length.
    context_parts: List[str] = []
    seen = set()
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            context_parts.append(d.page_content)

    context = "\n\n---\n\n".join(context_parts[:8])

    prompt = _build_prompt(question, context)
    answer = _generate_llm_answer(prompt)

    if not answer:
        return "The manual does not provide this information."

    return answer
