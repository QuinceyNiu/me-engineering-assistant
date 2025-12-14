"""Retrieval-Augmented Generation (RAG) utilities for answering ECU manual questions.

Builds and executes the retrieval + prompting + LLM generation flow, reusing cached components
to keep latency low during repeated requests.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

from .config import (
    TOP_K,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    LLM_BACKEND,
    HF_TOKEN_ENV_VAR,
    REMOTE_LLM_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Lazy-loaded local LLM
# ---------------------------------------------------------------------------

_DEVICE: Optional[str] = None
_DTYPE: Optional[torch.dtype] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None
_REMOTE_CLIENT: Optional[InferenceClient] = None

FALLBACK_ANSWER = "The manual does not provide this information."


# ---------------------------------------------------------------------------
# Device / model selection helpers
# ---------------------------------------------------------------------------


def _select_device_and_dtype() -> tuple[str, torch.dtype]:
    """
    Select the best available device and an appropriate dtype.

    - Prefer Apple Silicon (MPS) when available
    - Otherwise try CUDA
    - Fallback to CPU
    """
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
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

    This is only used when LLM_BACKEND == "local".
    The model is kept in process-wide globals so that repeated calls
    reuse the same instance.
    """
    global _DEVICE, _DTYPE, _TOKENIZER, _MODEL

    if _MODEL is not None and _TOKENIZER is not None:
        return

    device, dtype = _select_device_and_dtype()
    _DEVICE, _DTYPE = device, dtype

    print(
        f"[RAG] Loading local LLM '{LLM_MODEL_NAME}' "
        f"on device: {device}, dtype: {dtype}"
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model


def _ensure_local_model_loaded() -> None:
    """
    Backward-compatible alias.

    Some parts of the code call `_ensure_local_model_loaded()`. The actual
    implementation lives in `_ensure_model_loaded()`.
    """
    _ensure_model_loaded()


def _ensure_remote_client_loaded() -> None:
    """
    Lazily create a Hugging Face InferenceClient for the remote backend.

    This uses the free Hugging Face Inference API (within its rate limits).
    The user must provide an API token via environment variable.
    """
    global _REMOTE_CLIENT

    if _REMOTE_CLIENT is not None:
        return

    # Try the configured env var first, then fall back to HF_TOKEN for
    # convenience if users already have that set.
    token = os.getenv(HF_TOKEN_ENV_VAR) or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"Hugging Face API token not found. "
            f"Please set '{HF_TOKEN_ENV_VAR}' or 'HF_TOKEN' "
            f"when using LLM_BACKEND='remote'."
        )

    print(
        "[RAG] Using REMOTE LLM backend via Hugging Face Inference API.\n"
        f" Model: {REMOTE_LLM_MODEL_NAME}"
    )

    _REMOTE_CLIENT = InferenceClient(
        model=REMOTE_LLM_MODEL_NAME,
        token=token,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(question: str, context: str) -> str:
    """
    Build a prompt for the selected backend.

    - LOCAL:
        Use tokenizer chat template (when available) for best behavior.
        This requires the local tokenizer/model to be loaded.
    - REMOTE:
        Do NOT load local models. Build a plain-text prompt suitable for
        Hugging Face text-generation endpoints.
    """
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

    # Remote backend: never touch local model/tokenizer.
    if LLM_BACKEND == "remote":
        return system_msg + "\n\n" + user_msg + "\n\nAnswer:"

    # Local backend: load model/tokenizer lazily and prefer chat template.
    _ensure_model_loaded()
    assert _TOKENIZER is not None  # for type checkers
    tokenizer = _TOKENIZER

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # If chat template is not available, fall back to plain prompt.
        return system_msg + "\n\n" + user_msg + "\n\nAnswer:"


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _postprocess_full_text(full_text: str) -> str:
    if not full_text:
        return FALLBACK_ANSWER

    # 1) Remove the “prompt/assistant” prefix
    lowered = full_text.lower()
    for marker in ("assistant:", "assistant :"):
        idx = lowered.rfind(marker)
        if idx != -1:
            full_text = full_text[idx + len(marker):].strip()
            break

    # 2) If fallback is included but other content is present: remove fallback
    if FALLBACK_ANSWER in full_text:
        cleaned = full_text.replace(FALLBACK_ANSWER, "").strip()
        if cleaned:
            full_text = cleaned
        else:
            return FALLBACK_ANSWER

    # 3) Prioritize retaining code blocks (such as Q10)
    m = re.search(r"```(?:\w+)?\n.*?\n```", full_text, flags=re.S)
    if m:
        return m.group(0).strip()

    # 4) Prioritize preserving Markdown tables (such as Q8)
    if "|" in full_text and "\n" in full_text:
        return full_text.strip()

    # 5) Fall back to the “select one sentence” strategy;
    # however, if there is no punctuation, return the first non-empty line of text.
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        with_digits = [s for s in sentences if re.search(r"\d", s)]
        return (with_digits[-1] if with_digits else sentences[0]).strip()

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    return (lines[0] if lines else FALLBACK_ANSWER)


# ---------------------------------------------------------------------------
# LLM invocation (local + remote)
# ---------------------------------------------------------------------------


def _generate_llm_answer(
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Invoke the local LLM to generate the answer.

    The function:
      1) Calls the appropriate backend to get the raw generated text.
      2) Applies post-processing to extract a concise, answer-like sentence.
    """
    # ------------------------------------------------------------------
    # 1) Call the appropriate backend to get the raw generated text
    # ------------------------------------------------------------------
    if LLM_BACKEND == "remote":
        # Online open-source model via Hugging Face Inference API.
        _ensure_remote_client_loaded()
        assert _REMOTE_CLIENT is not None
        client = _REMOTE_CLIENT

        try:
            # First try the plain text-generation endpoint.
            full_text = client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
            if isinstance(full_text, str):
                full_text = full_text.strip()
            else:
                full_text = str(full_text).strip()
        except ValueError as exc:
            # Some providers (e.g., certain Llama 3.x instruct models) only
            # expose the "conversational" task. In that case, fall back to the
            # chat_completion API, which uses an OpenAI-style chat format.
            msg = str(exc).lower()
            if "conversational" not in msg:
                # If the error is unrelated, surface it to the caller.
                raise

            system_prompt = (
                'You are the "ME Engineering Assistant", an ECU technical expert. '
                "Answer strictly based on the provided ECU manual context. "
                'If the answer is not present in the context, reply exactly: '
                '"The manual does not provide this information."'
            )

            chat_response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )

            # InferenceClient returns an object with an OpenAI-like structure.
            choice = chat_response.choices[0]
            message = getattr(choice, "message", None)
            if isinstance(message, dict):
                full_text = str(message.get("content", "")).strip()
            else:
                # Fallback for the dataclass-style message object.
                content = getattr(message, "content", "")
                full_text = str(content).strip()

    else:
        # Local Phi-3 (or any local model configured via LLM_MODEL_NAME).
        _ensure_local_model_loaded()
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

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 2) Post-process: extract the best candidate sentence as final answer
    # ------------------------------------------------------------------
    return _postprocess_full_text(full_text)


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

    - Retrieve similar fragments from the selected vector libraries.
    - Concatenate the context while avoiding duplicates.
    - Invoke the selected LLM backend (local or remote) to generate the
      final answer.
    """
    docs = []
    for route in routes:
        vs = vs_dict[route]
        docs.extend(vs.similarity_search(question, k=TOP_K))

    if not docs:
        return FALLBACK_ANSWER

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
        return FALLBACK_ANSWER

    return answer
