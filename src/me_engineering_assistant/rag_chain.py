# src/me_engineering_assistant/rag_chain.py

from __future__ import annotations

from typing import Dict, List, Optional
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

from .config import (
    TOP_K,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    LLM_BACKEND,
    REMOTE_LLM_MODEL_NAME,
    HF_TOKEN_ENV_VAR,
)

# ---------------------------------------------------------------------------
# Global state for local and remote backends
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

    - Prefer Apple Silicon (MPS) when available.
    - Otherwise use CUDA if available.
    - Fallback to CPU.
    """
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # BF16 works well on MPS/CUDA; fall back to FP32 on CPU.
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    return device, dtype


def _ensure_local_model_loaded() -> None:
    """
    Lazily load the local Phi-3 (or any configured) model.

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

    For the LOCAL backend:
        - Use the tokenizer's chat template (when available).
        - This mirrors the original, more reliable Phi-3 usage pattern.

    For the REMOTE backend:
        - Use a plain-text instruction that can be sent directly to
          text-generation endpoints.
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

    # Local backend: use chat template when available for best behavior.
    if LLM_BACKEND == "local":
        _ensure_local_model_loaded()
        assert _TOKENIZER is not None
        tokenizer = _TOKENIZER

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            # Many instruct models (including Phi-3) define a chat template.
            # This usually yields much more stable, well-structured outputs.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # If the tokenizer does not support chat templates,
            # fall back to the plain-text prompt below.
            pass

    # Remote backend (or local fallback): plain-text prompt.
    prompt = system_msg + "\n\n" + user_msg + "\n\nAnswer:"
    return prompt


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _postprocess_full_text(full_text: str) -> str:
    """
    Extract the most answer-like sentence from the raw model output.

    The goal is to keep answers short, factual, and evaluation-friendly:
      - Prefer sentences that contain digits (often specifications).
      - Avoid echoing the question itself.
      - Skip obvious prompt boilerplate when possible.
    """
    if not full_text:
        return FALLBACK_ANSWER

    # If the model used a chat template, it may prefix with "Assistant:".
    lowered = full_text.lower()
    for marker in ("assistant:", "assistant :"):
        idx = lowered.rfind(marker)
        if idx != -1:
            full_text = full_text[idx + len(marker) :].strip()
            break

    # Split into sentences using ., ?, ! while preserving delimiters.
    parts = re.split(r"([\.?!])", full_text)
    sentences: List[str] = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        return FALLBACK_ANSWER

    # Prefer sentences that contain numbers (typically specs or limits).
    candidate_sentences = [s for s in sentences if re.search(r"\d", s)]

    if not candidate_sentences:
        # Fallback: take the last sentence that does not look like prompt
        # boilerplate (e.g., system instructions).
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

    # From candidate sentences, avoid returning the question itself.
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

    return answer if answer else FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# LLM invocation (local + remote)
# ---------------------------------------------------------------------------


def _generate_llm_answer(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Invoke the selected LLM backend (local or remote) to generate the answer.

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

        full_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

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
    # overly long prompts.
    context_parts: List[str] = []
    seen = set()
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            context_parts.append(d.page_content)

    # For these small manuals, 6–8 chunks are usually enough.
    context = "\n\n---\n\n".join(context_parts[:8])

    prompt = _build_prompt(question, context)
    answer = _generate_llm_answer(prompt)

    if not answer:
        return FALLBACK_ANSWER

    return answer
