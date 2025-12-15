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
    MAX_CONTEXT_DOCS,
    MAX_CONTEXT_CHARS_TOTAL,
    MAX_CONTEXT_CHARS_PER_DOC,
    TORCH_NUM_THREADS,
)

# ---------------------------------------------------------------------------
# Lazy-loaded LLM backends (process-wide singletons)
# ---------------------------------------------------------------------------

_DEVICE: Optional[str] = None
_DTYPE: Optional[torch.dtype] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None
_REMOTE_CLIENT: Optional[InferenceClient] = None

FALLBACK_ANSWER = "The manual does not provide this information."


def _maybe_set_num_threads() -> None:
    """Optionally cap PyTorch CPU threads for more predictable latency."""
    if TORCH_NUM_THREADS:
        try:
            torch.set_num_threads(int(TORCH_NUM_THREADS))
        except (ValueError, RuntimeError):
            # Ignore invalid values; default behavior is fine.
            return


# ---------------------------------------------------------------------------
# Device / model selection helpers
# ---------------------------------------------------------------------------


def _select_device_and_dtype() -> tuple[str, torch.dtype]:
    """
    Select the best available device and an appropriate dtype.

    - Prefer Apple Silicon (MPS) when available
    - Otherwise try CUDA
    - Fallback to CPU

    Dtype choices (pragmatic defaults):
    - MPS:  FP16 is usually faster/more stable than BF16.
    - CUDA: BF16 is often a good balance if supported; fall back to FP16.
    - CPU:  FP32 for correctness and broad compatibility.
    """
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        # BF16 is typically good on modern NVIDIA GPUs, otherwise FP16.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    return device, dtype


def _ensure_model_loaded() -> None:
    """
    Lazily load the tokenizer and model on first use (local backend only).

    The model is kept in process-wide globals so repeated calls reuse the same instance.
    """
    global _DEVICE, _DTYPE, _TOKENIZER, _MODEL

    if _MODEL is not None and _TOKENIZER is not None:
        return

    _maybe_set_num_threads()

    device, dtype = _select_device_and_dtype()
    _DEVICE, _DTYPE = device, dtype

    print(
        f"[RAG] Loading local LLM '{LLM_MODEL_NAME}' "
        f"on device: {device}, dtype: {dtype}"
    )

    # NOTE:
    # - Use torch_dtype (transformers arg name) instead of dtype.
    # - low_cpu_mem_usage helps reduce peak RAM on load.
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)

    # Some transformer versions accept attn_implementation; keep it optional.
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    try:
        model_kwargs["attn_implementation"] = "sdpa"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **model_kwargs)
    model.to(device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model


def _ensure_local_model_loaded() -> None:
    """Backward-compatible alias."""
    _ensure_model_loaded()


def _ensure_remote_client_loaded() -> None:
    """
    Lazily create a Hugging Face InferenceClient for the remote backend.

    The user must provide an API token via environment variable.
    """
    global _REMOTE_CLIENT

    if _REMOTE_CLIENT is not None:
        return

    # Try the configured env var first, then fall back to HF_TOKEN for convenience.
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
        Build a plain-text prompt suitable for Hugging Face text-generation endpoints.
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


def _generate_llm_answer(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Invoke the selected LLM backend to generate the answer.

    Steps:
      1) Call backend to get the raw generated text.
      2) Apply post-processing to extract a concise, answer-like sentence.
    """
    if LLM_BACKEND == "remote":
        _ensure_remote_client_loaded()
        assert _REMOTE_CLIENT is not None
        client = _REMOTE_CLIENT

        try:
            full_text = client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
            full_text = full_text.strip() if isinstance(full_text, str) else str(full_text).strip()
        except ValueError as exc:
            # Some providers expose only the "conversational" task.
            msg = str(exc).lower()
            if "conversational" not in msg:
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

            choice = chat_response.choices[0]
            message = getattr(choice, "message", None)
            if isinstance(message, dict):
                full_text = str(message.get("content", "")).strip()
            else:
                content = getattr(message, "content", "")
                full_text = str(content).strip()

    else:
        _ensure_local_model_loaded()
        assert _TOKENIZER is not None
        assert _MODEL is not None
        assert _DEVICE is not None

        tokenizer = _TOKENIZER
        model = _MODEL
        device = _DEVICE

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Ensure generation does not warn about missing pad token.
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return _postprocess_full_text(full_text)


# ---------------------------------------------------------------------------
# Public RAG API
# ---------------------------------------------------------------------------


def _compact_context(docs: List[object]) -> str:
    """
    Build a compact context string with hard caps to control prompt length.

    This improves local LLM latency and reduces the chance of "hanging"
    due to extremely long prompts.
    """
    context_parts: List[str] = []
    seen = set()
    total_chars = 0

    for d in docs[:MAX_CONTEXT_DOCS]:
        text = getattr(d, "page_content", "") or ""
        if not text or text in seen:
            continue
        seen.add(text)

        # Per-doc cap
        text = text[:MAX_CONTEXT_CHARS_PER_DOC]

        # Total cap
        remaining = MAX_CONTEXT_CHARS_TOTAL - total_chars
        if remaining <= 0:
            break
        text = text[:remaining]

        context_parts.append(text)
        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts)


def rag_answer(question: str, vs_dict: Dict[str, object], routes: List[str]) -> str:
    """
    Perform retrieval-augmented generation:

    - Retrieve similar fragments from the selected vector libraries.
    - Concatenate the context while avoiding duplicates and overly long prompts.
    - Invoke the selected LLM backend (local or remote) to generate the final answer.
    """
    docs: List[object] = []
    for route in routes:
        vs = vs_dict[route]
        docs.extend(vs.similarity_search(question, k=TOP_K))

    if not docs:
        return FALLBACK_ANSWER

    context = _compact_context(docs)
    prompt = _build_prompt(question, context)
    answer = _generate_llm_answer(prompt)

    return answer or FALLBACK_ANSWER
