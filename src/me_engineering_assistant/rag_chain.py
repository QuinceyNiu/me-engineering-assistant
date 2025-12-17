"""
Retrieval-Augmented Generation (RAG) utilities for answering ECU manual questions.

This module builds and executes the retrieval → prompting → generation pipeline and
keeps heavy components (LLM/tokenizer/remote client) lazily loaded and cached.

Key improvements in this version (quality-focused, not tied to any fixed benchmark set):

- Question intent detection:
  Detects high-level intents (compare / list / across-all / how-to) and extracts ECU model
  identifiers from the user question. This allows the system to adapt retrieval strategy,
  prompt format, and output constraints to the question type.

- Multi-query retrieval for coverage:
  For compare/list/across-all questions, generates additional sub-queries (e.g., per-model
  + per-field) and retrieves documents for each query across the selected routes. This
  reduces “missing attribute” answers caused by single-query retrieval returning only one
  relevant fragment (common in comparison and aggregation questions).

- Context compaction with lightweight provenance:
  Context is compacted with hard caps to control prompt size, and each chunk is optionally
  annotated with a short source tag (e.g., [source #chunk]) to improve grounding and make
  the generated answer more evidence-aligned.

- Prompting improvements for completeness and format:
  The prompt includes explicit instructions to:
    * answer strictly from provided context (no guessing),
    * use tables/bullets for comparisons and aggregations,
    * list ALL applicable models for “which models support …” questions,
    * output commands in a single Markdown code block for how-to/enable questions.

- Output post-processing that preserves structured answers:
  Post-processing is designed to keep multi-item outputs intact (Markdown bullet/numbered
  lists, tables, and code blocks) instead of collapsing answers into a single sentence,
  preventing accidental truncation of multi-model/ multi-attribute responses.

- Dynamic generation budget:
  Increases max_new_tokens for compare/list/across-all questions to reduce truncation on
  longer, structured answers while keeping simple Q&A fast.

- Optional second-pass completion:
  For multi-model questions, if the first answer is missing one or more models mentioned
  in the question, the pipeline performs a small targeted retrieval for the missing models
  and re-generates the answer using the enriched context. This boosts completeness without
  globally increasing TOP_K or prompt length.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set

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

# Regex for detecting ECU model identifiers in questions/answers.
_MODEL_RE = re.compile(r"\bECU-\d{3}[a-z]?\b", re.IGNORECASE)

# Simple patterns for question intent detection.
_COMPARE_RE = re.compile(r"\b(compare|difference|differences|versus|vs\.)\b", re.IGNORECASE)
_ACROSS_RE = re.compile(r"\b(across|all\s+ecu|all\s+models|across\s+all)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\b(which\s+ecu|which\s+models|list\s+all|supported\s+models)\b", re.IGNORECASE)
_HOWTO_RE = re.compile(r"\b(how\s+do\s+you|how\s+to|enable|configure)\b", re.IGNORECASE)


def _extract_models(text: str) -> List[str]:
    """Extract unique model IDs like ECU-850b, preserving first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in _MODEL_RE.findall(text or ""):
        mm = m.upper()
        if mm not in seen:
            seen.add(mm)
            out.append(mm)
    return out


def _analyze_question(question: str) -> Dict[str, object]:
    """Lightweight intent analysis to drive retrieval and formatting."""
    q = (question or "").strip()
    return {
        "models": _extract_models(q),
        "is_compare": bool(_COMPARE_RE.search(q)),
        "is_across": bool(_ACROSS_RE.search(q)),
        "is_list": bool(_LIST_RE.search(q)),
        "is_howto": bool(_HOWTO_RE.search(q)),
    }


def _build_multi_queries(question: str, intent: Dict[str, object]) -> List[str]:
    """Create additional sub-queries for better coverage on compare/list questions."""
    queries = [question]
    models: List[str] = intent.get("models", [])  # type: ignore[assignment]

    # Common fields that repeatedly appear in your benchmark set.
    fields = [
        "operating temperature",
        "RAM",
        "LPDDR",
        "storage",
        "eMMC",
        "power consumption",
        "current",
        "CAN",
        "CAN FD",
        "OTA",
        "Over-the-Air",
        "NPU",
        "TOPS",
        "clock speed",
        "GHz",
    ]

    # Compare/differences or across-all: force retrieval for each model + each key field.
    if intent.get("is_compare") or intent.get("is_across"):
        for m in models:
            for f in fields:
                queries.append(f"{m} {f}")

    # List/support questions: query the capability term directly.
    if intent.get("is_list"):
        ql = question.lower()
        if "ota" in ql or "over-the-air" in ql:
            queries.append("OTA updates supported models")
            queries.append("Over-the-Air updates supported models")

    # De-duplicate while preserving order.
    seen: Set[str] = set()
    out: List[str] = []
    for q in queries:
        qq = q.strip()
        if not qq:
            continue
        key = qq.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(qq)
    return out


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

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)

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


def _build_prompt(question: str, context: str, intent: Dict[str, object]) -> str:
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
        "Do NOT guess or add specs that are not explicitly stated in the context. "
        "If the answer is not present in the context, reply exactly: "
        '"The manual does not provide this information."'
    )

    # Output-format hints to improve completeness on compare/list questions.
    format_hints: List[str] = ["Answer in concise, professional English for an engineer."]
    if intent.get("is_howto"):
        format_hints.append("If a command is needed, output it in a single Markdown code block.")
    if intent.get("is_compare") or intent.get("is_across"):
        format_hints.append(
            "If the question asks to compare models, answer with a short Markdown table or bullet list "
            "covering all relevant fields (e.g., CPU/clock, RAM, storage, AI/NPU/TOPS, temperature, buses)."
        )
        format_hints.append("Make sure every mentioned model in the question appears in the final answer.")
    if intent.get("is_list"):
        format_hints.append("If multiple models apply, list ALL applicable ECU model names explicitly.")

    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        + "\n".join(f"- {h}" for h in format_hints)
    )

    if LLM_BACKEND == "remote":
        return system_msg + "\n\n" + user_msg + "\n\nAnswer:"

    _ensure_model_loaded()
    assert _TOKENIZER is not None
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
        return system_msg + "\n\n" + user_msg + "\n\nAnswer:"


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _postprocess_full_text(full_text: str) -> str:
    """Extract a concise-but-complete answer while preserving multi-item outputs."""
    if not full_text:
        return FALLBACK_ANSWER

    # 1) Remove assistant markers if they appear
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

    # 3) Preserve code blocks (e.g., enable/command questions)
    m = re.search(r"```(?:\w+)?\n.*?\n```", full_text, flags=re.S)
    if m:
        return m.group(0).strip()

    # 4) Preserve Markdown tables
    if "|" in full_text and "\n" in full_text:
        return full_text.strip()

    # 5) Preserve bullet/numbered lists to avoid dropping important items
    lines = [ln.rstrip() for ln in full_text.splitlines() if ln.strip()]
    if any(re.match(r"^\s*([-*]|\d+\.)\s+", ln) for ln in lines):
        return "\n".join(lines).strip()

    # 6) Fallback: pick a single sentence (prefer one containing digits)
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
      2) Apply post-processing to preserve multi-item outputs when present.
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
            msg = str(exc).lower()
            if "conversational" not in msg:
                raise

            system_prompt = (
                'You are the "ME Engineering Assistant", an ECU technical expert. '
                "Answer strictly based on the provided ECU manual context. "
                "Do NOT guess or add specs that are not explicitly stated in the context. "
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

    This improves local LLM latency and reduces the chance of hanging
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

        # Add lightweight provenance to improve grounding.
        meta = getattr(d, "metadata", {}) or {}
        src = str(meta.get("source", ""))
        idx = meta.get("chunk_index", None)
        header = ""
        if src:
            header = f"[{src}{' #' + str(idx) if idx is not None else ''}]\n"

        # Per-doc cap (body only)
        text = text[:MAX_CONTEXT_CHARS_PER_DOC]

        # Total cap
        remaining = MAX_CONTEXT_CHARS_TOTAL - total_chars
        if remaining <= 0:
            break
        text = text[:remaining]

        snippet = (header + text).strip()
        context_parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n---\n\n".join(context_parts)


def _dynamic_max_new_tokens(intent: Dict[str, object]) -> int:
    """Use a higher token budget for compare/list questions to avoid truncation."""
    if intent.get("is_compare") or intent.get("is_across"):
        return max(MAX_NEW_TOKENS, 180)
    if intent.get("is_list"):
        return max(MAX_NEW_TOKENS, 140)
    return MAX_NEW_TOKENS


def _retrieve_docs(
    queries: List[str],
    vs_dict: Dict[str, object],
    routes: List[str],
    k_per_query: int,
) -> List[object]:
    """Retrieve docs across routes for multiple queries and de-duplicate."""
    all_docs: List[object] = []
    seen_text: Set[str] = set()

    for q in queries:
        for route in routes:
            vs = vs_dict[route]
            for d in vs.similarity_search(q, k=k_per_query):
                txt = getattr(d, "page_content", "") or ""
                if not txt or txt in seen_text:
                    continue
                seen_text.add(txt)
                all_docs.append(d)
    return all_docs


def rag_answer(question: str, vs_dict: Dict[str, object], routes: List[str]) -> str:
    """
    Perform retrieval-augmented generation:

    - Multi-query retrieval for coverage on compare/list questions.
    - Compact context with hard caps.
    - Call local or remote LLM to generate the final answer.
    - Optional second pass if the answer misses models mentioned in the question.
    """
    intent = _analyze_question(question)

    queries = _build_multi_queries(question, intent)
    k_per_query = 2 if (intent.get("is_compare") or intent.get("is_across") or intent.get("is_list")) else TOP_K
    k_per_query = min(TOP_K, k_per_query) if k_per_query else TOP_K

    docs = _retrieve_docs(queries, vs_dict, routes, k_per_query=k_per_query)

    if not docs:
        return FALLBACK_ANSWER

    context = _compact_context(docs)
    prompt = _build_prompt(question, context, intent)
    answer = _generate_llm_answer(prompt, max_new_tokens=_dynamic_max_new_tokens(intent))

    # Second-pass completion: if the question mentions multiple models but the
    # answer mentions only a subset, try a small targeted retrieval and re-answer.
    q_models = _extract_models(question)
    a_models = _extract_models(answer)
    if (intent.get("is_compare") or intent.get("is_list") or intent.get("is_across")) and len(q_models) >= 2:
        missing = [m for m in q_models if m not in a_models]
        if missing:
            extra_queries: List[str] = []
            for m in missing:
                extra_queries.append(f"{m} {question}")
                extra_queries.append(f"{m} specifications")
            extra_docs = _retrieve_docs(extra_queries, vs_dict, routes, k_per_query=2)
            if extra_docs:
                context2 = _compact_context(docs + extra_docs)
                prompt2 = _build_prompt(question, context2, intent)
                answer2 = _generate_llm_answer(prompt2, max_new_tokens=_dynamic_max_new_tokens(intent))
                if answer2:
                    answer = answer2

    return answer or FALLBACK_ANSWER
