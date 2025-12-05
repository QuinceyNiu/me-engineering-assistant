# src/me_engineering_assistant/rag_chain.py

from typing import Dict, List
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import TOP_K

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32

print(f"[RAG] Loading local LLM '{MODEL_NAME}' on device: {DEVICE}, dtype: {DTYPE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
)
model.to(DEVICE)
model.eval()


def _build_prompt(question: str, context: str) -> str:
    """
    For models that support chat templates (such as Phi-3-mini-instruct),
    prioritize constructing prompts using `apply_chat_template`.
    If that fails, fall back to manually written text templates.
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

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Prioritize chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        # Fallback: Return the concatenated prompt
        return (
            system_msg
            + "\n\n"
            + user_msg
            + "\n\nAnswer:"
        )


def _generate_llm_answer(prompt: str, max_new_tokens: int = 256) -> str:
    """Invoke the local LLM to generate the answer,
    extract the “most answer-like sentence” from the full output, and return it."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    #If an assistant tag is present, prioritize extracting only the portion following the assistant tag.
    for key in ["<assistant>", "Assistant:", "assistant:"]:
        if key in full_text:
            full_text = full_text.split(key, 1)[-1].strip()
            break

    # Split by Sentences (Simply by . ? !)
    # Preserve delimiters for easy reassembly into complete sentences
    parts = re.split(r'([\.?!])', full_text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        return "The manual does not provide this information."

    # First, locate sentences that “contain numbers” (likely specifications or answers).
    candidate_sentences = [s for s in sentences if re.search(r'\d', s)]
    if not candidate_sentences:
        # Fallback: Find the last sentence without obvious prompt keywords
        for s in reversed(sentences):
            low = s.lower()
            if any(x in low for x in ["context:", "question:", "you are the", "answer in concise"]):
                continue
            return s.strip()
        # If all else fails, just go with the last line.
        return sentences[-1].strip()

    # Then, from these candidate sentences, try to eliminate the “problem itself.”
    filtered = []
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


def rag_answer(
    question: str,
    vs_dict: Dict[str, any],
    routes: List[str],
) -> str:
    """
    Perform retrieval-augmented generation:
    - Retrieve similar fragments from the selected vector library based on routes
    - Concatenate context
    - Invoke the local LLM to generate the answer
    """
    docs = []
    for route in routes:
        vs = vs_dict[route]
        docs.extend(vs.similarity_search(question, k=TOP_K))

    if not docs:
        return "The manual does not provide this information."

    # Concatenate the context to avoid excessive length.
    context_parts = []
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
