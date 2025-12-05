# src/me_engineering_assistant/rag_chain.py

from typing import Dict, List
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import TOP_K

# 使用一个适合问答的指令模型，体积不大，M1 Pro 能跑得动
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# 选择设备：优先用 M1 的 mps
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# mps 上用 bfloat16 更稳，cpu 用 float32
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
    对于支持 chat template 的模型（如 Phi-3-mini-instruct），
    优先用 apply_chat_template 构造 prompt。
    如果失败，就退回到手写文本模板。
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

    # 优先走 chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        # 兜底：退回拼接式 prompt
        return (
            system_msg
            + "\n\n"
            + user_msg
            + "\n\nAnswer:"
        )


def _generate_llm_answer(prompt: str, max_new_tokens: int = 256) -> str:
    """调用本地 LLM 生成答案，从完整输出中抽取“最像答案的一句”返回。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 如果有 assistant 标记，优先只取 assistant 之后的部分
    for key in ["<assistant>", "Assistant:", "assistant:"]:
        if key in full_text:
            full_text = full_text.split(key, 1)[-1].strip()
            break

    # 按句子切分（简单按 . ? ! 分）
    # 保留分隔符，方便重新拼回完整句子
    parts = re.split(r'([\.?!])', full_text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i + 1]).strip()
        if sent:
            sentences.append(sent)

    if not sentences:
        return "The manual does not provide this information."

    # 先找“同时包含数字”的句子（很可能是规格或者答案）
    candidate_sentences = [s for s in sentences if re.search(r'\d', s)]
    if not candidate_sentences:
        # 兜底：找不含明显 prompt 关键词的最后一句
        for s in reversed(sentences):
            low = s.lower()
            if any(x in low for x in ["context:", "question:", "you are the", "answer in concise"]):
                continue
            return s.strip()
        # 实在不行就最后一句
        return sentences[-1].strip()

    # 再从这些候选句子里，尽量排除“问题本身”
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
    进行检索增强生成：
    - 根据 routes 从选定向量库取相似片段
    - 拼接 context
    - 调用本地 LLM 生成答案
    """
    docs = []
    for route in routes:
        vs = vs_dict[route]
        docs.extend(vs.similarity_search(question, k=TOP_K))

    if not docs:
        return "The manual does not provide this information."

    # 拼接上下文，避免太长
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
