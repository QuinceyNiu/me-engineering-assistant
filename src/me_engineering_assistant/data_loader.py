from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown(path: Path) -> str:
    """读取 markdown 文本内容。"""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def make_chunks(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    """
    将长文档切分为小 chunk，附带来源元数据。

    返回格式：
    [
        {
            "page_content": "...",
            "metadata": {"source": "ECU-700", "chunk_index": 0}
        },
        ...
    ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [
        {
            "page_content": chunk,
            "metadata": {"source": source, "chunk_index": i},
        }
        for i, chunk in enumerate(chunks)
    ]
