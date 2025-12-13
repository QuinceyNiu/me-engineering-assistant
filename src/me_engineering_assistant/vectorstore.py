"""Vector store construction and access for ECU manuals.

Creates/loads per-manual vector indexes (embeddings + persistent storage) and exposes retrievers
used by the agent to fetch the most relevant context for a question.
"""

from typing import Dict, List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import (
    ECU_700_PATH,
    ECU_800_BASE_PATH,
    ECU_800_PLUS_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .data_loader import load_markdown, make_chunks


def _build_collection(
    docs: List[dict],
    embeddings,
    collection_name: str,
) -> Chroma:
    """Build a dedicated Chroma vector library for one ECU family."""
    return Chroma.from_documents(
        [
            Document(
                page_content=d["page_content"],
                metadata=d.get("metadata", {}),
            )
            for d in docs
        ],
        embedding=embeddings,
        collection_name=collection_name,
    )



def build_vectorstores() -> Dict[str, Chroma]:
    """
    Construct three vector libraries corresponding to:
    - ECU-700
    - ECU-800-base
    - ECU-800-plus
    Return a dictionary where keys are route names.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    ecu700_text = load_markdown(ECU_700_PATH)
    ecu800_base_text = load_markdown(ECU_800_BASE_PATH)
    ecu800_plus_text = load_markdown(ECU_800_PLUS_PATH)

    ecu700_chunks = make_chunks(ecu700_text, "ECU-700", CHUNK_SIZE, CHUNK_OVERLAP)
    ecu800_base_chunks = make_chunks(
        ecu800_base_text, "ECU-800-base", CHUNK_SIZE, CHUNK_OVERLAP
    )
    ecu800_plus_chunks = make_chunks(
        ecu800_plus_text, "ECU-800-plus", CHUNK_SIZE, CHUNK_OVERLAP
    )

    vs700 = _build_collection(ecu700_chunks, embeddings, "ECU-700")
    vs800_base = _build_collection(ecu800_base_chunks, embeddings, "ECU-800-base")
    vs800_plus = _build_collection(ecu800_plus_chunks, embeddings, "ECU-800-plus")

    return {
        "ECU-700": vs700,
        "ECU-800-base": vs800_base,
        "ECU-800-plus": vs800_plus,
    }
