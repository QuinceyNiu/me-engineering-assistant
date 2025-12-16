"""Vector store construction and access for ECU manuals.

Creates/loads per-manual vector indexes (embeddings + persistent storage) and exposes retrievers
used by the agent to fetch the most relevant context for a question.

Performance notes:
- Vector indexes are persisted on disk (Chroma persist_directory).
- Embeddings and vectorstores are cached as process-wide singletons.
  This avoids rebuilding indexes and reloading embedding models per request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import threading

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
    CHROMA_PERSIST_DIR,
    RAG_REBUILD_INDEX,
)
from .data_loader import load_markdown, make_chunks

# ---------------------------------------------------------------------------
# Process-wide caches
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHED_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_CACHED_VECTORSTORES: Optional[Dict[str, Chroma]] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create (or reuse) the embedding model instance."""
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _CACHED_EMBEDDINGS


def _collection_dir(base_dir: Path, collection_name: str) -> Path:
    """Return the on-disk directory for a specific Chroma collection."""
    # Keep directories stable and readable.
    safe_name = collection_name.replace("/", "_").replace(" ", "_")
    return base_dir / safe_name


def _has_persisted_collection(dir_path: Path) -> bool:
    """Best-effort check: consider it persisted if the directory has any files."""
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    try:
        return any(dir_path.iterdir())
    except OSError:
        return False


def _load_or_build_collection(
    docs: List[dict],
    collection_name: str,
    embeddings: HuggingFaceEmbeddings,
    persist_base: Path,
    force_rebuild: bool,
) -> Chroma:
    """
    Load a persisted Chroma collection if it exists; otherwise build it and persist.

    Why this matters:
    - Building the index requires embedding all chunks, which is slow.
    - Loading a persisted collection is fast and keeps per-query latency low.
    """
    persist_dir = _collection_dir(persist_base, collection_name)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if (not force_rebuild) and _has_persisted_collection(persist_dir):
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    # Rebuild: create an empty collection and add documents.
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    lc_docs = [
        Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for d in docs
    ]
    vs.add_documents(lc_docs)
    vs.persist()
    return vs


def build_vectorstores(force_rebuild: bool = False) -> Dict[str, Chroma]:
    """
    Construct (or load) three vector libraries corresponding to:
    - ECU-700
    - ECU-800-base
    - ECU-800-plus

    Return a dictionary where keys are route names.
    """
    embeddings = _get_embeddings()
    persist_base = CHROMA_PERSIST_DIR
    persist_base.mkdir(parents=True, exist_ok=True)

    # When chunking parameters change, users should rebuild the index.
    # You can set RAG_REBUILD_INDEX=1 or pass force_rebuild=True.
    force = force_rebuild or RAG_REBUILD_INDEX

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

    vs700 = _load_or_build_collection(ecu700_chunks, "ECU-700", embeddings, persist_base, force)
    vs800_base = _load_or_build_collection(ecu800_base_chunks, "ECU-800-base", embeddings, persist_base, force)
    vs800_plus = _load_or_build_collection(ecu800_plus_chunks, "ECU-800-plus", embeddings, persist_base, force)

    return {
        "ECU-700": vs700,
        "ECU-800-base": vs800_base,
        "ECU-800-plus": vs800_plus,
    }


def get_vectorstores(force_rebuild: bool = False) -> Dict[str, Chroma]:
    """
    Return cached vectorstores (build once per process).

    This should be used by the agent/graph so that requests do not pay the
    cost of building/loading vectorstores multiple times.
    """
    global _CACHED_VECTORSTORES

    if _CACHED_VECTORSTORES is not None and not force_rebuild:
        return _CACHED_VECTORSTORES

    with _CACHE_LOCK:
        if _CACHED_VECTORSTORES is not None and not force_rebuild:
            return _CACHED_VECTORSTORES
        _CACHED_VECTORSTORES = build_vectorstores(force_rebuild=force_rebuild)
        return _CACHED_VECTORSTORES
