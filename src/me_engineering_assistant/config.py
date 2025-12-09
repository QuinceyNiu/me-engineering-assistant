from __future__ import annotations

"""
Global configuration constants for the ME Engineering Assistant.

Most values can be overridden via environment variables so that
the behavior can be tuned without changing code.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory of the repository (two levels above this file)
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directory and files
DATA_DIR = BASE_DIR / "data"

ECU_700_PATH = DATA_DIR / "ECU-700_Series_Manual.md"
ECU_800_BASE_PATH = DATA_DIR / "ECU-800_Series_Base.md"
ECU_800_PLUS_PATH = DATA_DIR / "ECU-800_Series_Plus.md"

TEST_QUESTIONS_PATH = DATA_DIR / "test-questions.csv"

# ---------------------------------------------------------------------------
# RAG parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# ---------------------------------------------------------------------------
# Embeddings & LLM
# ---------------------------------------------------------------------------

# Embedding model used for vector search
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Default local LLM for answer generation
DEFAULT_LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Allow overriding the LLM model name via environment variable
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)

# Limit the number of generated tokens to keep latency within the challenge
# expectations while still allowing sufficiently detailed answers.
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "96"))
