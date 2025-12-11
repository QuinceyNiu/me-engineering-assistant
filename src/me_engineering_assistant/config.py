from __future__ import annotations

"""
Global configuration constants for the ME Engineering Assistant.

Most values can be overridden via environment variables so that
the behavior can be tuned without changing code.
"""

from pathlib import Path
import os

def _get_data_dir() -> Path:
    """
    Detect the directory that contains the ECU manuals.

    Priority:
    1. Environment variable ME_ASSISTANT_DATA_DIR
    2. Local source tree: <project_root>/data
    3. Docker image default: /app/data
    4. Package-relative: <this_package>/data
    """
    # 1. Env override (for maximum flexibility)
    env_dir = os.getenv("ME_ASSISTANT_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    here = Path(__file__).resolve()

    # 2. Running from source tree (src/me_engineering_assistant/...)
    project_root = here.parents[2]
    src_data = project_root / "data"
    if src_data.exists():
        return src_data

    # 3. Docker image: we COPY data/ into /app/data
    docker_data = Path("/app/data")
    if docker_data.exists():
        return docker_data

    # 4. Package-relative fallback (if you ever package data with the wheel)
    pkg_data = here.parent / "data"
    if pkg_data.exists():
        return pkg_data

    # 5. Last resort: still return src_data (will likely fail, but path is explicit)
    return src_data

DATA_DIR = _get_data_dir()

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


# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------

# LLM_BACKEND controls whether we use a local model or an online model.
# - "local": use the local Phi-3 model via transformers (default)
# - "remote": use a remote open-source model via Hugging Face Inference API
LLM_BACKEND = os.getenv("LLM_BACKEND", "remote").lower()

# Default remote model for the online backend (must be a text-generation model
# available on Hugging Face Hub). You can override this via environment
# variable if you want to experiment with other models.
REMOTE_LLM_MODEL_NAME = os.getenv(
    "REMOTE_LLM_MODEL_NAME",
    "meta-llama/Llama-3.2-1B-Instruct",
)

# Environment variable name that should contain the Hugging Face API token.
# The token is required when using the online backend.
HF_TOKEN_ENV_VAR = "HUGGINGFACEHUB_API_TOKEN"
