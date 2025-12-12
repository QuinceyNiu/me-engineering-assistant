"""
ME Engineering Assistant package.

Provides:
- Config & data loading
- Vectorstore construction
- Router + RAG pipeline
- LangGraph agent
- MLflow pyfunc model wrapper
- FastAPI REST API
"""

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
# This is useful for local development and for Docker runs that pass an .env
# file via `--env-file`.
load_dotenv()

__all__ = [
    "config",
    "data_loader",
    "vectorstore",
    "router",
    "rag_chain",
    "graph",
    "mlflow_model",
    "api",
]
