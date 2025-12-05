# ------------------------------------------------------------------------------
# ME Engineering Assistant – Dockerfile
# - Python 3.11
# - Installs the package from pyproject.toml
# - Includes data/ and mlruns/ so the MLflow model can be loaded via MODEL_URI
# ------------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

# General Python & HF settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System dependencies (minimal but safe for PyTorch / Transformers / FAISS etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code
COPY pyproject.toml ./
COPY src ./src
COPY data ./data

# Optional: include locally logged MLflow runs so MODEL_URI like
# runs:/<RUN_ID>/me_engineering_assistant_model works inside the container.
# COPY mlruns ./mlruns
COPY saved_model ./saved_model

# Install the project as a package
RUN pip install --upgrade pip && \
    pip install .

# Create non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI default port
EXPOSE 8000

# MODEL_URI should be provided at runtime, e.g.:
#   runs:/<RUN_ID>/me_engineering_assistant_model
ENV MODEL_URI="saved_model/me_engineering_assistant_model"

# Start the FastAPI server via package entrypoint
CMD ["python", "-m", "me_engineering_assistant"]
