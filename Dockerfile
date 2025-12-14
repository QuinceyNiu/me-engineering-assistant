# ------------------------------------------------------------------------------
# ME Engineering Assistant – Dockerfile
# - Python 3.11
# - Installs the package from pyproject.toml
# - Includes data/ and saved_model/ so the app can run without extra config
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

# System dependencies (minimal but safe for PyTorch / Transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code
COPY pyproject.toml ./
COPY src ./src
COPY data ./data

# Copy the exported MLflow model artifacts into a fixed location.
# NOTE: Before building the image, run:
#   python -m me_engineering_assistant.log_model
# so that ./saved_model exists at the project root.
COPY saved_model ./saved_model

# Install the project as a package
RUN pip install --upgrade pip && \
    pip install .

# Create non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI default port
EXPOSE 8000

# Start the FastAPI server via the package entry point
CMD ["python", "-m", "me_engineering_assistant"]
