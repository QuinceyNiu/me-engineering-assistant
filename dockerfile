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

# ✅ 现在我们拷贝的是 mlruns（而不是 saved_model）
#   注意：构建镜像前，本地要已经执行过：
#     python -m me_engineering_assistant.log_model
COPY mlruns ./mlruns

# Install the project as a package
RUN pip install --upgrade pip && \
    pip install .

# Tell MLflow to look at /app/mlruns inside the container
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"

# Create non-root user for better security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI default port
EXPOSE 8000

# ❌ 不再在这里写死 MODEL_URI
# MODEL_URI 在 docker run 时通过 -e 传入，比如：
# runs:/7812eb1225d64ebeb6e8b71c108a0492/me_engineering_assistant_model

# Start the FastAPI server via package entrypoint
CMD ["python", "-m", "me_engineering_assistant"]
