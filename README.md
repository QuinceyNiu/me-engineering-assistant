# ME Engineering Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers engineering questions using ECU (Electronic Control Unit) technical manuals.

This project demonstrates:
- Local LLM inference (Phi-3 Mini)
- Online open-source LLM inference (Llama 3.x via HuggingFace API)
- Multi-document RAG
- Query routing across multiple ECU models
- FastAPI RESTful service
- MLflow model packaging & serving
- Dockerized API serving (validated; supports .env configuration)

The system can run fully offline (local LLM) or partially online (remote LLM) based on user configuration.

---

## 🔍 1. Overview

Modern engineering teams often rely on large, unstructured PDF/manual collections.  
**ME Engineering Assistant** transforms ECU manuals into an intelligent question-answering agent powered by:

- Embedding-based retrieval
- Query routing
- Local or remote LLM reasoning
- LangGraph-based orchestration

The system supports the following manuals:

- ECU-700 Series
- ECU-800 Base
- ECU-800 Plus


---

## ✨ 2. Key Features

### ✔ Multi-manual RAG  
Automatically routes each query to the correct ECU manual family.

### ✔ Configurable LLM Backend
Supports two interchangeable inference modes:

| Backend           | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `local` (default) | Runs `microsoft/Phi-3-mini-4k-instruct` locally                   |
| `remote`          | Calls a free HuggingFace-hosted LLM (e.g., Llama-3.2-1B-Instruct) |

Switch by environment variable:
```bash
export LLM_BACKEND=local      # or: remote
```

### ✔ Efficient Vector Retrieval
HuggingFace embeddings + in-memory Chroma.

### ✔ Modular LangGraph Workflow  
Clean separation: routing → retrieval → answer generation.

### ✔ MLflow Model Packaging  
Exports the entire pipeline as a custom pyfunc model with versioning + ```prod``` alias.

### ✔ REST API with FastAPI  
Provides standard ```/predict``` endpoint served from MLflow model.

### ✔ Docker (Validated)
Runs as a self-contained HTTP API with a single command using `--env-file .env`.
No need to manually locate MLflow artifact paths.

---

## 📂 3. Repository Structure

```text
me-engineering-assistant/
│
├── README.md                      # Project documentation
├── pyproject.toml                 # Dependencies & build config
├── dockerfile                     # Docker build instructions (validated; .env-friendly)
├── project_tree.txt               # Auto-generated project structure
├── .env.example                   # Environment template (no secrets)
├── .gitignore                     # Ignore local artifacts & secrets
│
├── data/                          # ECU manuals + test questions
│   ├── ECU-700_Series_Manual.md
│   ├── ECU-800_Series_Base.md
│   ├── ECU-800_Series_Plus.md
│   └── test-questions.csv
│
├── saved_model/                   # Exported MLflow pyfunc artifacts (generated, gitignored)
│   └── ... (latest model artifacts; used by Docker as /app/saved_model)
│
├── mlruns/                        # Local MLflow tracking directory (generated, gitignored)
│   └── ... (local runs / registry metadata; optional for end users)
│
├── src/
│   └── me_engineering_assistant/
│       ├── __init__.py
│       ├── __main__.py            # FastAPI entrypoint (auto-detects MODEL_URI)
│       ├── api.py                 # REST handlers
│       ├── config.py              # Robust data path resolution (local + Docker)
│       ├── data_loader.py
│       ├── graph.py               # LangGraph orchestration (router → RAG)
│       ├── log_model.py           # MLflow logging + export to ./saved_model
│       ├── mlflow_model.py        # MLflow pyfunc interface
│       ├── rag_chain.py           # Retrieval + LLM generation
│       ├── router.py              # Document routing logic
│       ├── sandbox_test.py        # Simple local CLI test
│       └── vectorstore.py         # Embeddings + vectorstore builder
│
└── tests/
    ├── benchmark.py               # Benchmark verification
    └── test_agent_e2e.py          # End-to-end verification
```

---

## 🧠 4. System Architecture

![ME Engineering Assistant Architecture](me_engineering_assistant_architecture.svg)

---

## ⚙️ 5. Installation

### 5.1 Clone the repository

```bash
git clone <repo-url>
cd me-engineering-assistant
```

### 5.2 Create a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install .
```

For development mode:
```bash
pip install -e .
```

---

## 🌐 6. LLM Backend Configuration

### 6.1 Local Phi-3 (default)
```bash
export LLM_BACKEND=local
export LLM_MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
```
### 6.2 Remote open-source LLM (HuggingFace Inference API)

Create `.env` from the template:

    cp .env.example .env

Edit `.env` and set:

    LLM_BACKEND=remote
    REMOTE_LLM_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
    HUGGINGFACEHUB_API_TOKEN=hf_xxx

The project autoloads `.env` via `python-dotenv`.

All entrypoints (CLI, FastAPI, MLflow logging) obey these settings.

---

## 🚀 7. MLflow Model Logging
Before serving the API, you must register the MLflow model.

### 7.1 Set tracking URI
```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
```
### 7.2 Log the model
```bash
python -m me_engineering_assistant.log_model
```
This will:
- Run the complete pipeline (routing → RAG → LLM)
- Log a new MLflow model version
- Update the prod alias

Example output:
```bash
Created version '8' of model 'me-engineering-assistant'
alias = prod
```

Additionally, the script exports the latest model artifacts to:

    ./saved_model/

This fixed path is used by Docker to avoid manual MLflow artifact path lookup.

Optional (Model Registry alias, local usage only):

    models:/me-engineering-assistant@prod

## 🌐 8. Start the FastAPI Server

```bash
python -m me_engineering_assistant
```
The server will start at:
```bash
http://localhost:8000/predict
```

---

## 📡 9. Example API Requests

### 9.1 cURL

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is the maximum operating temperature of the ECU-850b?"]}'
```

### 9.2 Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={"questions": ["What is the maximum operating temperature of the ECU-850b?"]}
)
print(resp.json())
```

### 9.3 Postman

* POST ```http://localhost:8000/predict```
* Body (JSON)
```json
{
  "questions": ["What is the maximum operating temperature of the ECU-850b?"]
}
```

---

## 🧪 10. Testing & Validation Strategy (Local & Remote LLM Backends)

This project uses a two-layer testing strategy to validate functional correctness and real-world performance across both LLM backends:

- Local backend: Phi-3-Mini (offline, deterministic latency, high correctness)
- Remote backend: Llama-3.x via Hugging Face Inference API (online, low latency, rate-limited)

All tests work with either backend and automatically respect the environment variable:
```bash
export LLM_BACKEND=local    # or: remote
```

### 10.1 Functional Testing (Pytest)

The end-to-end pytest (```tests/test_agent_e2e.py```) validates that the agent answers the majority of evaluation questions correctly.

The test performs the following:

- Loads all questions from test-questions.csv
- Runs the full pipeline (routing → retrieval → LLM generation)
- Records per-query latency
- Counts how many answers are non-fallback
- Ensures ≥ 80% accuracy, meeting the challenge requirement

**Run the test**

```bash
pytest -q -s
```

**Behavior by backend**

| Backend                | Typical Accuracy | Latency Pattern                                |
|------------------------|------------------|------------------------------------------------|
| **local (Phi-3-Mini)** | 90–100%          | Higher latency due to local model inference    |
| **remote (Llama-3.x)** | 80–90%           | Much faster (1–3s), occasional API variability |


### 10.2 Performance & Answer Inspection (Benchmark Script)

A dedicated benchmark script (tests/benchmark.py) provides deeper inspection of:

- Raw answers returned by the system
- End-to-end latency per question
- Total runtime and accuracy
- Differences between local and remote backends

**Run the benchmark**

```bash
python -m tests.benchmark
```

**Example (local backend)**
```bash
01. [OK ] 4.82s
    Q: How much RAM does the ECU-850 have?
    A: The ECU-850 has 2 GB of LPDDR4 RAM.
...
Summary:
- Questions         : 10
- Answered          : 10 (100%)
- Avg time / q      : 11.03s
- Max time / q      : 23.77s
```

**Example (remote backend)**
```bash
01. [OK ] 1.13s
    Q: How much RAM does the ECU-850 have?
    A: The ECU-850 has 2 GB of RAM.
...
Summary:
- Questions         : 10
- Answered          : 8 (80%)
- Avg time / q      : 2.14s
- Max time / q      : 9.87s
```
Notes:

- Remote backend is much faster (1–3s)
- Local backend is more consistent, especially for complex comparative questions


### 10.3 Validation Criteria (Backend-Aware)
The agent is considered valid when **either backend** meets:

✔ **Functional correctness**
- ≥ 80% non-fallback answers over the 10-question evaluation set
- No hallucinated information when context is unclear
- Router selects correct document families

✔ **Performance expectations**

| Backend                | Acceptable Latency | Notes                               |
|------------------------|--------------------|-------------------------------------|
| **Local (Phi-3)**      | Avg ≤ 20–30s       | Includes warm-up + local inference  |
| **Remote (Llama-3.x)** | Avg ≤ 3–5s         | Subject to internet/API variability |

✔ **Stability**
- No runtime errors across all evaluation questions
- Behavior must remain deterministic given the same backend

---

## 🐳 11. Containerization (Docker)

A Dockerfile is included to serve the agent as an HTTP API.
The container can serve the agent using either LLM backend:

- Local: Phi-3-mini model loaded via transformers
- Remote: Llama-3.x hosted on HuggingFace Inference API (free tier compatible)

### 11.1 Build image

> Make sure you have already logged a model locally (see section **7. MLflow Model Logging**)
> so that the `mlruns/` directory contains the latest artifacts.

```bash
docker build -t me-assistant .
```

The Dockerfile copies the following into the image:

- src/ → installed as a Python package
- data/ → /app/data
- saved_model/ → /app/saved_model

### 11.2 Prepare environment variables

Copy the template and fill in your HuggingFace token:

    cp .env.example .env

### 11.3 Run container (one-liner)

    docker run --env-file .env -p 8000:8000 me-assistant

Notes:

- `MODEL_URI` is auto-detected inside Docker (`/app/saved_model`).
- ECU manuals are loaded from `/app/data`.
- API endpoint: http://localhost:8000/predict


---

## ⚠️ 12. Limitations

### LLM-related

- Local backend (Phi-3)

    - Highest correctness but slower inference
    - May hallucinate under ambiguous context

- Remote backend (Llama-3.x)

    - Lower latency but dependent on HuggingFace API rate limits
    - Occasional fallback responses when API returns minimal content

### System limitations

- Router is rule-based (no embedding classifier yet
- Vectorstore is rebuilt at runtime (non-persistent)
- No streaming inference
- Remote backend adds external dependency (network + HF API availability)

---

## 🚧 13. Future Work

### MLOps Enhancements

- MLflow-native model serving (mlflow models serve)
- Cloud-ready tracking server (SQLite / Postgres)
- GPU-enabled images for faster local inference

### Agent Improvements

- Embedding-based router
- Confidence scoring + fallback arbitration
- Multi-hop reasoning
- Support additional ECU model families

### Retrieval Performance

- Persistent FAISS / Chroma index
- Chunk-level re-ranking (Cross-Encoder)
- Hybrid sparse + dense retrieval
- Quantized LLMs for faster local inference

### LLM Backend Enhancements

- Intelligent backend selection (dynamic switch local ↔ remote)
- Caching of remote responses
- Automatic degradation policy (handle API rate limits gracefully)

---

## 🏁 14. Challenge Requirements Alignment

| Requirement                 | Status                                    |         |
|-----------------------------|-------------------------------------------|---------|
| Multi-source RAG            | ✔ Implemented                             |         |
| Intelligent routing         | ✔ Router node                             |         |
| LangGraph agent             | ✔ Two-node workflow (router → RAG)        |         |
| MLflow model logging        | ✔ Custom pyfunc, versioned, prod alias    |         |
| REST API                    | ✔ FastAPI `/predict` loading MLflow model |         |
| Dockerization               | ✔ Validated for remote backend            |         |
| Local LLM inference         | ✔ Phi-3-mini (offline)                    |         |
| Online LLM inference        | ✔ Llama-3.x via HuggingFace API (free)    |         |
| Backend configurability     | ✔ `LLM_BACKEND=local                      | remote` |
| Architectural documentation | ✔ Included                                |         |
| Testing strategy            | ✔ Local + Remote benchmarks & pytest      |         |
| Limitations & future work   | ✔ Documented                              |         |


---

## 🙌 Acknowledgements

- Microsoft Phi-3
- Meta Llama 3.x
- HuggingFace Inference API
- LangChain / LangGraph
- MLflow
- ChromaDB