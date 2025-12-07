# ME Engineering Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers engineering questions using ECU (Electronic Control Unit) technical manuals.

This project demonstrates:
- Local LLM inference (Phi-3 Mini)
- Multi-document RAG
- Query routing across multiple ECU models
- FastAPI RESTful service
- MLflow model packaging & serving
- **Dockerfile template (experimental, not fully validated yet)**

The system runs fully **offline**, making it suitable for on-premise or restricted environments.

---

## 🔍 1. Overview

Modern engineering teams often rely on large, unstructured PDF/manual collections.  
**ME Engineering Assistant** turns ECU manuals into an intelligent Q&A assistant powered by:

- Embedding-based retrieval  
- Query routing  
- Lightweight local LLM reasoning  

The system supports the following manuals:

- ECU-700 Series
- ECU-800 Base
- ECU-800 Plus

Routing ensures each query is answered using the most relevant manual family.

---

## ✨ 2. Key Features

### ✔ Multi-manual RAG  
Automatically routes each question to the correct ECU manual family using rule-based classification.

### ✔ Local Phi-3 LLM  
Runs fully offline using `microsoft/Phi-3-mini-4k-instruct` on CPU or Apple Silicon (MPS + BF16).

### ✔ Efficient Vector Search  
Uses HuggingFace embeddings + Chroma vectorstore for high-recall context retrieval.

### ✔ Modular LangGraph Workflow  
Separates concerns: routing → retrieval → answer generation.

### ✔ MLflow Model Packaging  
Exports the entire pipeline as a custom pyfunc model with versioning + prod alias.

### ✔ REST API with FastAPI  
Provides a standard /predict endpoint.

### ⚪ Docker (Experimental)  
A Dockerfile is provided as a **work-in-progress template**.  
It is not yet fully validated end-to-end and may require additional configuration.

---

## 📂 3. Repository Structure

```text
me-engineering-assistant/
│
├── README.md                      # Project documentation
├── pyproject.toml                 # Dependencies & build config
├── dockerfile                     # Docker build instructions (experimental)
├── project_tree.txt               # Auto-generated project structure
│
├── data/                          # ECU manuals + test questions
│   ├── ECU-700_Series_Manual.md
│   ├── ECU-800_Series_Base.md
│   ├── ECU-800_Series_Plus.md
│   └── test-questions.csv
│
├── mlruns/                        # Local MLflow tracking directory
│   └── ... (multiple registered runs/models)
│
├── src/
│   └── me_engineering_assistant/
│       ├── __init__.py
│       ├── __main__.py            # FastAPI entrypoint
│       ├── api.py                 # REST handlers
│       ├── config.py
│       ├── data_loader.py
│       ├── graph.py               # LangGraph orchestration (router → RAG)
│       ├── log_model.py           # MLflow model logging utility
│       ├── mlflow_model.py        # MLflow pyfunc interface
│       ├── rag_chain.py           # Retrieval + LLM generation
│       ├── router.py              # Document routing logic
│       ├── sandbox_test.py        # Simple local CLI test
│       └── vectorstore.py         # Embeddings + vectorstore builder
│
└── tests/
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

## 🚀 6. Running Locally (CLI)

```bash
python -m me_engineering_assistant.sandbox_test
```
Example output:
```vbnet
Question: What is the maximum operating temperature for the ECU-850b?
Answer: The maximum operating temperature for the ECU-850b is +105°C.
```

---

## 🌐 7. Start the FastAPI Server

```bash
python -m me_engineering_assistant
```
The server will start at:
```bash
http://localhost:8000/predict
```

---

## 📡 8. Example API Requests

### 8.1 cURL

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is the maximum operating temperature of the ECU-850b?"]}'
```

### 8.2 Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={"questions": ["What is the maximum operating temperature of the ECU-850b?"]}
)
print(resp.json())
```

### 8.3 Postman

* POST ```http://localhost:8000/predict```
* Body (JSON)
```json
{
  "questions": ["What is the maximum operating temperature of the ECU-850b?"]
}
```

---

## 📦 9. MLflow Model Logging & Loading

### Log the model
```bash
python -m me_engineering_assistant.log_model
```
This will:
- Log the full agent pipeline
- Create a new version in MLflow Model Registry
- Update the prod alias

Example output:
```bash
Created version '7' of model 'me-engineering-assistant'
alias = prod
```

### Recommended MODEL_URI for serving
```bash
models:/me-engineering-assistant@prod
```

### Load the model in Python
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/me-engineering-assistant@prod")
```

### FastAPI with MODEL_URI
```bash
export MODEL_URI="models:/me-engineering-assistant@prod"
python -m me_engineering_assistant
```

---

## 🧪 10. Testing & Validation Strategy

### Included automated tests

```test_agent_e2e.py``` validates:

- Routing correctness
- Retrieval behavior
- Final answer quality
- End-to-end execution stability

Run tests:
```bash
pytest -q
```

---

## 🐳 11. Containerization (Experimental)

A Dockerfile is included as a template, but not yet fully validated.

Example usage:
```bash
docker build -t me-assistant .
docker run -p 8000:8000 \
    -e MODEL_URI=models:/me-engineering-assistant@prod \
    me-assistant
```

Areas requiring further work:
- Preloading Phi-3 model
- MLflow filesystem mounting
- Performance tuning inside container

---

## ⚠️ 12. Limitations

- Phi-3 may hallucinate on ambiguous questions
- Router is rule-based (no embedding classifier yet)
- Vectorstore rebuilt at runtime (non-persistent)
- Docker build still experimental
- No streaming answer support

---

## 🚧 13. Future Work

### MLOps Enhancements

- Fully hardened Docker deployment
- MLflow model serving (```mlflow models serve```)
- Upgrade to SQLite/Postgres backend


### Agent Improvements

- Embedding-based router classifier
- Human-in-the-loop validation
- Multi-step reasoning with tool use
- Support more ECU manual families

### Retrieval Performance

- Persistent FAISS/Chroma index
- Quantized Phi-3 for faster inference

---

## 🏁 14. Challenge Requirements Alignment

| Requirement                 | Status                                 |
| --------------------------- | -------------------------------------- |
| Multi-source RAG            | ✔ Implemented                          |
| Intelligent routing         | ✔ Router node                          |
| LangGraph agent             | ✔ Two-node workflow                    |
| MLflow model logging        | ✔ Custom pyfunc, versioned, prod alias |
| REST API                    | ✔ FastAPI `/predict`                   |
| Dockerization               | ⚪ Template included (experimental)     |
| Architectural documentation | ✔ Included                             |
| Testing strategy            | ✔ Automated + proposed framework       |
| Limitations & future work   | ✔ Documented                           |

---

## 🙌 Acknowledgements

- Microsoft Phi-3 Model
- LangChain / LangGraph
- MLflow
- ChromaDB