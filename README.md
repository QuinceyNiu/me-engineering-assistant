# ME Engineering Assistant – RAG + LangGraph + MLflow Agent

A production-ready Retrieval-Augmented Generation (RAG) agent designed to help ME engineers quickly retrieve and compare specifications from the **ECU-700** and **ECU-800** series manuals.

The system intelligently routes queries, retrieves relevant content, and generates concise technical answers using a local LLM (Phi-3 by default). It is packaged as a Python module, logged with MLflow, and exposes a FastAPI prediction endpoint.

---

## 1. Project Overview

Engineers frequently need to answer questions such as:

- “What is the maximum operating temperature of the ECU-850b?”
- “Compare CAN bus speeds between ECU-750 and ECU-850.”

This agent solves the problem by combining:

- Automated document routing using LangGraph  
- Vector-based retrieval with FAISS  
- Local LLM reasoning (Phi-3-mini)  
- MLflow model packaging  
- FastAPI serving

---

## 2. Repository Structure

```text
me-engineering-assistant/
│
├── README.md
├── pyproject.toml
│
├── data/
│   ├── ECU-700_Series_Manual.md
│   ├── ECU-800_Series_Base.md
│   ├── ECU-800_Series_Plus.md
│   └── test-questions.csv
│
├── mlruns/
│   └── 0/
│
├── src/
│   └── me_engineering_assistant/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── config.py
│       ├── data_loader.py
│       ├── graph.py
│       ├── log_model.py
│       ├── mlflow_model.py
│       ├── rag_chain.py
│       ├── router.py
│       ├── sandbox_test.py
│       ├── vectorstore.py
│
├── tests/
│   └── test_agent_e2e.py
```


---

## 3. Architecture Overview

### 3.1 System Components

| Component | Description |
|----------|-------------|
| `router.py` | Routes questions to the correct manual set using LangGraph |
| `vectorstore.py` | Builds FAISS index + HuggingFace embeddings |
| `rag_chain.py` | Retrieval-Augmented Generation pipeline |
| `graph.py` | LangGraph workflow orchestrating router + RAG |
| `api.py` | FastAPI server exposing `/predict` |
| `mlflow_model.py` | MLflow pyfunc wrapper with predict() |
| `log_model.py` | Logs the agent as an MLflow model |
| `sandbox_test.py` | Command-line testing script |

---

## 4. Installation

### 4.1 Clone the repository

```bash
git clone <repo-url>
cd me-engineering-assistant
```

### 4.2 Create a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Running the Agent Locally (CLI Test)

```bash
python -m me_engineering_assistant.sandbox_test
```
Example output:
```vbnet
Question: What is the maximum operating temperature for the ECU-850b?
Answer: The maximum operating temperature for the ECU-850b is +105°C.
```

---

## 6. Start the FastAPI Server

```bash
python -m me_engineering_assistant
```
The server will start at:
```bash
http://localhost:8000/predict
```

---

## 7. Example API Usage

### 7.1 cURL

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is the maximum operating temperature of the ECU-850b?"]}'
```

### 7.2 Python

```bash
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    json={"questions": ["What is the maximum operating temperature of the ECU-850b?"]}
)
print(resp.json())
```

### 7.3 Postman

* POST ```http://localhost:8000/predict```
* Body (JSON)
```json
{
  "questions": ["What is the maximum operating temperature of the ECU-850b?"]
}
```

---

## 8. MLflow Model Logging

Log the model:
```bash
python -m me_engineering_assistant.log_model
```

MLflow will generate a run under:
```php-template
mlruns/0/<run_id>/
```

Set the model URI:
```bash
export MODEL_URI="runs:/<run_id>/me_engineering_assistant_model"
```

---

## 9. Running Tests

```bash
pytest -q
```

Expected:
```text
1 passed, 0 failed
```
The tests validate routing, retrieval, and answer correctness.

---

## 10. Docker Deployment (Optional)

Build:
```bash
docker build -t me-assistant .
```

Run:
```bash
docker run -p 8000:8000 -e MODEL_URI=$MODEL_URI me-assistant
```

---

## 11. Design Decisions

Why LangGraph?

* Deterministic routing

* Declarative workflow

* Better traceability vs. vanilla LangChain agents

Why Local LLM (Phi-3)?

* Reproducible

* Zero external API dependencies

* Fast on Apple Silicon (MPS + BF16)

Why MLflow?

* Model versioning

* Packaging reproducibility

* Standardized predict() interface

Chunking Strategy

* 500-char window with 50-char overlap

* Optimized for dense technical manuals

---

## 12. Limitations

| Limitation                             | Notes                                     |
| -------------------------------------- | ----------------------------------------- |
| Local LLM may hallucinate              | Mitigated by strict prompt + RAG context  |
| Router is rule-based                   | Could be upgraded to embedding classifier |
| Not optimized for very large documents | Current FAISS index fits small manuals    |
| No streaming responses                 | Could be added with FastAPI websockets    |

---

## 13. Future Work