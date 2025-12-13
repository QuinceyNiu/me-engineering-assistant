"""FastAPI service layer for the ME Engineering Assistant.

Defines request/response schemas and exposes HTTP endpoints (e.g., /predict) that call the
underlying MLflow/agent pipeline to produce answers.
"""

from typing import List

import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    questions: List[str]


class AnswerResponse(BaseModel):
    answers: List[str]


def create_app(model_uri: str) -> FastAPI:
    """
    Create a FastAPI application and load the MLflow pyfunc model.

    The model_uri parameter can be:
      - A specific model from a particular run:
          runs:/<run_id>/me_engineering_assistant_model
      - An alias via the Model Registry (recommended):
          models:/me-engineering-assistant@prod
    """
    app = FastAPI(title="ME Engineering Assistant API")

    # Load your custom pyfunc model that you just logged using MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    @app.post("/predict", response_model=AnswerResponse)
    def predict(req: QuestionRequest) -> AnswerResponse:
        # MLflow model convention: Input is a DataFrame with column name 'question'
        df = pd.DataFrame({"question": req.questions})
        out = model.predict(df)
        return AnswerResponse(answers=out["answer"].tolist())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
