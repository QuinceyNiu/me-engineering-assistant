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
    Create a FastAPI application and load the MLflow model.
    """
    app = FastAPI(title="ME Engineering Assistant API")

    model = mlflow.pyfunc.load_model(model_uri)

    @app.post("/predict", response_model=AnswerResponse)
    def predict(req: QuestionRequest) -> AnswerResponse:
        df = pd.DataFrame({"question": req.questions})
        out = model.predict(df)
        return AnswerResponse(answers=out["answer"].tolist())

    return app
