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
    """
    app = FastAPI(title="ME Engineering Assistant API")

    # 通过 MLflow 加载你刚刚 log 的自定义 pyfunc 模型
    model = mlflow.pyfunc.load_model(model_uri)

    @app.post("/predict", response_model=AnswerResponse)
    def predict(req: QuestionRequest) -> AnswerResponse:
        # MLflow 模型约定输入是 DataFrame，列名为 'question'
        df = pd.DataFrame({"question": req.questions})
        out = model.predict(df)
        return AnswerResponse(answers=out["answer"].tolist())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
