import pandas as pd
import mlflow.pyfunc

from .graph import run_agent


class MEEngineeringAssistantModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper around the ME Engineering Assistant agent.

    Input:  pandas.DataFrame with column 'question'
    Output: pandas.DataFrame with column 'answer'
    """

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if "question" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'question' column.")

        questions = model_input["question"].tolist()
        answers = []

        for q in questions:
            state = run_agent(q)
            answers.append(state["answer"])

        return pd.DataFrame({"answer": answers})
