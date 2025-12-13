"""MLflow pyfunc wrapper for serving the ME Engineering Assistant pipeline.

Implements an MLflow-compatible model that routes questions, retrieves context, and generates
final answers through the internal agent workflow.
"""

import pandas as pd
import mlflow.pyfunc


class MEEngineeringAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ME Engineering Assistant agent."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Run the agent on a DataFrame with a 'question' column."""
        # Lazy loading to avoid loading the entire RAG/large model when importing this module
        from .graph import run_agent

        del params  # unused, kept for API compatibility

        if "question" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'question' column.")

        questions = model_input["question"].tolist()
        answers = []

        for q in questions:
            state = run_agent(q)
            answers.append(state["answer"])

        return pd.DataFrame({"answer": answers})

    def predict_stream(self, context, model_input: pd.DataFrame, params=None):
        """Optional streaming API not implemented for this model."""
        raise NotImplementedError("Streaming predictions are not implemented.")
