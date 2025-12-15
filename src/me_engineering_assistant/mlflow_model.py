"""MLflow pyfunc wrapper for serving the ME Engineering Assistant pipeline.

Implements an MLflow-compatible model that routes questions, retrieves context, and generates
final answers through the internal agent workflow.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import mlflow.pyfunc

# Cache the agent entrypoint so repeated predict() calls do not repeatedly import graph.
_RUN_AGENT: Optional[Callable[[str], dict]] = None


def _get_run_agent() -> Callable[[str], dict]:
    """Lazily import and cache the agent runner."""
    global _RUN_AGENT
    if _RUN_AGENT is None:
        from .graph import run_agent  # pylint: disable=import-outside-toplevel
        _RUN_AGENT = run_agent
    return _RUN_AGENT


class MEEngineeringAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ME Engineering Assistant agent."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Run the agent on a DataFrame with a 'question' column."""
        del context  # unused, kept for API compatibility
        del params  # unused, kept for API compatibility

        if "question" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'question' column.")

        run_agent = _get_run_agent()
        questions = model_input["question"].tolist()

        answers = []
        for q in questions:
            state = run_agent(q)
            answers.append(state.get("answer", ""))

        return pd.DataFrame({"answer": answers})

    def predict_stream(self, context, model_input: pd.DataFrame, params=None):
        """Optional streaming API not implemented for this model."""
        raise NotImplementedError("Streaming predictions are not implemented.")
