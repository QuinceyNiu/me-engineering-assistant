from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel

# Fixed Registered Model name, referenced later via alias
REGISTERED_MODEL_NAME = "me-engineering-assistant"
EXPERIMENT_NAME = "me-engineering-assistant"


def main() -> None:
    """
    Log the current RAG agent as an MLflow pyfunc model, register it,
    and update the "prod" alias to point to the latest version.

    Operation method:
        python -m me_engineering_assistant.log_model

    Afterwards, you can use:
        MODEL_URI = “models:/me-engineering-assistant@prod”
    to load the latest production model without manually copying the run_id.
    """

    # Construct a small input_example to facilitate subsequent inference and automatic signature deduction.
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    # Instantiate the current RAG Agent / Model Encapsulation
    model = MEEngineeringAssistantModel()

    # Reserve a directory for the local file backend (even if you're using a DB backend, it won't cause any issues).
    tracking_dir = Path("mlruns")
    tracking_dir.mkdir(exist_ok=True)

    # ⚠️ Do not force the tracking_uri; respect the MLFLOW_TRACKING_URI environment variable.
    # If you really want to force the use of local file storage, uncomment the following two lines:
    # mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    # Use a dedicated experiment name to avoid mixing with other experiments.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start a new run and log + register the model within it.
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        # Wrap the current Agent as a pyfunc model and:
        # 1) Record it in the artifacts of the current run
        # 2) Register it in the Model Registry with the name REGISTERED_MODEL_NAME
        _ = mlflow.pyfunc.log_model(
            artifact_path="me_engineering_assistant_model",  # 此参数虽有弃用 warning，但仍完全可用
            python_model=model,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"Logged MLflow model to experiment '{EXPERIMENT_NAME}', run_id = {run_id}")
        print(
            "Run-specific MODEL_URI (仅调试用):\n"
            f"  runs:/{run_id}/me_engineering_assistant_model"
        )

    # Use the alias feature of the Model Registry to automatically point “prod” to the latest version.
    client = MlflowClient()

    # Use `search_model_versions` (recommended API) instead of the outdated `get_latest_versions`
    # Filter condition: Retrieve all versions under this Registered Model name
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"No versions found for registered model '{REGISTERED_MODEL_NAME}'. "
            "Please make sure log_model ran successfully."
        )

    # Take the version with the highest number as the “latest version”
    latest_version = max(versions, key=lambda v: int(v.version))

    # Set the alias “prod” to point to this latest version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="prod",
        version=latest_version.version,
    )

    print(
        "\nRegistered model updated:"
        f"\n  name    = {REGISTERED_MODEL_NAME}"
        f"\n  version = {latest_version.version}"
        f"\n  alias   = prod"
    )
    print(
        "\nRecommended MODEL_URI:\n"
        f"  models:/{REGISTERED_MODEL_NAME}@prod\n"
    )
    print("Model logging & alias update finished successfully.\n")


if __name__ == "__main__":
    main()
