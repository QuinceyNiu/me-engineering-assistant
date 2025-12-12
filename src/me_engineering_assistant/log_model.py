from pathlib import Path
import shutil

import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel

# Fixed Registered Model name, referenced later via alias
REGISTERED_MODEL_NAME = "me-engineering-assistant"
EXPERIMENT_NAME = "me-engineering-assistant"

# Subdirectory name used when logging the pyfunc model as an artifact
ARTIFACT_SUBPATH = "me_engineering_assistant_model"


def _export_latest_model_to_saved_dir(run_id: str) -> Path:
    """
    Export the latest logged pyfunc model artifacts into a fixed directory
    called `saved_model/` under the current working directory.

    This makes the model easy to consume both in local development and inside
    Docker, without needing to know the internal MLflow `mlruns/` layout.
    """
    # Use MLflow's `runs:/` URI to locate the pyfunc model artifacts
    model_uri = f"runs:/{run_id}/{ARTIFACT_SUBPATH}"

    # Download the artifacts to a local directory (MLflow chooses the temp path)
    local_artifacts_path = Path(download_artifacts(model_uri))

    # Assume the command is invoked from the project root
    project_root = Path.cwd()
    saved_model_dir = project_root / "saved_model"

    # Replace any previous exported model
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    shutil.copytree(local_artifacts_path, saved_model_dir)

    print(
        f"\n[log_model] Exported latest model artifacts to: {saved_model_dir}\n"
        "You can now load the model via:\n"
        "  - MODEL_URI=./saved_model        (local development)\n"
        "  - MODEL_URI=/app/saved_model     (inside Docker)\n"
    )
    return saved_model_dir


def main() -> None:
    """
    Log the current RAG agent as an MLflow pyfunc model, register it, update the
    `prod` alias to the latest version, and export the artifacts to `saved_model/`.

    Usage:

        python -m me_engineering_assistant.log_model

    After running this once, you can start the API with:

        - Local dev: MODEL_URI=./saved_model
        - Docker:    MODEL_URI=/app/saved_model  (the Dockerfile copies it there)
    """
    # Small input_example to help MLflow infer the model signature
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    # Wrap the current RAG Agent / Model implementation
    model = MEEngineeringAssistantModel()

    # Reserve a directory for the local file backend (safe even if you later
    # switch to a DB backend).
    tracking_dir = Path("mlruns")
    tracking_dir.mkdir(exist_ok=True)

    # DO NOT override MLFLOW_TRACKING_URI here; respect the environment variable.
    # If you really want to force file storage, you could uncomment:
    # mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    # Use a dedicated experiment to avoid mixing with other runs
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log and register the model within a single run
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        # 1) Log the model artifacts under ARTIFACT_SUBPATH
        # 2) Register it in the Model Registry with the given name
        _ = mlflow.pyfunc.log_model(
            artifact_path=ARTIFACT_SUBPATH,
            # This parameter is deprecated but still fully functional in MLflow 2.x
            python_model=model,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        run_id = run.info.run_id
        print(
            f"Logged MLflow model to experiment '{EXPERIMENT_NAME}', "
            f"run_id = {run_id}"
        )
        print(
            "Run-specific MODEL_URI (for testing only):\n"
            f"  runs:/{run_id}/{ARTIFACT_SUBPATH}\n"
        )

    # Update the Model Registry alias "prod" to point to the latest version
    client = MlflowClient()

    # Use `search_model_versions` (recommended API) instead of the outdated
    # `get_latest_versions`.
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No versions found for registered model '{REGISTERED_MODEL_NAME}'. "
            "Please make sure log_model ran successfully."
        )

    # Take the version with the highest number as the "latest" one
    latest_version = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="prod",
        version=latest_version.version,
    )

    print(
        "\nRegistered model updated:"
        f"\n  name    = {REGISTERED_MODEL_NAME}"
        f"\n  version = {latest_version.version}"
        "\n  alias   = prod\n"
    )
    print(
        "Recommended MODEL_URI when using the Model Registry alias:\n"
        f"  models:/{REGISTERED_MODEL_NAME}@prod\n"
    )

    # Export artifacts to a fixed `saved_model/` directory for easier consumption
    _export_latest_model_to_saved_dir(run_id)

    print("Model logging, alias update, and export finished successfully.\n")


if __name__ == "__main__":
    main()
