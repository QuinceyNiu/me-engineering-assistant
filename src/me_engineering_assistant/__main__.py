import os
from pathlib import Path

import uvicorn

from .api import create_app

# Fallback if no filesystem model is available
DEFAULT_MODEL_URI = "models:/me-engineering-assistant@prod"


def _detect_default_model_uri() -> str:
    """
    Detect the best MODEL_URI to use in the current runtime environment.

    Priority:
    1. Explicit environment variable MODEL_URI
    2. Docker image: /app/saved_model
    3. Local development: <project_root>/saved_model
    4. MLflow Model Registry alias: models:/me-engineering-assistant@prod
    """
    # 1. Respect explicit override
    env_uri = os.getenv("MODEL_URI")
    if env_uri:
        return env_uri

    # 2. Inside Docker image: the Dockerfile copies saved_model/ to /app/saved_model
    docker_saved = Path("/app/saved_model")
    if docker_saved.exists():
        return str(docker_saved)

    # 3. Local development: assume we run from the installed package under src/
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # …/src/ -> project root parent
    local_saved = project_root / "saved_model"
    if local_saved.exists():
        return str(local_saved)

    # 4. Fallback: use the Model Registry alias
    return DEFAULT_MODEL_URI


def main() -> None:
    """
    Command-line entry point.

    - Detect the appropriate MODEL_URI (filesystem or MLflow registry)
    - Create the FastAPI app around the loaded MLflow pyfunc model
    - Run the Uvicorn HTTP server
    """
    model_uri = _detect_default_model_uri()

    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    print(f"Starting API with MODEL_URI = {model_uri}")
    app = create_app(model_uri)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
