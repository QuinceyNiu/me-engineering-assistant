import os

import uvicorn

from .api import create_app


# 默认使用 Model Registry 的别名 "prod"
DEFAULT_MODEL_URI = "models:/me-engineering-assistant@prod"


def main() -> None:
    """
    As a command-line entry point:
    - Read the environment variable MODEL_URI (MLflow model URL), if provided
    - Otherwise default to models:/me-engineering-assistant@prod
    - Launch FastAPI + Uvicorn
    """
    model_uri = os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)
    port = int(os.environ.get("PORT", 8000))

    print(f"Starting API with MODEL_URI = {model_uri}")
    app = create_app(model_uri)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
