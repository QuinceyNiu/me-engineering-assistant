import os

import uvicorn

from .api import create_app


def main() -> None:
    """
    As a command-line entry point:
    - Read the environment variable MODEL_URI (MLflow model URL)
    - Launch FastAPI + Uvicorn
    """
    model_uri = os.environ.get(
        "MODEL_URI",
        "runs:/REPLACE_ME/me_engineering_assistant_model",
    )
    port = int(os.environ.get("PORT", 8000))

    app = create_app(model_uri)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
