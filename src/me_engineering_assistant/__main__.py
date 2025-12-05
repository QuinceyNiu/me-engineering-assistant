import os

import uvicorn

from .api import create_app


def main() -> None:
    """
    作为命令行入口：
    - 读取环境变量 MODEL_URI（MLflow 模型地址）
    - 启动 FastAPI + Uvicorn
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
