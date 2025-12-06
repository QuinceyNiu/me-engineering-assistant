from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel

# 固定的 Registered Model 名字，后面通过别名来引用
REGISTERED_MODEL_NAME = "me-engineering-assistant"
EXPERIMENT_NAME = "me-engineering-assistant"


def main() -> None:
    """
    Log the current RAG agent as an MLflow pyfunc model, register it,
    and update the "prod" alias to point to the latest version.

    运行方式:
        python -m me_engineering_assistant.log_model

    之后即可使用:
        MODEL_URI = "models:/me-engineering-assistant@prod"
    来加载最新的生产模型，而无需手动复制 run_id。
    """

    # 1. 构造一个小的 input_example，方便后续推理 & 自动推断签名
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    # 2. 实例化当前的 RAG Agent / 模型封装
    model = MEEngineeringAssistantModel()

    # 3. 为本地文件后端预留目录（即使你用的是 DB backend，也不会有坏影响）
    tracking_dir = Path("mlruns")
    tracking_dir.mkdir(exist_ok=True)

    # ⚠️ 不强行指定 tracking_uri，尊重环境变量 MLFLOW_TRACKING_URI
    # 如果你真的想强制使用本地文件存储，可以解除下面两行注释：
    # mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    # 4. 使用一个专门的 experiment 名称，避免和别的实验混在一起
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 5. 开始一个新的 run，并在其中 log + 注册模型
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        # 将当前 Agent 封装为 pyfunc 模型并:
        # 1) 记录到当前 run 的 artifacts
        # 2) 注册到 Model Registry，名称 REGISTERED_MODEL_NAME
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

    # 6. 使用 Model Registry 的 alias 功能，自动把 "prod" 指向最新版本
    client = MlflowClient()

    # 使用 search_model_versions（推荐 API），而不是旧的 get_latest_versions
    # 过滤条件：只查这个 Registered Model 名字下的所有版本
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")

    if not versions:
        # 理论上不会发生，因为上面刚刚 log 了一个版本
        raise RuntimeError(
            f"No versions found for registered model '{REGISTERED_MODEL_NAME}'. "
            "Please make sure log_model ran successfully."
        )

    # 取 version 数字最大的作为“最新版本”
    latest_version = max(versions, key=lambda v: int(v.version))

    # 将别名 "prod" 指向这个最新版本（如果已存在会被覆盖）
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
        "\nRecommended MODEL_URI (建议全部环境都用这个):\n"
        f"  models:/{REGISTERED_MODEL_NAME}@prod\n"
    )
    print("Model logging & alias update finished successfully.\n")


if __name__ == "__main__":
    main()
