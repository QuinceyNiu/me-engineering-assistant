from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel


def main() -> None:
    """Log the current RAG agent as an MLflow pyfunc model into mlruns/0."""

    # 构造一个小的 input_example，方便后续调试和推理
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    model = MEEngineeringAssistantModel()

    # 使用本地 ./mlruns 作为 tracking 目录（其实默认也是这样）
    tracking_dir = Path("mlruns")
    tracking_dir.mkdir(exist_ok=True)

    # 明确指定一下 tracking uri，避免以后改 cwd 搞混
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    # 默认 experiment 名叫 "Default"（对应 mlruns/0）
    mlflow.set_experiment("Default")

    with mlflow.start_run() as run:
        # 注意：这里不再传 code_path（你的当前 mlflow 版本不支持）
        mlflow.pyfunc.log_model(
            artifact_path="me_engineering_assistant_model",
            python_model=model,
            input_example=input_example,
        )

        print("Logged MLflow model to experiment 'Default', run_id=", run.info.run_id)
        print(
            "You can use MODEL_URI like: "
            f"runs:/{run.info.run_id}/me_engineering_assistant_model"
        )


if __name__ == "__main__":
    main()
