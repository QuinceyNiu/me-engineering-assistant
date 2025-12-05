from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel


def main() -> None:
    # 构造一个小的 input_example，方便后续推理和调试
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    model = MEEngineeringAssistantModel()

    # 直接保存到本地目录，不走 runs:/ 的文件结构
    save_path = Path("saved_model") / "me_engineering_assistant_model"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.pyfunc.save_model(
        path=str(save_path),
        python_model=model,
        input_example=input_example,
    )

    print("Saved MLflow model to:", save_path.resolve())


if __name__ == "__main__":
    main()
