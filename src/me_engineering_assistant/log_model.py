from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from .mlflow_model import MEEngineeringAssistantModel


def main() -> None:
    # Construct a small input_example for subsequent inference and debugging.
    input_example = pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-850b?"]}
    )

    model = MEEngineeringAssistantModel()

    # Save directly to the local directory without going through the runs:/ file structure.
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
