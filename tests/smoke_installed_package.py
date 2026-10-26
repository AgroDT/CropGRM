"""Check a wheel installation with isolated Python from outside the checkout."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

import cropgrm
from cropgrm import CropClassifier, ModelType
from cropgrm.config import DEFAULT_MODEL_PATHS
from cropgrm.prediction.processors.make_predictions import CropPredictor


def main():
    checkout = Path(__file__).resolve().parents[1]
    package = Path(cropgrm.__file__).resolve().parent
    if Path.cwd().resolve().is_relative_to(checkout):
        raise AssertionError("Run this check outside the checkout.")
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(f"Expected an installed package, imported {package}")
    if package.is_relative_to(checkout):
        raise AssertionError(f"Imported package from the checkout: {package}")
    if importlib.util.find_spec("crop_classifier") is not None:
        raise AssertionError("The old import package is still available.")

    predictor = CropPredictor()
    for model_type in ModelType:
        path = DEFAULT_MODEL_PATHS[model_type].resolve()
        if not path.is_file() or not path.is_relative_to(package / "models"):
            raise AssertionError(f"Missing bundled model: {path}")
        model = predictor.load_model(model_type)
        if not model.feature_names_:
            raise AssertionError(f"Missing feature names: {model_type.value}")
        data = pd.DataFrame(0.0, index=range(2), columns=model.feature_names_)
        data["field_id"] = ["001", "002"]
        result = CropClassifier(model_type=model_type).predict(data)
        if result["field_id"].tolist() != ["001", "002"]:
            raise AssertionError("Prediction did not preserve field identifiers.")
        if not {"class", "class_name", "probability"}.issubset(result.columns):
            raise AssertionError("Prediction columns are missing.")
        print(f"Loaded and predicted with bundled model: {model_type.value}")
    print(f"Wheel smoke test passed: {package}")


if __name__ == "__main__":
    main()
