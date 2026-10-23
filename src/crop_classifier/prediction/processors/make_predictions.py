from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from crop_classifier.config import CLASS_NAMES, DEFAULT_MODEL_PATHS, ModelType


class CropPredictor:
    """Load local CatBoost models and classify crop features."""

    def __init__(self, custom_model_paths: Dict[ModelType, Path] = None):
        """Configure paths to the available crop models.

        Args:
            custom_model_paths (dict[ModelType, Path] | None): Optional model path
                mapping; None or an empty mapping uses the configured defaults.
        """
        self.model_paths = custom_model_paths or DEFAULT_MODEL_PATHS

    def load_model(self, model_type: Union[ModelType, str]) -> CatBoostClassifier:
        """Load the selected CatBoost model from disk.

        Args:
            model_type (ModelType | str): Model variant or its configured string value.

        Returns:
            CatBoostClassifier: Loaded model ready for inference.

        Raises:
            ValueError: If model_type is not a supported model variant.
            FileNotFoundError: If the model has no configured path or the file does not
                exist.
        """
        model_type = ModelType(model_type)
        path = self.model_paths.get(model_type)

        if not path or not Path(path).exists():
            raise FileNotFoundError(f"File {model_type.value} not found: {path}")

        return CatBoostClassifier().load_model(str(path))

    def predict(
        self,
        df: pd.DataFrame,
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Predict crop labels and reject low-confidence results.

        Args:
            df (pandas.DataFrame): Input table containing every feature named by the
                selected model.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Required lower bound for the maximum class probability;
                equality is rejected.

        Returns:
            pandas.DataFrame: Copy of df with integer class and mapped class_name
                and probability columns; rejected rows have class 0 and an unmapped
                class_name. Probability is the maximum before thresholding.

        Raises:
            ValueError: If model_type is unsupported or required model features are
                missing, for example because double-logistic fitting failed.
            FileNotFoundError: If the selected model file is missing.
        """
        model = self.load_model(model_type)
        missing = [name for name in model.feature_names_ if name not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required model features: {', '.join(missing)}. "
                "Check feature preparation and input data. Double-logistic features "
                "may be unavailable if fewer than six observations remain after "
                "DOY >= 90 filtering or fitting fails on noisy data."
            )
        df_result = df.copy()

        probabilities = model.predict_proba(df_result[model.feature_names_])

        max_probabilities = np.max(probabilities, axis=1)
        df_result["probability"] = max_probabilities
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_classes = [model.classes_[idx] for idx in predicted_indices]

        df_result["class"] = np.where(
            max_probabilities > threshold, predicted_classes, np.nan
        )
        df_result["class"] = df_result["class"].fillna(0).astype(int)
        df_result["class_name"] = df_result["class"].map(CLASS_NAMES)

        return df_result
