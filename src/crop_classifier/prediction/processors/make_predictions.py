from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from crop_classifier.config import DEFAULT_MODEL_PATHS, ModelType
from crop_classifier.config import CLASS_NAMES

class CropPredictor:
    """Downlowding and inference CatBoost model."""

    def __init__(self, custom_model_paths: Dict[ModelType, Path] = None):
        self.model_paths = custom_model_paths or DEFAULT_MODEL_PATHS

    def load_model(self, model_type: Union[ModelType, str]) -> CatBoostClassifier:
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
        """Model inference and result filtration on threshold."""
        model = self.load_model(model_type)
        df_result = df.copy()

        probabilities = model.predict_proba(df_result[model.feature_names_])

        max_probabilities = np.max(probabilities, axis=1)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_classes = [model.classes_[idx] for idx in predicted_indices]

        df_result["class"] = np.where(
            max_probabilities > threshold, predicted_classes, np.nan
        )
        df_result["class"] = df_result["class"].fillna(0).astype(int)
        df_result["class_name"] = df_result["class"].map(CLASS_NAMES)

        return df_result