from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from cropgrm.config import CLASS_NAMES, DEFAULT_MODEL_PATHS, ModelType


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
        """Predict crop labels, skipping failed fits and rejecting low confidence.

        Args:
            df (pandas.DataFrame): Input table containing every feature named by the
                selected model. Optional ndmi_fit_status and wrdvi_fit_status columns
                must both be present when either is supplied; only success is eligible.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Required lower bound for the maximum class probability;
                equality is rejected.

        Returns:
            pandas.DataFrame: Input columns with integer class and mapped class_name
                and probability columns; rejected rows have class 0 and an unmapped
                class_name. Probability is the maximum before thresholding, or NaN
                for skipped fits. prediction_status records the rejection reason.

        Raises:
            ValueError: If model_type is unsupported or required model features are
                missing, for example because double-logistic fitting failed.
            FileNotFoundError: If the selected model file is missing.
        """
        model = self.load_model(model_type)
        fit_columns = ["ndmi_fit_status", "wrdvi_fit_status"]
        has_fit_status = any(name in df.columns for name in fit_columns)
        eligible = np.ones(len(df), dtype=bool)
        statuses = np.full(len(df), "predicted", dtype=object)
        if has_fit_status:
            if not all(name in df.columns for name in fit_columns):
                raise ValueError(
                    "Both ndmi_fit_status and wrdvi_fit_status are required."
                )
            fits = df[fit_columns]
            if (
                not fits.isin(["success", "insufficient_observations", "fit_failed"])
                .all()
                .all()
            ):
                raise ValueError("Invalid or missing curve fit status.")
            eligible = fits.eq("success").all(axis=1).to_numpy()
            statuses[~eligible] = "fit_failed"
            statuses[fits.eq("insufficient_observations").any(axis=1).to_numpy()] = (
                "insufficient_observations"
            )
        missing = [name for name in model.feature_names_ if name not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required model features: {', '.join(missing)}. "
                "Check feature preparation and input data. Double-logistic features "
                "may be unavailable if fewer than six observations remain after "
                "DOY >= 90 filtering or fitting fails on noisy data."
            )
        df_result = df.copy()
        max_probabilities = np.full(len(df), np.nan)
        classes = np.zeros(len(df), dtype=int)
        if eligible.any():
            probabilities = model.predict_proba(
                df_result.loc[eligible, model.feature_names_]
            )
            maxima = np.max(probabilities, axis=1)
            predicted_classes = np.asarray(model.classes_)[
                np.argmax(probabilities, axis=1)
            ]
            max_probabilities[eligible] = maxima
            classes[eligible] = np.where(maxima > threshold, predicted_classes, 0)
            statuses[eligible] = np.where(
                maxima > threshold, "predicted", "low_confidence"
            )
        df_result["probability"] = max_probabilities
        df_result["class"] = classes
        df_result["class_name"] = df_result["class"].map(CLASS_NAMES)
        df_result["prediction_status"] = statuses

        return df_result
