from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from cropgrm.config import ModelType
from cropgrm.prediction.processors.make_predictions import CropPredictor


class CropClassifierPipeline:
    """Classify prepared features and return the full prediction table."""

    def __init__(self, custom_model_paths: Optional[Dict[ModelType, Path]] = None):
        """Initialize the predictor with optional model paths.

        Args:
            custom_model_paths (dict[ModelType, Path] | None): Optional model path
                mapping; None or an empty mapping uses the configured defaults.
        """
        self.predictor = CropPredictor(custom_model_paths=custom_model_paths)

    def run(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Predict crop classes without exporting.

        Args:
            input_data (str | Path | pandas.DataFrame): Feature Parquet path or table
                containing the selected model features and spatial keys.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Class probability that must be strictly exceeded; other
                rows receive class 0.

        Returns:
            pandas.DataFrame: Input columns plus crop classes and probabilities.

        Raises:
            FileNotFoundError: If the selected model file is missing.
            ValueError: If the model type is unsupported.
        """
        if isinstance(input_data, (str, Path)):
            df = pd.read_parquet(input_data)
        else:
            df = input_data

        return self.predictor.predict(
            df=df,
            model_type=model_type,
            threshold=threshold,
        )
