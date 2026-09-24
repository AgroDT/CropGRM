from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

from crop_classifier.config import ModelType, OutputFormat, CLASS_NAMES
from crop_classifier.prediction.processors.make_predictions import CropPredictor
from crop_classifier.prediction.exporters import ExporterFactory


class CropClassifierPipeline:
    """Data clasification and export."""

    def __init__(self, custom_model_paths: Optional[Dict[ModelType, Path]] = None):
        self.predictor = CropPredictor(custom_model_paths=custom_model_paths)

    def run(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        output_prefix: str,
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
        output_format: Union[OutputFormat, str] = OutputFormat.TABLE,
        fields_geometry_path: Optional[Union[str, Path]] = None,
        epsg_code: str = "EPSG:32637",
    ) -> None:
        
        if isinstance(input_data, (str, Path)):
            df = pd.read_parquet(input_data)
        else:
            df = input_data.copy()

        df_predicted = self.predictor.predict(
            df=df,
            model_type=model_type,
            threshold=threshold,
        )

        fmt = OutputFormat(output_format)
        exporter = ExporterFactory.get_exporter(fmt, df_predicted)

        exporter.export(
            df=df_predicted,
            output_prefix=output_prefix,
            fields_geometry_path=fields_geometry_path,
            class_names=CLASS_NAMES,
            epsg_code=epsg_code,
        )