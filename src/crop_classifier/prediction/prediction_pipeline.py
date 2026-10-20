from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from crop_classifier.config import CLASS_NAMES, ModelType, OutputFormat
from crop_classifier.prediction.exporters import ExporterFactory
from crop_classifier.prediction.processors.make_predictions import CropPredictor


class CropClassifierPipeline:
    """Classify prepared features and export crop predictions."""

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
        output_prefix: str,
        use_zonal_spectral: bool,
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
        output_format: Union[OutputFormat, str] = OutputFormat.TABLE,
        fields_geometry_path: Optional[Union[str, Path]] = None,
        input_epsg_code: Optional[str] = "EPSG:4326",
        output_epsg_code: Optional[str] = None,
    ) -> None:
        """Predict crop classes and write the requested output.

        Args:
            input_data (str | Path | pandas.DataFrame): Feature Parquet path or table
                containing the selected model features and spatial keys.
            output_prefix (str | Path): Destination path prefix without a file
                extension.
            use_zonal_spectral (bool): Whether raster output uses field polygons instead
                of pixel coordinates.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Class probability that must be strictly exceeded; other
                rows receive class 0.
            output_format (OutputFormat | str): Requested table, vector, or raster
                format.
            fields_geometry_path (str | Path | None): Field geometry file joined by
                field_id; required for vector and polygon raster exports.
            input_epsg_code (str | None): CRS of pixel coordinates, passed to the raster
                exporter.
            output_epsg_code (str | None): Output raster CRS; None supplies no CRS
                override.

        Raises:
            FileNotFoundError: If the selected model file is missing.
            ValueError: If a model or output format is unsupported, or required vector
                or polygon export inputs are missing.
        """
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
        exporter = ExporterFactory.get_exporter(fmt, use_zonal_spectral, df_predicted)

        exporter.export(
            df=df_predicted,
            output_prefix=output_prefix,
            fields_geometry_path=fields_geometry_path,
            class_names=CLASS_NAMES,
            input_epsg_code=input_epsg_code,
            output_epsg_code=output_epsg_code,
        )
