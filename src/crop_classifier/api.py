from math import isfinite
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from crop_classifier._optional import import_optional
from crop_classifier.config import ModelType, OutputFormat
from crop_classifier.prediction.exporters import ExporterFactory
from crop_classifier.prediction.prediction_pipeline import CropClassifierPipeline


class CropClassifier:
    """Pipeline for crop classification and mapping.

    Handles the end-to-end workflow:
    1. Google Earth Engine (GEE) data downloading.
    2. Feature engineering and extraction.
    3. Model inference and output export.
    """

    def __init__(
        self,
        shapefile_path: Optional[Union[str, Path]] = None,
        gee_project: Optional[str] = None,
        year: Optional[int] = None,
        output_dir: Union[str, Path] = ".",
        use_zonal_spectral: bool = True,
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
        output_format: Union[OutputFormat, str] = OutputFormat.TABLE,
        epsg_code: Optional[str] = None,
    ):
        """Initialize the crop classification workflow.

        Args:
            shapefile_path (str | Path | None): Input shapefile defining the regions of
                interest. Optional for exporting saved tables.
            gee_project (str | None): Earth Engine project; required for downloading.
            year (int | None): Imagery and weather year; required for downloading.
            output_dir (str | Path): Root directory for raw data, features, and
                predictions.
            use_zonal_spectral (bool): Whether to use field statistics rather than pixel
                data.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Minimum class probability to exceed for an accepted
                prediction.
            output_format (OutputFormat | str): Default format for save() and final
                export in run().
            epsg_code (str | None): Default CRS for raster save() and run() output.
                A CRS is required for point raster output.
        """
        self.input_file = Path(shapefile_path) if shapefile_path is not None else None
        self.project_id = gee_project
        self.year = year
        self.output_dir = Path(output_dir)
        self.use_zonal_spectral = use_zonal_spectral
        self.model_type = model_type
        self.threshold = threshold
        self.output_format = OutputFormat(output_format)
        self.epsg_code = epsg_code
        self._predictions: Optional[pd.DataFrame] = None

    # --- Properties for Default File Paths ---

    @property
    def _input_stem(self) -> str:
        return self.input_file.stem if self.input_file is not None else "predictions"

    @property
    def _file_ext(self) -> str:
        """Get the raw data extension for the extraction mode.

        Returns:
            str: "csv" for field statistics or "parquet" for pixel data.
        """
        return "csv" if self.use_zonal_spectral else "parquet"

    @property
    def default_spectral_path(self) -> Path:
        """Get the default path for raw spectral data.

        Returns:
            Path: Spectral CSV or Parquet path under the raw directory.
        """
        return self.output_dir / "raw" / f"{self._input_stem}_spectral.{self._file_ext}"

    @property
    def default_meteo_path(self) -> Path:
        """Get the default path for raw meteorological data.

        Returns:
            Path: Weather CSV or Parquet path under the raw directory.
        """
        return self.output_dir / "raw" / f"{self._input_stem}_meteo.{self._file_ext}"

    @property
    def default_processed_path(self) -> Path:
        """Get the default path for the feature dataset.

        Returns:
            Path: Feature Parquet path under the processed directory.
        """
        return self.output_dir / "processed" / f"{self._input_stem}_input.parquet"

    @property
    def default_prediction_prefix(self) -> Path:
        """Get the default prefix for prediction files.

        Returns:
            Path: Output path prefix under the final directory, without an extension.
        """
        return self.output_dir / "final" / f"{self._input_stem}"

    # --- Pipeline Steps ---

    def download_data(self) -> None:
        """Download Earth Engine observations; requires the gee extra."""
        if self.input_file is None or self.project_id is None or self.year is None:
            raise ValueError(
                "Downloading requires shapefile_path, gee_project, and year."
            )
        pipeline = import_optional("crop_classifier.gee_extractor.gee_pipeline", "gee")
        pipeline.GEEDownloadingDataPipeline(
            project_id=self.project_id,
            year=self.year,
            output_dir=self.output_dir,
            use_zonal_spectral=self.use_zonal_spectral,
        ).run(self.input_file)

    def build_features(
        self,
        spectral_path: Optional[Union[str, Path]] = None,
        meteo_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Build spectral and meteorological features.

        Args:
            spectral_path (str | Path | None): Raw spectral data path; None uses
                default_spectral_path.
            meteo_path (str | Path | None): Raw weather data path; None uses
                default_meteo_path.
            output_path (str | Path | None): Feature destination; None uses
                default_processed_path.

        Returns:
            Path: Path to the generated feature Parquet file.

        Raises:
            ValueError: If the datasets have no common spatial IDs or a file format is
                unsupported.
            ModuleNotFoundError: If the features extra is not installed.
        """
        src_spectral = (
            Path(spectral_path) if spectral_path else self.default_spectral_path
        )
        src_meteo = Path(meteo_path) if meteo_path else self.default_meteo_path
        dst_path = Path(output_path) if output_path else self.default_processed_path

        pipeline = import_optional(
            "crop_classifier.feature.feature_pipeline", "features"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.PredictorPipeline().build_features(
            self.use_zonal_spectral, src_spectral, src_meteo, dst_path
        )

        return dst_path

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify an in-memory feature table and cache predictions without export.

        Args:
            data (pandas.DataFrame): Prepared model features and spatial keys.
                The input table is not modified.

        Returns:
            pandas.DataFrame: All input columns, classes, and maximum probability.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            FileNotFoundError: If the selected model is unavailable.
            ValueError: If the model type is unsupported or required features are
                missing. A failed prediction clears the previous cached result.
        """
        self._predictions = None
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        predictions = CropClassifierPipeline().run(
            input_data=data,
            model_type=self.model_type,
            threshold=self.threshold,
        )
        self._predictions = predictions.copy(deep=True)
        return predictions

    def save(
        self,
        output_format: Optional[Union[OutputFormat, str]] = None,
        output_prefix: Optional[Union[str, Path]] = None,
        *,
        predictions: Optional[Union[pd.DataFrame, str, Path]] = None,
        fields_geometry_path: Optional[Union[str, Path]] = None,
        input_epsg_code: Optional[str] = "EPSG:4326",
        output_epsg_code: Optional[str] = None,
        pixel_size: Optional[float] = None,
    ) -> Path:
        """Export predictions without loading models or computing features.

        Args:
            output_format (OutputFormat | str | None): Format; None uses the
                constructor's output_format.
            output_prefix (str | Path | None): Prefix without extension; None uses
                default_prediction_prefix. Exporters append .csv, .tif, or .fgb.
            predictions (DataFrame | str | Path | None): Explicit table or saved CSV;
                None uses the last successful predict(). CSV preserves identifier
                strings. This argument takes precedence over the cached table.
            fields_geometry_path (str | Path | None): Field polygons with field_id;
                None uses shapefile_path. Required for vector and zonal raster.
            input_epsg_code (str | None): Point coordinate CRS, default EPSG:4326.
                Nondefault values require point raster output.
            output_epsg_code (str | None): Raster CRS; None uses epsg_code. Required
                for point raster; polygon raster otherwise retains its source CRS.
            pixel_size (float | None): Positive finite raster spacing in output CRS
                units; None uses 30.0. Raster options require raster output.

        Returns:
            Path: Created output file.

        Raises:
            ValueError: If predictions, required columns, geometry, CRS, or valid
                export options are missing, or the input format is unsupported.
            FileNotFoundError: If an explicit CSV or geometry file is unavailable.
            ModuleNotFoundError: If geographic export dependencies are missing.
        """
        fmt = OutputFormat(
            output_format if output_format is not None else self.output_format
        )
        if fmt != OutputFormat.RASTER and (
            pixel_size is not None
            or output_epsg_code is not None
            or input_epsg_code != "EPSG:4326"
        ):
            raise ValueError("CRS and pixel_size options require raster output.")
        if fmt == OutputFormat.TABLE and fields_geometry_path is not None:
            raise ValueError("fields_geometry_path requires vector or raster output.")
        if (
            fmt == OutputFormat.RASTER
            and self.use_zonal_spectral
            and input_epsg_code != "EPSG:4326"
        ):
            raise ValueError("input_epsg_code requires point raster output.")
        size = 30.0 if pixel_size is None else pixel_size
        if fmt == OutputFormat.RASTER and (not isfinite(size) or size <= 0):
            raise ValueError("pixel_size must be finite and positive.")
        source = self._predictions if predictions is None else predictions
        if source is None:
            raise ValueError(
                "No predictions available. Call predict() or pass predictions."
            )
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.suffix.lower() != ".csv":
                raise ValueError("Prediction input must be a saved CSV table (.csv).")
            source = pd.read_csv(
                source, dtype={"field_id": str}, float_precision="round_trip"
            )
        if not isinstance(source, pd.DataFrame):
            raise ValueError("predictions must be a DataFrame or a saved CSV path.")
        exporter = ExporterFactory.get_exporter(fmt, self.use_zonal_spectral, source)
        geometry = fields_geometry_path
        if geometry is None:
            geometry = self.input_file
        needs_geometry = fmt == OutputFormat.VECTOR or (
            fmt == OutputFormat.RASTER and self.use_zonal_spectral
        )
        if needs_geometry and geometry is None:
            raise ValueError(
                "fields_geometry_path is required for field geometry export."
            )
        epsg = output_epsg_code if output_epsg_code is not None else self.epsg_code
        if fmt == OutputFormat.RASTER and not self.use_zonal_spectral:
            if fields_geometry_path is not None:
                raise ValueError("Point raster uses lon/lat, not fields_geometry_path.")
            if not input_epsg_code or not epsg:
                raise ValueError(
                    "Point raster requires input_epsg_code and output_epsg_code."
                )
        prefix = (
            Path(output_prefix)
            if output_prefix is not None
            else self.default_prediction_prefix
        )
        prefix.parent.mkdir(parents=True, exist_ok=True)
        exporter.export(
            df=source.copy(deep=True),
            output_prefix=prefix,
            fields_geometry_path=geometry,
            input_epsg_code=input_epsg_code,
            output_epsg_code=epsg,
            **({"pixel_size": size} if fmt == OutputFormat.RASTER else {}),
        )
        suffix = {
            OutputFormat.TABLE: ".csv",
            OutputFormat.RASTER: ".tif",
            OutputFormat.VECTOR: ".fgb",
        }[fmt]
        return Path(f"{prefix}{suffix}")

    # --- Complete Pipeline Execution ---

    def run(self, pixel_size: Optional[float] = None) -> pd.DataFrame:
        """Download, build features, predict once, and export output_format.

        Args:
            pixel_size (float | None): Raster spacing in output CRS units; defaults
                to 30.0 for raster. An explicit value requires raster output.

        Returns:
            pandas.DataFrame: Predictions also saved as CSV and cached for save().

        Raises:
            ValueError: If raster spacing is invalid or supplied for another format.
        """
        fmt = OutputFormat(self.output_format)
        if pixel_size is not None and fmt != OutputFormat.RASTER:
            raise ValueError("pixel_size requires raster output.")
        if pixel_size is not None and (not isfinite(pixel_size) or pixel_size <= 0):
            raise ValueError("pixel_size must be finite and positive.")
        self._predictions = None
        self.download_data()
        processed_path = self.build_features()
        data = pd.read_parquet(processed_path)
        predictions = self.predict(data)
        self.save(OutputFormat.TABLE, predictions=predictions)
        if fmt != OutputFormat.TABLE:
            self.save(fmt, predictions=predictions, pixel_size=pixel_size)
        return predictions
