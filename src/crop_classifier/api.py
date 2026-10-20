from pathlib import Path
from typing import Optional, Union

from crop_classifier.config import ModelType, OutputFormat
from crop_classifier.feature.feature_pipeline import PredictorPipeline
from crop_classifier.gee_extractor.gee_pipeline import GEEDownloadingDataPipeline
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
        shapefile_path: Union[str, Path],
        gee_project: str,
        year: int,
        output_dir: Union[str, Path],
        use_zonal_spectral: bool = True,
        model_type: Union[ModelType, str] = ModelType.FINETUNED,
        threshold: float = 0.5,
        output_format: Union[OutputFormat, str] = OutputFormat.TABLE,
        epsg_code: Optional[str] = None,
    ):
        """Initialize the crop classification workflow.

        Args:
            shapefile_path (str | Path): Input shapefile defining the regions of
                interest.
            gee_project (str): Google Earth Engine project ID.
            year (int): Calendar year of the imagery and weather data.
            output_dir (str | Path): Root directory for raw data, features, and
                predictions.
            use_zonal_spectral (bool): Whether to use field statistics rather than pixel
                data.
            model_type (ModelType | str): Model variant or its configured string value.
            threshold (float): Minimum class probability to exceed for an accepted
                prediction.
            output_format (OutputFormat | str): Prediction export format or its string
                value.
            epsg_code (str | None): Output raster CRS identifier; ignored for table
                and vector exports. A CRS is required for point raster output.
        """
        self.input_file = Path(shapefile_path)
        self.project_id = gee_project
        self.year = year
        self.output_dir = Path(output_dir)
        self.use_zonal_spectral = use_zonal_spectral
        self.model_type = model_type
        self.threshold = threshold
        self.output_format = output_format
        self.epsg_code = epsg_code

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Properties for Default File Paths ---

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
        return (
            self.output_dir
            / "raw"
            / f"{self.input_file.stem}_spectral.{self._file_ext}"
        )

    @property
    def default_meteo_path(self) -> Path:
        """Get the default path for raw meteorological data.

        Returns:
            Path: Weather CSV or Parquet path under the raw directory.
        """
        return (
            self.output_dir / "raw" / f"{self.input_file.stem}_meteo.{self._file_ext}"
        )

    @property
    def default_processed_path(self) -> Path:
        """Get the default path for the feature dataset.

        Returns:
            Path: Feature Parquet path under the processed directory.
        """
        return self.output_dir / "processed" / f"{self.input_file.stem}_input.parquet"

    @property
    def default_prediction_prefix(self) -> Path:
        """Get the default prefix for prediction files.

        Returns:
            Path: Output path prefix under the final directory, without an extension.
        """
        return self.output_dir / "final" / f"{self.input_file.stem}"

    # --- Pipeline Steps ---

    def download_data(self) -> None:
        """Download spectral and meteorological observations from Earth Engine."""
        GEEDownloadingDataPipeline(
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
        """
        src_spectral = (
            Path(spectral_path) if spectral_path else self.default_spectral_path
        )
        src_meteo = Path(meteo_path) if meteo_path else self.default_meteo_path
        dst_path = Path(output_path) if output_path else self.default_processed_path

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        PredictorPipeline().build_features(
            self.use_zonal_spectral, src_spectral, src_meteo, dst_path
        )

        return dst_path

    def predict(
        self,
        processed_path: Optional[Union[str, Path]] = None,
        output_prefix: Optional[Union[str, Path]] = None,
        input_epsg_code: Optional[str] = "EPSG:4326",
        output_epsg_code: Optional[str] = None,
    ) -> None:
        """Classify prepared features and export predictions.

        Args:
            processed_path (str | Path | None): Feature Parquet path; None uses
                default_processed_path.
            output_prefix (str | Path | None): Output prefix without an extension; None
                uses default_prediction_prefix.
            input_epsg_code (str | None): CRS of pixel coordinates, passed to the
                exporter.
            output_epsg_code (str | None): Output raster CRS override; None uses the
                configured epsg_code. Ignored for table and vector exports.
        """
        src_data = (
            Path(processed_path) if processed_path else self.default_processed_path
        )
        dst_prefix = (
            Path(output_prefix) if output_prefix else self.default_prediction_prefix
        )

        dst_prefix.parent.mkdir(parents=True, exist_ok=True)
        fields_geometry = self.input_file if self.use_zonal_spectral else None
        epsg_code = output_epsg_code or self.epsg_code

        CropClassifierPipeline().run(
            input_data=src_data,
            output_prefix=dst_prefix,
            model_type=self.model_type,
            use_zonal_spectral=self.use_zonal_spectral,
            threshold=self.threshold,
            output_format=self.output_format,
            fields_geometry_path=fields_geometry,
            input_epsg_code=input_epsg_code,
            output_epsg_code=epsg_code,
        )

    # --- Complete Pipeline Execution ---

    def run(self) -> None:
        """Download data, build features, and export crop predictions."""
        self.download_data()
        self.build_features()
        self.predict()
