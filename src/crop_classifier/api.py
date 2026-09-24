from pathlib import Path
from typing import Union, Optional

from crop_classifier import ModelType, OutputFormat
from crop_classifier.gee_extractor.gee_pipeline import GEEDownloadingDataPipeline
from crop_classifier.feature.feature_pipeline import PredictorPipeline
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
        epsg_code: str = "EPSG:32637",
    ):
        """Initialize the CropClassifier pipeline.

        Args:
            shapefile_path: Path to the input shapefile specifying regions of interest.
            gee_project: Google Earth Engine project ID or cloud project identifier.
            year: Target year for satellite imagery and weather data extraction.
            output_dir: Root directory path where raw, processed, and final files will be saved.
            use_zonal_spectral: If True, uses zonal statistics (CSV); if False, pixel-based extraction (Parquet). Defaults to True.
            model_type: Type of model to use for inference. Defaults to ModelType.FINETUNED.
            threshold: Classification probability threshold. Defaults to 0.5.
            output_format: Output format for final predictions (e.g., OutputFormat.TABLE or OutputFormat.RASTER). Defaults to OutputFormat.TABLE.
            epsg_code: Target EPSG coordinate system code. Defaults to "EPSG:32637".
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
        """Get raw data file extension based on spectral mode."""
        return "csv" if self.use_zonal_spectral else "parquet"

    @property
    def default_spectral_path(self) -> Path:
        """Get default path for downloaded raw spectral data."""
        return self.output_dir / "raw" / f"{self.input_file.stem}_spectral.{self._file_ext}"

    @property
    def default_meteo_path(self) -> Path:
        """Get default path for downloaded raw meteorological data."""
        return self.output_dir / "raw" / f"{self.input_file.stem}_meteo.{self._file_ext}"

    @property
    def default_processed_path(self) -> Path:
        """Get default path for engineered feature dataset."""
        return self.output_dir / "processed" / f"{self.input_file.stem}_input.parquet"

    @property
    def default_prediction_prefix(self) -> Path:
        """Get default prefix for prediction output files."""
        return self.output_dir / "final" / f"{self.input_file.stem}"

    # --- Pipeline Steps ---

    def download_data(self) -> None:
        """Step 1: Download raw satellite spectral data and meteorological metrics from GEE."""
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
        """Step 2: Generate features from raw spectral and meteorological datasets.

        Args:
            spectral_path: Custom path to raw spectral data. Defaults to default_spectral_path.
            meteo_path: Custom path to raw meteorological data. Defaults to default_meteo_path.
            output_path: Custom destination path for processed features. Defaults to default_processed_path.

        Returns:
            Path: Path to the generated feature Parquet file.
        """
        src_spectral = Path(spectral_path) if spectral_path else self.default_spectral_path
        src_meteo = Path(meteo_path) if meteo_path else self.default_meteo_path
        dst_path = Path(output_path) if output_path else self.default_processed_path

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        PredictorPipeline().build_features(src_spectral, src_meteo, dst_path)
        
        return dst_path

    def predict(
        self,
        processed_path: Optional[Union[str, Path]] = None,
        output_prefix: Optional[Union[str, Path]] = None,
    ) -> None:
        """Step 3: Run crop classification inference and export the final output.

        Args:
            processed_path: Custom path to engineered feature file. Defaults to default_processed_path.
            output_prefix: Custom output path prefix for prediction artifacts. Defaults to default_prediction_prefix.
        """
        src_data = Path(processed_path) if processed_path else self.default_processed_path
        dst_prefix = Path(output_prefix) if output_prefix else self.default_prediction_prefix

        dst_prefix.parent.mkdir(parents=True, exist_ok=True)
        fields_geometry = self.input_file if self.use_zonal_spectral else None

        CropClassifierPipeline().run(
            input_data=src_data,
            output_prefix=dst_prefix,
            model_type=self.model_type,
            threshold=self.threshold,
            output_format=self.output_format,
            fields_geometry_path=fields_geometry,
            epsg_code=self.epsg_code,
        )

    # --- Complete Pipeline Execution ---

    def run(self) -> None:
        """Execute the entire pipeline sequentially: download_data -> build_features -> predict."""
        self.download_data()
        self.build_features()
        self.predict()