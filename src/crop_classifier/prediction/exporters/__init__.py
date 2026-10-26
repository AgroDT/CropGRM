import pandas as pd

from crop_classifier._optional import import_optional
from crop_classifier.config import OutputFormat
from crop_classifier.prediction.exporters.base import BaseExporter
from crop_classifier.prediction.exporters.table import TableExporter


class ExporterFactory:
    """Select an exporter for the output format and spatial data mode."""

    @staticmethod
    def get_exporter(
        output_format: OutputFormat, use_zonal_spectral: bool, df: pd.DataFrame
    ) -> BaseExporter:
        """Create an exporter matching the requested output.

        Args:
            output_format (OutputFormat): Requested table, vector, or raster format.
            use_zonal_spectral (bool): Whether raster output uses field polygons instead
                of points.
            df (pandas.DataFrame): Prediction table with class and the columns needed
                by the selected exporter.

        Returns:
            BaseExporter: New table, vector, polygon raster, or point raster exporter.

        Raises:
            ValueError: If the format is unsupported, required columns are missing,
                or a point raster table is empty.
        """
        output_format = OutputFormat(output_format)
        required = {"class"}
        if output_format == OutputFormat.VECTOR:
            required.update(("field_id", "class_name"))
        elif output_format == OutputFormat.RASTER:
            required.update(("field_id",) if use_zonal_spectral else ("lat", "lon"))
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for {output_format.value} export: "
                f"{', '.join(sorted(missing))}"
            )
        if output_format == OutputFormat.RASTER and not use_zonal_spectral and df.empty:
            raise ValueError("Point raster requires a nonempty prediction table.")

        if output_format == OutputFormat.TABLE:
            return TableExporter()

        if output_format == OutputFormat.VECTOR:
            module = import_optional(
                "crop_classifier.prediction.exporters.vector", "geo"
            )
            return module.VectorExporter()

        if output_format == OutputFormat.RASTER:
            if use_zonal_spectral:
                module = import_optional(
                    "crop_classifier.prediction.exporters.polygon_raster", "geo"
                )
                return module.PolygonRasterExporter()
            elif not use_zonal_spectral:
                module = import_optional(
                    "crop_classifier.prediction.exporters.point_raster", "geo"
                )
                return module.PointRasterExporter()

        raise ValueError(f"Unsupported export format: {output_format}")
