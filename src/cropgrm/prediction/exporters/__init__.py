import pandas as pd

from cropgrm._optional import import_optional
from cropgrm.config import OutputFormat
from cropgrm.prediction.exporters.base import BaseExporter
from cropgrm.prediction.exporters.table import TableExporter


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
        return ExporterFactory._get_exporter(
            OutputFormat(output_format), use_zonal_spectral, df
        )

    @staticmethod
    def _get_exporter(
        output_format: OutputFormat, use_zonal_spectral: bool, df: pd.DataFrame
    ) -> BaseExporter:
        """Select an exporter after public format normalization."""
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

        module_name, class_name = {
            OutputFormat.VECTOR: ("vector", "VectorExporter"),
            OutputFormat.RASTER: (
                ("polygon_raster", "PolygonRasterExporter")
                if use_zonal_spectral
                else ("point_raster", "PointRasterExporter")
            ),
        }[output_format]
        module = import_optional(f"cropgrm.prediction.exporters.{module_name}", "geo")
        return getattr(module, class_name)()
