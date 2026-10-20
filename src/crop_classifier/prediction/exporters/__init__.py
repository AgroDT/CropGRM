import pandas as pd

from crop_classifier.config import OutputFormat
from crop_classifier.prediction.exporters.base import BaseExporter
from crop_classifier.prediction.exporters.point_raster import PointRasterExporter
from crop_classifier.prediction.exporters.polygon_raster import PolygonRasterExporter
from crop_classifier.prediction.exporters.table import TableExporter
from crop_classifier.prediction.exporters.vector import VectorExporter


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
            df (pandas.DataFrame): Prediction table; vector output requires a field_id
                column.

        Returns:
            BaseExporter: New table, vector, polygon raster, or point raster exporter.

        Raises:
            ValueError: If the format is unsupported or vector output lacks field_id.
        """
        if output_format == OutputFormat.TABLE:
            return TableExporter()

        if output_format == OutputFormat.VECTOR:
            if "field_id" not in df.columns:
                raise ValueError("For vector export column 'field_id' is required.")
            return VectorExporter()

        if output_format == OutputFormat.RASTER:
            if use_zonal_spectral:
                return PolygonRasterExporter()
            elif not use_zonal_spectral:
                return PointRasterExporter()

        raise ValueError(f"Unsupported export format: {output_format}")
