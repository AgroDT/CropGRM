import pandas as pd

from crop_classifier.config import OutputFormat
from crop_classifier.prediction.exporters.base import BaseExporter
from crop_classifier.prediction.exporters.point_raster import PointRasterExporter
from crop_classifier.prediction.exporters.polygon_raster import PolygonRasterExporter
from crop_classifier.prediction.exporters.table import TableExporter
from crop_classifier.prediction.exporters.vector import VectorExporter


class ExporterFactory:
    """Choosing export strategy."""

    @staticmethod
    def get_exporter(
        output_format: OutputFormat, use_zonal_spectral: bool, df: pd.DataFrame
    ) -> BaseExporter:
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
