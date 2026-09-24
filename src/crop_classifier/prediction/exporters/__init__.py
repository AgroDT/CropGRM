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
    def get_exporter(output_format: OutputFormat, df: pd.DataFrame) -> BaseExporter:
        if output_format == OutputFormat.TABLE:
            return TableExporter()

        if output_format == OutputFormat.VECTOR:
            if "field_id" not in df.columns:
                raise ValueError("For vector export column 'field_id' is required.")
            return VectorExporter()

        if output_format == OutputFormat.RASTER:
            if "field_id" in df.columns:
                return PolygonRasterExporter()
            elif "lat" in df.columns and "lon" in df.columns:
                return PointRasterExporter()
            else:
                raise ValueError("For ratser export columns 'field_id' or 'lat'/'lon' are required.")

        raise ValueError(f"Unsupported export format: {output_format}")