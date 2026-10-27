# exporters/base.py
from abc import ABC, abstractmethod
from math import isfinite

import pandas as pd


def validate_pixel_size(pixel_size: float) -> None:
    """Require finite positive raster spacing at a public entry point."""
    if not isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive.")


def validate_geometry_path(fields_geometry_path) -> None:
    """Require a geometry path for field exports at a public entry point."""
    if not fields_geometry_path:
        raise ValueError("fields_geometry_path is required for field geometry export.")


class BaseExporter(ABC):
    """Define the interface for exporting crop predictions."""

    @abstractmethod
    def export(self, df: pd.DataFrame, output_prefix: str, **kwargs) -> None:
        """Export a prediction table to the chosen destination.

        Args:
            df (pandas.DataFrame): Predictions with spatial keys and class columns
                required by the concrete exporter.
            output_prefix (str | Path): Destination path prefix without a file
                extension.
            **kwargs (object): Exporter-specific options, such as geometry paths or CRS
                identifiers.
        """
        pass

    def _export_prepared(self, df: pd.DataFrame, output_prefix: str, **kwargs) -> None:
        """Write prepared inputs; simple exporters need no separate validation."""
        self.export(df, output_prefix, **kwargs)
