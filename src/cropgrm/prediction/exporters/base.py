# exporters/base.py
from abc import ABC, abstractmethod

import pandas as pd


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
