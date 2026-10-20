from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from crop_classifier.prediction.exporters.base import BaseExporter


class VectorExporter(BaseExporter):
    """Vector data export."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        fields_geometry_path: Union[str, Path] = None,
        **kwargs,
    ) -> None:
        """Join crop labels to field geometries and write FlatGeobuf.

        Args:
            df (pandas.DataFrame): Predictions containing field_id, class, and
                class_name.
            output_prefix (str | Path): Destination prefix; .fgb is appended.
            fields_geometry_path (str | Path | None): Vector file containing geometry
                and field_id; a path is required.
            **kwargs (object): Additional export options, ignored by this exporter.

        Raises:
            ValueError: If fields_geometry_path is missing.
        """
        if not fields_geometry_path:
            raise ValueError("For vector exporting 'fields_geometry_path' required.")

        gdf = gpd.read_file(fields_geometry_path).merge(
            df[["field_id", "class", "class_name"]], on="field_id", how="left"
        )

        gdf[["field_id", "class_name", "geometry"]].to_file(
            f"{output_prefix}.fgb", encoding="utf8"
        )
