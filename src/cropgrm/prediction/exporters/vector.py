import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from cropgrm.prediction.exporters.base import BaseExporter, validate_geometry_path

logger = logging.getLogger(__name__)


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
        validate_geometry_path(fields_geometry_path)
        self._export_prepared(df, output_prefix, fields_geometry_path)

    def _export_prepared(self, df, output_prefix, fields_geometry_path, **kwargs):
        """Join and write fields using validated export options."""
        gdf = gpd.read_file(fields_geometry_path)
        if "field_id" not in gdf.columns or "geometry" not in gdf.columns:
            raise ValueError("Field geometry requires field_id and geometry columns.")
        if (
            gdf.empty
            or not (
                gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid
            ).any()
        ):
            raise ValueError("At least one usable geometry is required.")
        unmatched = (
            ~gdf["field_id"].astype(str).isin(df["field_id"].astype(str))
        ).sum()
        if unmatched:
            logger.warning(
                "%d field geometries have no matching prediction field_id; "
                "their vector labels will be missing.",
                unmatched,
            )
        gdf = gdf.assign(field_id=gdf["field_id"].astype(str)).merge(
            df[["field_id", "class", "class_name"]].assign(
                field_id=df["field_id"].astype(str)
            ),
            on="field_id",
            how="left",
        )

        gdf[["field_id", "class_name", "geometry"]].to_file(
            f"{output_prefix}.fgb", encoding="utf8"
        )
