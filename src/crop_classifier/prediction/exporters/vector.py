from pathlib import Path
from typing import Dict, Union
import geopandas as gpd
import pandas as pd
from crop_classifier.prediction.exporters.base import BaseExporter
from crop_classifier.config import CLASS_NAMES


class VectorExporter(BaseExporter):
    """Vector data export."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        fields_geometry_path: Union[str, Path] = None,
        **kwargs
    ) -> None:
        if not fields_geometry_path or class_names is None:
            raise ValueError("For vector exporting 'fields_geometry_path' required.")

        gdf = (
            gpd.read_file(fields_geometry_path)
            .merge(df[["field_id", "class", "class_name"]], on="field_id", how="left")
        )
        
        gdf[["field_id", "class_name", "geometry"]].to_file(f'{output_prefix}.fgb', encoding="utf8")