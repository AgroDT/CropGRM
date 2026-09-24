from pathlib import Path
from typing import Union
import pandas as pd
from crop_classifier.prediction.exporters.base import BaseExporter
from crop_classifier.config import CLASS_NAMES

class TableExporter(BaseExporter):
    """Tabular result export."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        **kwargs
    ) -> None:

        cols_to_save = [col for col in ["field_id", "class_name", "class", "lat", "lon"] if col in df.columns]
        df[cols_to_save].to_csv(f'{output_prefix}.csv', index=False)