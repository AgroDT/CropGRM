import pandas as pd

from cropgrm.prediction.exporters.base import BaseExporter


class TableExporter(BaseExporter):
    """Tabular result export."""

    def export(self, df: pd.DataFrame, output_prefix: str, **kwargs) -> None:
        """Write available prediction and spatial columns to CSV.

        Args:
            df (pandas.DataFrame): Table from which available field_id, class_name,
                class, lat, lon, probability, and prediction_status columns are exported.
            output_prefix (str | Path): Destination prefix; .csv is appended.
            **kwargs (object): Additional export options, ignored by this exporter.
        """
        cols_to_save = [
            col
            for col in [
                "field_id",
                "class_name",
                "class",
                "lat",
                "lon",
                "probability",
                "prediction_status",
            ]
            if col in df.columns
        ]
        df[cols_to_save].to_csv(f"{output_prefix}.csv", index=False)
