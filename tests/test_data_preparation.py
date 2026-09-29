import tempfile
import unittest
from pathlib import Path

import polars as pl

from crop_classifier.feature.processors.data_preparer import DataPreparer


class DataPreparationTests(unittest.TestCase):
    def test_csv_is_read_and_wide_field_data_becomes_daily_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "spectral.csv"
            pl.DataFrame(
                {
                    "field_id": [7],
                    "20150401_red": [0.2],
                    "20150401_nir": [0.8],
                    "20150403_red": [0.3],
                    "20150403_nir": [0.7],
                }
            ).write_csv(csv_path)

            preparer = DataPreparer()
            data = preparer.read_dataset(csv_path).collect()
            result = preparer.prepare_data_chunk(
                preparer.ensure_field_id(data, ["field_id"]),
                ["red", "nir"],
                ["field_id"],
            )

        self.assertEqual(result["DOY"].to_list(), [91, 93])
        self.assertEqual(result["month"].to_list(), [4, 4])
        self.assertEqual(result["red"].to_list(), [0.2, 0.3])
        self.assertEqual(result["nir"].to_list(), [0.8, 0.7])
        self.assertEqual(result["field_id"].to_list(), [7, 7])

    def test_pixel_rows_receive_spatial_id_and_calendar_features(self):
        data = pl.DataFrame(
            {
                "lat": [51.0, 51.0],
                "lon": [40.0, 40.0],
                "date": ["2015-04-03", "2015-04-01"],
                "red": [0.3, 0.2],
            }
        )
        preparer = DataPreparer()
        result = preparer.prepare_data_chunk(
            data,
            ["red"],
            ["lat", "lon"],
        )

        names = preparer.ensure_field_id(data, ["lat", "lon"])

        self.assertEqual(result["DOY"].to_list(), [91, 93])
        self.assertEqual(result["red"].to_list(), [0.2, 0.3])
        self.assertEqual(names["field_id"].n_unique(), 1)

    def test_unknown_file_format_is_rejected(self):
        with self.assertRaises(ValueError):
            DataPreparer.read_dataset("input.txt")
