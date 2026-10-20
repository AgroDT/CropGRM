import tempfile
import unittest
from datetime import date, datetime
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

        self.assertEqual(result.schema["date"], pl.Date)
        self.assertEqual(result["year"].to_list(), [2015, 2015])
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

        self.assertEqual(result.schema["date"], pl.Date)
        self.assertEqual(result["year"].to_list(), [2015, 2015])
        self.assertEqual(result["DOY"].to_list(), [91, 93])
        self.assertEqual(result["red"].to_list(), [0.2, 0.3])
        self.assertEqual(names["field_id"].n_unique(), 1)

    def test_unknown_file_format_is_rejected(self):
        with self.assertRaises(ValueError):
            DataPreparer.read_dataset("input.txt")

    def test_calendar_columns_are_recomputed_from_normalized_dates(self):
        data = pl.DataFrame(
            {
                "date": ["2015-04-01", "2024-04-01"],
                "year": [0, 0],
                "DOY": [0, 0],
                "month": [0, 0],
            }
        )

        result = DataPreparer.add_doy_and_month(data)

        self.assertEqual(result.schema["date"], pl.Date)
        self.assertEqual(result["date"].to_list(), [date(2015, 4, 1), date(2024, 4, 1)])
        self.assertEqual(result["year"].to_list(), [2015, 2024])
        self.assertEqual(result["DOY"].to_list(), [91, 92])
        self.assertEqual(result["month"].to_list(), [4, 4])

    def test_invalid_or_missing_dates_are_rejected(self):
        for data in [
            pl.DataFrame({"red": [0.2]}),
            pl.DataFrame({"date": [None]}),
            pl.DataFrame({"date": ["not-a-date"]}),
        ]:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "date"):
                DataPreparer.add_doy_and_month(data)

    def test_pixel_timestamps_are_aggregated_by_calendar_date(self):
        data = pl.DataFrame(
            {
                "field_id": [7, 7, 7],
                "date": [
                    datetime(2024, 4, 2, 12),
                    datetime(2024, 4, 1, 8),
                    datetime(2024, 4, 1, 16),
                ],
                "red": [0.8, 0.2, 0.6],
            }
        )

        result = DataPreparer().prepare_pixel_data(data, ["red"], ["field_id"])

        self.assertEqual(result["date"].to_list(), [date(2024, 4, 1), date(2024, 4, 2)])
        self.assertEqual(result["year"].to_list(), [2024, 2024])
        self.assertEqual(result["DOY"].to_list(), [92, 93])
        self.assertAlmostEqual(result["red"][0], 0.4)
        self.assertAlmostEqual(result["red"][1], 0.8)

    def test_field_and_pixel_inputs_share_the_same_calendar(self):
        preparer = DataPreparer()
        field = pl.DataFrame({"field_id": [7], "20150401_red": [0.2]})
        pixel = pl.DataFrame({"field_id": [7], "date": ["2015-04-01"], "red": [0.2]})
        columns = ["field_id", "date", "year", "DOY", "month", "red"]

        field_result = preparer.prepare_field_data(field, ["red"], ["field_id"])
        pixel_result = preparer.prepare_pixel_data(pixel, ["red"], ["field_id"])

        self.assertEqual(
            field_result.select(columns).to_dict(as_series=False),
            pixel_result.select(columns).to_dict(as_series=False),
        )
