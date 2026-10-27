import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from cropgrm.config import BASIC_INDICES, METEO_INDICES
from cropgrm.feature.feature_pipeline import PredictorPipeline


class FeatureInputValidationTests(unittest.TestCase):
    def test_mixed_or_mismatched_years_are_rejected_before_feature_calculation(self):
        for wide in (False, True):
            for spectral_years, meteo_years in (
                ([2023, 2024], [2024, 2024]),
                ([2024, 2024], [2023, 2024]),
                ([2023, 2023], [2024, 2024]),
            ):
                with (
                    self.subTest(wide=wide, spectral=spectral_years, meteo=meteo_years),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    paths = [root / "spectral.csv", root / "meteo.parquet"]
                    for path, years, bands in zip(
                        paths,
                        [spectral_years, meteo_years],
                        [BASIC_INDICES, METEO_INDICES],
                    ):
                        if wide:
                            data = {"field_id": [1, 2]}
                            for year in set(years):
                                for band in bands:
                                    data[f"{year}0401_{band}"] = [
                                        0.2 if observed_year == year else None
                                        for observed_year in years
                                    ]
                        else:
                            data = {
                                "field_id": [1, 2],
                                "date": [f"{year}-04-01" for year in years],
                                **{band: [0.2, 0.2] for band in bands},
                            }
                        frame = pl.DataFrame(data)
                        if path.suffix == ".csv":
                            frame.write_csv(path)
                        else:
                            frame.write_parquet(path)
                    output = root / "features.parquet"
                    output.write_bytes(b"previous output")
                    pipeline = PredictorPipeline()
                    pipeline.chunk_size = 1
                    with (
                        patch.object(
                            pipeline.spectral_extractor, "calculate_spectral_features"
                        ) as spectral,
                        patch.object(
                            pipeline.meteo_extractor, "calculate_meteo_features"
                        ) as meteo,
                        self.assertRaisesRegex(ValueError, "one common calendar year"),
                    ):
                        pipeline.build_features(True, *paths, output)
                    spectral.assert_not_called()
                    meteo.assert_not_called()
                    self.assertEqual(output.read_bytes(), b"previous output")

    def write_input(self, path, bands, wide=False, zonal=True):
        data = {"field_id": [7]} if zonal else {"lat": [51.0], "lon": [40.0]}
        if not wide:
            data["date"] = ["2024-04-01"]
        for band in bands:
            name = f"source_20240401_{band}" if wide else band
            data[name] = [0.2]
        data["extra"] = [42]
        frame = pl.DataFrame(data)
        if path.suffix == ".csv":
            frame.write_csv(path)
        else:
            frame.write_parquet(path)

    def assert_rejected(self, spectral, meteo, output, role, missing, detected):
        pipeline = PredictorPipeline()
        original_inputs = [path.read_bytes() for path in (spectral, meteo)]
        original_output = output.read_bytes() if output.exists() else None
        with (
            patch.object(pipeline.preparer, "ensure_field_id") as spatial_ids,
            patch.object(pipeline.preparer, "prepare_data_chunk") as prepare,
            patch.object(
                pipeline.spectral_extractor, "calculate_spectral_features"
            ) as spectral_features,
            patch.object(
                pipeline.meteo_extractor, "calculate_meteo_features"
            ) as meteo_features,
            patch(
                "cropgrm.feature.processors.spectral_calculator.add_indices"
            ) as indices,
            self.assertRaises(ValueError) as error,
        ):
            pipeline.build_features(True, spectral, meteo, output)

        message = str(error.exception)
        path = spectral if role == "spectral" else meteo
        self.assertIn(f"Invalid {role} input '{path}'", message)
        self.assertIn(f"missing required bands: {sorted(missing)}", message)
        self.assertIn(f"detected bands: {sorted(detected)}", message)
        for mock in (spatial_ids, prepare, spectral_features, meteo_features, indices):
            mock.assert_not_called()
        if original_output is None:
            self.assertFalse(output.exists())
        else:
            self.assertEqual(output.read_bytes(), original_output)
        self.assertEqual(
            [path.read_bytes() for path in (spectral, meteo)], original_inputs
        )

    def test_valid_inputs_allow_extra_columns_in_all_supported_structures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".csv", ".parquet"):
                for wide in (False, True):
                    for zonal in (False, True):
                        with self.subTest(suffix=suffix, wide=wide, zonal=zonal):
                            # Deliberately misleading filenames must be ignored.
                            spectral = root / f"weather{suffix}"
                            meteo = root / f"reflectance{suffix}"
                            output = root / "features.parquet"
                            self.write_input(spectral, BASIC_INDICES, wide, zonal)
                            self.write_input(meteo, METEO_INDICES, wide, zonal)
                            pipeline = PredictorPipeline()
                            with (
                                patch.object(
                                    pipeline.spectral_extractor,
                                    "calculate_spectral_features",
                                    return_value=pl.DataFrame(
                                        {"field_id": [7], "spectral_feature": [1.0]}
                                    ),
                                ) as spectral_features,
                                patch.object(
                                    pipeline.meteo_extractor,
                                    "calculate_meteo_features",
                                    return_value=pl.DataFrame(
                                        {"field_id": [7], "meteo_feature": [2.0]}
                                    ),
                                ) as meteo_features,
                            ):
                                pipeline.build_features(zonal, spectral, meteo, output)
                            spectral_features.assert_called_once()
                            meteo_features.assert_called_once()
                            self.assertTrue(output.exists())
                            self.assertTrue(
                                set(BASIC_INDICES).issubset(
                                    spectral_features.call_args.args[0].columns
                                )
                            )

    def test_swapped_files_are_rejected_without_touching_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".csv", ".parquet"):
                for wide in (False, True):
                    for existing in (False, True):
                        with self.subTest(suffix=suffix, wide=wide, existing=existing):
                            spectral = root / f"spectral{suffix}"
                            meteo = root / f"meteo{suffix}"
                            output = root / f"output-{suffix}-{wide}-{existing}.parquet"
                            self.write_input(spectral, METEO_INDICES, wide)
                            self.write_input(meteo, BASIC_INDICES, wide)
                            if existing:
                                output.write_bytes(b"existing output")
                            self.assert_rejected(
                                spectral,
                                meteo,
                                output,
                                "spectral",
                                BASIC_INDICES,
                                METEO_INDICES,
                            )

    def test_each_required_band_is_checked_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".csv", ".parquet"):
                for wide in (False, True):
                    for role, bands in (
                        ("spectral", BASIC_INDICES),
                        ("meteorological", METEO_INDICES),
                    ):
                        for missing in bands:
                            with self.subTest(
                                suffix=suffix, wide=wide, role=role, missing=missing
                            ):
                                spectral = root / f"spectral{suffix}"
                                meteo = root / f"meteo{suffix}"
                                output = root / "output.parquet"
                                remaining = [b for b in bands if b != missing]
                                self.write_input(
                                    spectral,
                                    remaining if role == "spectral" else BASIC_INDICES,
                                    wide,
                                )
                                self.write_input(
                                    meteo,
                                    remaining
                                    if role == "meteorological"
                                    else METEO_INDICES,
                                    wide,
                                )
                                self.assert_rejected(
                                    spectral, meteo, output, role, [missing], remaining
                                )

    def test_service_columns_only_are_rejected_for_both_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for wide in (False, True):
                for role, bands in (
                    ("spectral", BASIC_INDICES),
                    ("meteorological", METEO_INDICES),
                ):
                    with self.subTest(wide=wide, role=role):
                        spectral = root / "spectral.csv"
                        meteo = root / "meteo.parquet"
                        output = root / "output.parquet"
                        self.write_input(
                            spectral, [] if role == "spectral" else BASIC_INDICES, wide
                        )
                        self.write_input(
                            meteo,
                            [] if role == "meteorological" else METEO_INDICES,
                            wide,
                        )
                        output.write_bytes(b"preserve this output")
                        self.assert_rejected(spectral, meteo, output, role, bands, [])

    def test_band_substrings_do_not_count_as_required_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for wide in (False, True):
                with self.subTest(wide=wide):
                    spectral = root / "spectral.csv"
                    meteo = root / "meteo.csv"
                    remaining = [b for b in BASIC_INDICES if b != "green"]
                    self.write_input(spectral, [*remaining, "green_backup"], wide)
                    self.write_input(meteo, METEO_INDICES, wide)
                    self.assert_rejected(
                        spectral,
                        meteo,
                        root / "output.parquet",
                        "spectral",
                        ["green"],
                        remaining,
                    )
