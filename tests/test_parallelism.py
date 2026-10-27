import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from cropgrm import CropClassifier


class PublicParallelismTests(unittest.TestCase):
    def test_default_and_explicit_worker_counts(self):
        self.assertEqual(CropClassifier().n_jobs, 1)
        for workers in (1, 2, -1, -2):
            with self.subTest(n_jobs=workers):
                self.assertEqual(CropClassifier(n_jobs=workers).n_jobs, workers)

    def test_invalid_worker_count_fails_without_importing_optional_capabilities(self):
        for workers in (0, True, False, 1.5, "2", None):
            with self.subTest(n_jobs=workers):
                with (
                    patch("cropgrm.api.import_optional") as optional,
                    self.assertRaisesRegex(ValueError, "n_jobs.*nonzero integer"),
                ):
                    CropClassifier(n_jobs=workers)
                optional.assert_not_called()

    def test_build_features_and_run_pass_worker_count_to_feature_pipeline(self):
        for operation in ("build_features", "run"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as folder,
            ):
                classifier = CropClassifier(output_dir=folder, n_jobs=3)
                predictions = pd.DataFrame({"class": [1]})
                feature_module = Mock()
                with (
                    patch(
                        "cropgrm.api.import_optional",
                        return_value=feature_module,
                    ),
                    patch.object(classifier, "download_data"),
                    patch("pandas.read_parquet", return_value=pd.DataFrame()) as read,
                    patch.object(
                        classifier, "predict", return_value=predictions
                    ) as predict,
                    patch.object(classifier, "save"),
                ):
                    result = getattr(classifier, operation)()
                feature_module.PredictorPipeline.assert_called_once_with(n_jobs=3)
                feature_module.PredictorPipeline.return_value.build_features.assert_called_once()
                if operation == "run":
                    read.assert_called_once_with(classifier.default_processed_path)
                    predict.assert_called_once_with(read.return_value)
                    self.assertIs(result, predictions)
                else:
                    self.assertEqual(
                        result, Path(folder) / "processed/predictions_input.parquet"
                    )
                    read.assert_not_called()
                    predict.assert_not_called()


class FeatureParallelismTests(unittest.TestCase):
    def test_worker_count_reaches_joblib_for_every_processor(self):
        from cropgrm.feature.feature_pipeline import PredictorPipeline

        for workers in (1, 3, -1):
            pipeline = PredictorPipeline(n_jobs=workers)
            for processor in ("mean", "curve", "extrema"):
                with self.subTest(n_jobs=workers, processor=processor):
                    with patch(
                        "cropgrm.feature.processors.spectral_calculator.Parallel"
                    ) as parallel:
                        parallel.return_value.return_value = []
                        pipeline.spectral_extractor._run_pipeline(processor, "ndmi", [])
                    parallel.assert_called_once_with(n_jobs=workers)

    def test_direct_feature_constructors_validate_worker_count(self):
        from cropgrm.feature.feature_pipeline import PredictorPipeline
        from cropgrm.feature.processors.spectral_calculator import (
            SpectralFeatureCalculator,
        )

        for constructor in (PredictorPipeline, SpectralFeatureCalculator):
            for workers in (0, True, 1.5, "2", None):
                with self.subTest(constructor=constructor, n_jobs=workers):
                    with self.assertRaisesRegex(ValueError, "n_jobs.*nonzero integer"):
                        constructor(n_jobs=workers)

    def test_two_workers_produce_the_same_features_as_sequential_processing(self):
        import polars as pl
        from polars.testing import assert_frame_equal

        from cropgrm.feature.processors.spectral_calculator import (
            SpectralFeatureCalculator,
        )

        data = pl.DataFrame(
            {
                "field_id": ["001"] * 3 + ["002"] * 3,
                "year": [2024] * 6,
                "DOY": [100, 130, 160] * 2,
                "red": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
                "blue": [0.1] * 6,
                "green": [0.2] * 6,
                "nir": [0.8] * 6,
                "swir1": [0.2] * 6,
                "swir2": [0.3] * 6,
            }
        )
        results = []
        for workers in (1, 2):
            calculator = SpectralFeatureCalculator(n_jobs=workers)
            calculator.feature_config = {
                "red": {"pipeline": "mean"},
                "ndyi": {"pipeline": "extrema"},
            }
            results.append(
                calculator.calculate_spectral_features(data).sort("field_id")
            )
        assert_frame_equal(results[0], results[1])
