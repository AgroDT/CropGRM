import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from crop_classifier.config import ModelType
from crop_classifier.prediction.processors.make_predictions import CropPredictor


class PredictionProbabilityTests(unittest.TestCase):
    def test_missing_curve_features_raise_clear_error_before_inference(self):
        data = pd.DataFrame({"field_id": [7], "red_min": [0.1]})
        original = data.copy(deep=True)
        model = Mock()
        model.feature_names_ = ["red_min", "ndmi_S", "wrdvi_A"]
        predictor = CropPredictor()
        with patch.object(predictor, "load_model", return_value=model) as load:
            with self.assertRaisesRegex(
                ValueError, "Missing required model features: ndmi_S, wrdvi_A"
            ) as error:
                predictor.predict(data)
        self.assertIn("fewer than six observations", str(error.exception))
        self.assertIn("noisy data", str(error.exception))
        load.assert_called_once()
        model.predict_proba.assert_not_called()
        pd.testing.assert_frame_equal(data, original)

    def test_class_selection_and_probability_threshold(self):
        data = pd.DataFrame(
            {
                "field_id": ["a", "b", "c", "d", "e"],
                "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            index=[10, 20, 30, 40, 50],
        )
        original = data.copy(deep=True)

        model = Mock()
        model.feature_names_ = ["feature_a", "feature_b"]
        model.classes_ = np.array([7, 1, 13])
        model.predict_proba.return_value = np.array(
            [
                [0.20, 0.70, 0.10],
                [0.10, 0.20, 0.70],
                [0.50, 0.30, 0.20],
                [0.40, 0.35, 0.25],
                [0.80, 0.10, 0.10],
            ]
        )
        predictor = CropPredictor()

        cases = [
            (
                0.5,
                [1, 13, 0, 0, 7],
                ["winter wheat", "spring wheat", None, None, "sunflower"],
            ),
            (0.75, [0, 0, 0, 0, 7], [None, None, None, None, "sunflower"]),
        ]

        for threshold, classes, names in cases:
            with self.subTest(threshold=threshold):
                model.predict_proba.reset_mock()
                with patch.object(
                    predictor, "load_model", return_value=model
                ) as load_model:
                    result = predictor.predict(
                        data,
                        model_type=ModelType.SMALL,
                        threshold=threshold,
                    )

                load_model.assert_called_once_with(ModelType.SMALL)
                model.predict_proba.assert_called_once()
                pd.testing.assert_frame_equal(
                    model.predict_proba.call_args.args[0],
                    data[["feature_a", "feature_b"]],
                )
                self.assertEqual(result["class"].tolist(), classes)
                np.testing.assert_array_equal(
                    result["probability"], [0.7, 0.7, 0.5, 0.4, 0.8]
                )
                self.assertTrue(pd.api.types.is_integer_dtype(result["class"]))

                for actual, expected in zip(result["class_name"], names):
                    if expected is None:
                        self.assertTrue(pd.isna(actual))
                    else:
                        self.assertEqual(actual, expected)

                pd.testing.assert_frame_equal(result[data.columns], original)
                pd.testing.assert_frame_equal(data, original)
