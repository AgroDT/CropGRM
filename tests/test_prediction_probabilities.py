import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from cropgrm import CropClassifier
from cropgrm.config import ModelType
from cropgrm.prediction.prediction_pipeline import CropClassifierPipeline
from cropgrm.prediction.processors.make_predictions import CropPredictor


class PredictionProbabilityTests(unittest.TestCase):
    def test_input_result_and_cache_are_independent(self):
        data = pd.DataFrame({"field_id": ["001", "002"], "feature": [0.2, 0.8]})
        original = data.copy(deep=True)
        model = Mock()
        model.feature_names_ = ["feature"]
        model.classes_ = np.array([1, 7])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        classifier = CropClassifier()
        with patch.object(CropPredictor, "load_model", return_value=model):
            pipeline_result = CropClassifierPipeline().run(data)
            result = classifier.predict(data)
        pd.testing.assert_frame_equal(data, original)
        cached = classifier._predictions.copy(deep=True)
        data.loc[0, "feature"] = -100.0
        self.assertEqual(pipeline_result.loc[0, "feature"], 0.2)
        self.assertEqual(result.loc[0, "feature"], 0.2)
        pipeline_result.loc[1, "feature"] = -200.0
        self.assertEqual(data.loc[1, "feature"], 0.8)
        self.assertEqual(result.loc[1, "feature"], 0.8)
        result.loc[0, "class"] = 7
        result.loc[0, "feature"] = -300.0
        pd.testing.assert_frame_equal(classifier._predictions, cached)

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
