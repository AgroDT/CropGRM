import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from cropgrm import CropClassifier
from cropgrm.config import ModelType
from cropgrm.prediction.prediction_pipeline import CropClassifierPipeline
from cropgrm.prediction.processors.make_predictions import CropPredictor


class PredictionProbabilityTests(unittest.TestCase):
    def test_fit_checks_follow_model_feature_names(self):
        data = pd.DataFrame(
            {
                "ndmi_S": [0.8, 0.5, 0.3, 0.9, 0.6],
                "wrdvi_A": [0.2, 0.7, 0.5, 0.9, 0.6],
                "median_ndmi_fitted_4": [0.1] * 5,
                "median_wrdvi_fitted_7": [0.2] * 5,
                "red_min": [0.1] * 5,
                "ndmi_fit_status": [
                    "success",
                    "success",
                    "fit_failed",
                    "fit_failed",
                    "success",
                ],
                "wrdvi_fit_status": [
                    "success",
                    "fit_failed",
                    "success",
                    "fit_failed",
                    "insufficient_observations",
                ],
            },
            index=[10, 20, 30, 40, 50],
        )
        original = data.copy(deep=True)
        cases = [
            (["ndmi_S"], [0, 1, 4]),
            (["median_ndmi_fitted_4"], [0, 1, 4]),
            (["wrdvi_A"], [0, 2]),
            (["median_wrdvi_fitted_7"], [0, 2]),
            (["ndmi_S", "wrdvi_A"], [0]),
            (["red_min"], [0, 1, 2, 3, 4]),
        ]
        for feature_names, allowed in cases:
            with self.subTest(feature_names=feature_names):
                model = Mock(feature_names_=feature_names, classes_=np.array([1, 7]))
                model.predict_proba.side_effect = lambda frame: np.column_stack(
                    (frame.iloc[:, 0].to_numpy(), 1 - frame.iloc[:, 0].to_numpy())
                )
                with patch.object(CropPredictor, "load_model", return_value=model):
                    baseline = CropPredictor().predict(
                        data.drop(columns=["ndmi_fit_status", "wrdvi_fit_status"])
                    )
                    result = CropPredictor().predict(data)
                pd.testing.assert_frame_equal(
                    result.iloc[allowed][
                        ["class", "class_name", "probability", "prediction_status"]
                    ],
                    baseline.iloc[allowed][
                        ["class", "class_name", "probability", "prediction_status"]
                    ],
                )
                pd.testing.assert_frame_equal(
                    model.predict_proba.call_args.args[0],
                    data.iloc[allowed][feature_names],
                )
                blocked = result.iloc[[i for i in range(len(data)) if i not in allowed]]
                self.assertTrue(blocked["class"].eq(0).all())
                self.assertTrue(blocked["class_name"].isna().all())
                self.assertTrue(blocked["probability"].isna().all())
                pd.testing.assert_frame_equal(result[data.columns], original)

    def test_unused_status_is_ignored_and_only_used_status_is_required(self):
        model = Mock(feature_names_=["ndmi_S"], classes_=np.array([1, 7]))
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        data = pd.DataFrame({"ndmi_S": [0.1], "ndmi_fit_status": ["success"]})
        with patch.object(CropPredictor, "load_model", return_value=model):
            baseline = CropPredictor().predict(data.drop(columns="ndmi_fit_status"))
            result = CropPredictor().predict(data)
            pd.testing.assert_frame_equal(result[baseline.columns], baseline)
            result = CropPredictor().predict(data.assign(wrdvi_fit_status="invalid"))
            pd.testing.assert_frame_equal(result[baseline.columns], baseline)
            with self.assertRaisesRegex(
                ValueError, "Missing required curve fit statuses: ndmi_fit_status"
            ):
                CropPredictor().predict(
                    data.drop(columns="ndmi_fit_status").assign(
                        wrdvi_fit_status="fit_failed"
                    )
                )

    def test_unavailable_feature_names_raise_clear_error(self):
        for feature_names in (None, [], [""], [None]):
            with self.subTest(feature_names=feature_names):
                model = Mock(feature_names_=feature_names)
                with patch.object(CropPredictor, "load_model", return_value=model):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Model input feature names are unavailable or invalid",
                    ):
                        CropPredictor().predict(pd.DataFrame({"feature": [0.1]}))
                model.predict_proba.assert_not_called()
        model = Mock(spec=["predict_proba"])
        with patch.object(CropPredictor, "load_model", return_value=model):
            with self.assertRaisesRegex(
                ValueError, "Model input feature names are unavailable or invalid"
            ):
                CropPredictor().predict(pd.DataFrame({"feature": [0.1]}))
        model.predict_proba.assert_not_called()

    def test_missing_model_features_are_rejected_even_for_blocked_rows(self):
        model = Mock(feature_names_=["ndmi_S", "red_min"])
        data = pd.DataFrame({"ndmi_S": [0.1], "ndmi_fit_status": ["fit_failed"]})
        with patch.object(CropPredictor, "load_model", return_value=model):
            with self.assertRaisesRegex(
                ValueError, "Missing required model features: red_min"
            ):
                CropPredictor().predict(data)
        model.predict_proba.assert_not_called()

    def test_bundled_models_ignore_unused_wrdvi_failure(self):
        path = (
            Path(__file__).resolve().parents[1] / "data/processed/fields_input.parquet"
        )
        data = (
            pd.read_parquet(path)
            .iloc[:1]
            .drop(columns=["ndmi_fit_status", "wrdvi_fit_status"], errors="ignore")
        )
        predictor = CropPredictor()
        for model_type in (ModelType.SMALL, ModelType.FINETUNED):
            with self.subTest(model_type=model_type):
                baseline = predictor.predict(data, model_type=model_type)
                result = predictor.predict(
                    data.assign(
                        ndmi_fit_status="success", wrdvi_fit_status="fit_failed"
                    ),
                    model_type=model_type,
                )
                pd.testing.assert_frame_equal(result[baseline.columns], baseline)

    def test_failed_fit_rows_are_preserved_and_never_sent_to_model(self):
        data = pd.DataFrame(
            {
                "ndmi_S": [0.1, 0.2, np.nan, 0.4],
                "wrdvi_A": [0.1, 0.2, 0.3, 0.4],
                "ndmi_fit_status": ["success", "success", "fit_failed", "success"],
                "wrdvi_fit_status": [
                    "success",
                    "fit_failed",
                    "success",
                    "insufficient_observations",
                ],
            },
            index=[9, 9, 3, 4],
        )
        model = Mock(feature_names_=["ndmi_S", "wrdvi_A"], classes_=np.array([1, 7]))
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        for frame in (data, data.iloc[1:]):
            with self.subTest(all_failed=len(frame) == 3):
                model.predict_proba.reset_mock()
                with patch.object(CropPredictor, "load_model", return_value=model):
                    result = CropPredictor().predict(frame)
                if len(frame) == 4:
                    pd.testing.assert_frame_equal(
                        model.predict_proba.call_args.args[0],
                        frame.iloc[:1][model.feature_names_],
                    )
                    self.assertEqual(result["class"].tolist(), [1, 0, 0, 0])
                else:
                    model.predict_proba.assert_not_called()
                self.assertTrue(result.iloc[-3:]["probability"].isna().all())
                self.assertEqual(
                    result.iloc[-3:]["prediction_status"].tolist(),
                    ["fit_failed", "fit_failed", "insufficient_observations"],
                )
                pd.testing.assert_frame_equal(result[frame.columns], frame)

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
