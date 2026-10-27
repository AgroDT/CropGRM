import unittest
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

from cropgrm.feature.math.curves import (
    double_logistic_function,
    fit_curve,
)
from cropgrm.feature.math.derivatives import get_fourth_derivative
from cropgrm.feature.processors.add_indices import add_indices
from cropgrm.feature.processors.meteo_calculator import MeteoFeatureCalculator
from cropgrm.feature.processors.spectral_calculator import (
    SpectralFeatureCalculator,
)
from cropgrm.prediction.processors.make_predictions import CropPredictor


class FeatureCalculatorTests(unittest.TestCase):
    def test_five_observations_produce_class_zero_with_bundled_model(self):
        observations = self.curve_input([90, 110, 130, 150, 170]).with_columns(
            pl.lit(0.3).alias("swir2")
        )
        spectral = SpectralFeatureCalculator().calculate_spectral_features(observations)
        weather = MeteoFeatureCalculator().calculate_meteo_features(
            pl.DataFrame(
                {
                    "field_id": [7] * 7,
                    "month": list(range(4, 11)),
                    "temperature": [280.0] * 7,
                    "precipitation": [0.001] * 7,
                }
            )
        )
        features = spectral.join(weather, on="field_id").to_pandas()
        with patch(
            "catboost.CatBoostClassifier.predict_proba",
            side_effect=AssertionError("Insufficient observations must skip inference"),
        ):
            result = CropPredictor().predict(features)
        self.assertEqual(result["class"].tolist(), [0])
        self.assertTrue(result["probability"].isna().all())
        self.assertEqual(
            result["prediction_status"].tolist(), ["insufficient_observations"]
        )

    def test_curve_schema_and_failure_status_do_not_depend_on_other_fields(self):
        calculator = SpectralFeatureCalculator()
        calculator.feature_config = {"ndmi": {"pipeline": "curve"}}
        short = self.curve_input([90, 110, 130, 150, 170])
        complete = self.curve_input([90, 110, 130, 150, 170, 190]).with_columns(
            pl.lit(8).cast(short.schema["field_id"]).alias("field_id")
        )
        early = short.with_columns(
            pl.lit(9).cast(short.schema["field_id"]).alias("field_id"),
            pl.col("DOY") - 90,
        )
        alone = calculator.calculate_spectral_features(short)
        params = np.array([0.1, 0.9, 120.0, 260.0, 0.1, 0.1])
        with patch(
            "cropgrm.feature.processors.spectral_calculator.fit_curve",
            side_effect=lambda days, values, bounds: (
                fit_curve(days, values, bounds) if len(days) < 6 else params
            ),
        ):
            together = calculator.calculate_spectral_features(
                pl.concat([short, complete, early])
            )
        self.assertEqual(alone.schema, together.schema)
        self.assertTrue(alone.equals(together.filter(pl.col("field_id") == 7)))
        self.assertEqual(
            together.sort("field_id")["ndmi_fit_status"].to_list(),
            ["insufficient_observations", "success", "insufficient_observations"],
        )

    def test_absent_weather_months_have_null_columns(self):
        data = pl.DataFrame(
            {
                "field_id": [7],
                "month": [4],
                "temperature": [280.0],
                "precipitation": [0.001],
            }
        )
        result = MeteoFeatureCalculator().calculate_meteo_features(data)
        for prefix in ("median_t", "sum_t", "median_prec", "sum_prec"):
            for month in range(5, 11):
                self.assertIsNone(result[f"{prefix}_{month}"][0])

    def test_fourth_derivative_is_cached_and_preserves_the_seasonal_formula(self):
        import sympy as sp

        get_fourth_derivative.cache_clear()
        self.addCleanup(get_fourth_derivative.cache_clear)
        with patch.object(sp, "lambdify", wraps=sp.lambdify) as build:
            derivative = get_fourth_derivative()
            self.assertIs(get_fourth_derivative(), derivative)
            build.assert_called_once()
        x = sp.symbols("x")
        baseline, peak, growth, decay, growth_slope, decay_slope = sp.symbols(
            "wNDVI mNDVI S A mS mA"
        )
        expression = baseline + (peak - baseline) * (
            1 / (1 + sp.exp(-growth_slope * (x - growth)))
            + 1 / (1 + sp.exp(decay_slope * (x - decay)))
            - 1
        )
        for _ in range(4):
            expression = sp.diff(expression, x)
        reference = sp.lambdify(
            [x, baseline, peak, growth, decay, growth_slope, decay_slope],
            expression,
            "numpy",
        )
        days = np.linspace(90, 305, 30)
        params = [0.1, 0.9, 130, 260, 0.05, 0.07]
        np.testing.assert_allclose(
            derivative(days, *params), reference(days, *params), atol=1e-12
        )

    def test_missing_index_values_keep_dates_aligned_before_fitting(self):
        calculator = SpectralFeatureCalculator(n_jobs=1)
        calculator.feature_config = {"ndmi": {"pipeline": "curve"}}
        data = self.curve_input([90, 100, 110, 120, 130, 140, 150]).with_columns(
            pl.Series("swir1", [0.2, None, 0.2, 0.2, 0.2, 0.2, 0.2])
        )
        module = "cropgrm.feature.processors.spectral_calculator"
        with patch(f"{module}.fit_curve", return_value=None) as fit:
            calculator.calculate_spectral_features(data)
        days, values, _ = fit.call_args.args
        np.testing.assert_array_equal(days, [90, 110, 120, 130, 140, 150])
        np.testing.assert_allclose(values, 0.6)

    def test_missing_observations_are_not_counted_toward_fit_minimum(self):
        calculator = SpectralFeatureCalculator()
        group = pl.DataFrame(
            {
                "field_id": ["001"] * 6,
                "year": [2024] * 6,
                "DOY": [90, 100, 110, 120, 130, 140],
                "ndmi": [0.1, None, 0.2, 0.3, 0.4, 0.5],
            }
        )
        with patch("cropgrm.feature.math.curves.curve_fit") as fit:
            result = calculator.process_spectral_features_curve(
                group, "ndmi", calculator.bounds_config["ndmi"]
            )
        self.assertIsNone(result)
        fit.assert_not_called()

    def test_extrema_and_interpolation_ignore_missing_values(self):
        calculator = SpectralFeatureCalculator()
        group = pl.DataFrame(
            {
                "field_id": ["001"] * 4,
                "year": [2024] * 4,
                "DOY": [100, 110, 120, 130],
                "red": [0.2, None, float("inf"), 0.8],
            }
        )
        extrema = calculator.process_ndyi_features(group, "red")
        self.assertEqual(extrema["red_doy_min"], 100)
        self.assertEqual(extrema["red_doy_max"], 130)
        means = calculator.process_spectral_features_means(group, "red")
        self.assertTrue(np.isfinite(means["red_min"]))
        self.assertTrue(np.isfinite(means["red_max"]))

    def test_all_missing_weather_sums_remain_null_but_real_zero_is_preserved(self):
        data = pl.DataFrame(
            {
                "field_id": ["001", "001", "002"],
                "month": [4, 4, 4],
                "temperature": [None, None, 280.0],
                "precipitation": [None, None, 0.0],
            }
        )
        result = MeteoFeatureCalculator().calculate_meteo_features(data)
        missing = result.filter(pl.col("field_id") == "001")
        measured = result.filter(pl.col("field_id") == "002")
        self.assertIsNone(missing["sum_t_4"][0])
        self.assertIsNone(missing["sum_prec_4"][0])
        self.assertEqual(measured["sum_prec_4"][0], 0.0)

    def curve_input(self, days):
        return pl.DataFrame(
            {
                "field_id": [7] * len(days),
                "year": [2024] * len(days),
                "DOY": days,
                "blue": [0.1] * len(days),
                "green": [0.2] * len(days),
                "red": [0.1] * len(days),
                "nir": [0.8] * len(days),
                "swir1": [0.2] * len(days),
            }
        )

    def test_short_curve_series_logs_reason_without_attempting_fit(self):
        calculator = SpectralFeatureCalculator(n_jobs=1)
        calculator.feature_config = {"ndmi": {"pipeline": "curve"}}
        data = self.curve_input([50, 90, 100, 110, 120, 130])
        module = "cropgrm.feature.processors.spectral_calculator"
        with (
            patch("cropgrm.feature.math.curves.curve_fit") as fit,
            self.assertLogs(module, level="WARNING") as logs,
        ):
            result = calculator.calculate_spectral_features(data)
        fit.assert_not_called()
        self.assertEqual(result["field_id"].to_list(), [7])
        self.assertIsNone(result["ndmi_S"][0])
        self.assertIn("index 'ndmi'", logs.output[-1])
        self.assertIn("1 groups have fewer than six observations", logs.output[-1])
        self.assertIn("0 curve fits failed", logs.output[-1])

    def test_failed_curve_fits_log_reason_and_preserve_spatial_rows(self):
        module = "cropgrm.feature.processors.spectral_calculator"
        for parameters in (None, np.full(6, np.nan)):
            with self.subTest(parameters=parameters):
                calculator = SpectralFeatureCalculator(n_jobs=1)
                calculator.feature_config = {"ndmi": {"pipeline": "curve"}}
                data = self.curve_input([90, 110, 130, 150, 170, 190])
                original = data.clone()
                with (
                    patch(f"{module}.fit_curve", return_value=parameters) as fit,
                    self.assertLogs(module, level="WARNING") as logs,
                ):
                    result = calculator.calculate_spectral_features(data)
                fit.assert_called_once()
                self.assertEqual(result["field_id"].to_list(), [7])
                self.assertIsNone(result["ndmi_S"][0])
                self.assertIn(
                    "0 groups have fewer than six observations", logs.output[-1]
                )
                self.assertIn("1 curve fits failed", logs.output[-1])
                self.assertTrue(data.equals(original))

    def test_short_series_does_not_discard_successful_field_features(self):
        calculator = SpectralFeatureCalculator(n_jobs=1)
        calculator.feature_config = {"ndmi": {"pipeline": "curve"}}
        short = self.curve_input([90, 110, 130, 150, 170])
        complete = self.curve_input([90, 110, 130, 150, 170, 190]).with_columns(
            pl.col("field_id") + 1
        )
        data = pl.concat([short, complete])
        module = "cropgrm.feature.processors.spectral_calculator"
        parameters = np.array([0.1, 0.9, 120.0, 260.0, 0.1, 0.1])
        with (
            patch(
                f"{module}.fit_curve",
                side_effect=lambda days, values, bounds: (
                    fit_curve(days, values, bounds) if len(days) < 6 else parameters
                ),
            ) as fit,
            self.assertLogs(module, level="WARNING"),
        ):
            result = calculator.calculate_spectral_features(data)
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(set(result["field_id"]), {7, 8})
        self.assertIsNone(result.filter(pl.col("field_id") == 7)["ndmi_S"][0])
        self.assertEqual(result.filter(pl.col("field_id") == 8)["ndmi_S"][0], 120.0)

    def test_spectral_indices_match_hand_calculation(self):
        data = pl.DataFrame(
            {"green": [0.4], "blue": [0.2], "nir": [0.8], "swir1": [0.3], "red": [0.1]}
        )

        result = add_indices(data).row(0, named=True)

        self.assertAlmostEqual(result["ndyi"], (0.4 - 0.2) / (0.4 + 0.2))
        self.assertAlmostEqual(result["ndmi"], (0.8 - 0.3) / (0.8 + 0.3))
        self.assertAlmostEqual(result["wrdvi"], (0.8 * 0.1 - 0.1) / (0.8 * 0.1 + 0.1))

    def test_meteorological_features_aggregate_by_month(self):
        data = pl.DataFrame(
            {
                "field_id": ["7", "7", "7"],
                "month": [4, 4, 5],
                "temperature": [280.0, 290.0, 285.0],
                "precipitation": [0.001, 0.003, 0.002],
            }
        )

        result = (
            MeteoFeatureCalculator().calculate_meteo_features(data).row(0, named=True)
        )

        self.assertEqual(result["field_id"], "7")
        self.assertAlmostEqual(result["median_t_4"], 285.0)
        self.assertAlmostEqual(result["sum_t_4"], 570.0)
        self.assertAlmostEqual(result["median_prec_4"], 0.002)
        self.assertAlmostEqual(result["sum_prec_4"], 0.004)
        self.assertAlmostEqual(result["sum_t_5"], 285.0)

    def test_ndyi_extrema_return_values_and_days(self):
        data = pl.DataFrame(
            {"field_id": ["7"] * 3, "DOY": [90, 120, 150], "ndyi": [0.2, 0.8, 0.1]}
        )

        result = SpectralFeatureCalculator().process_ndyi_features(data, "ndyi")

        self.assertEqual(result["field_id"], "7")
        self.assertAlmostEqual(result["ndyi_min"], 0.1)
        self.assertAlmostEqual(result["ndyi_max"], 0.8)
        self.assertEqual(result["ndyi_doy_min"], 150)
        self.assertEqual(result["ndyi_doy_max"], 120)

    def test_day_grid_uses_the_observation_year_and_integer_days(self):
        calculator = SpectralFeatureCalculator()
        calculator.x_values = np.array([59, 60, 61, 90, 91.9, 92, 120, 121, 304, 305])

        np.testing.assert_array_equal(
            calculator._months_for_doy(2015), [2, 3, 3, 3, 4, 4, 4, 5, 10, 11]
        )
        np.testing.assert_array_equal(
            calculator._months_for_doy(2024), [2, 2, 3, 3, 3, 4, 4, 4, 10, 10]
        )

    def test_interpolated_medians_follow_calendar_month_boundaries(self):
        calculator = SpectralFeatureCalculator()
        calculator.x_values = np.array([90, 91, 120, 121, 273, 274, 304, 305])
        results = {}
        for year in [2015, 2024]:
            data = pl.DataFrame(
                {
                    "field_id": [7, 7],
                    "year": [year, year],
                    "DOY": [90, 305],
                    "red": [90.0, 305.0],
                }
            )
            results[year] = calculator.process_spectral_features_means(data, "red")

        self.assertAlmostEqual(results[2015]["median_red_fitted_4"], 105.5)
        self.assertAlmostEqual(results[2024]["median_red_fitted_4"], 120.5)
        self.assertAlmostEqual(results[2015]["median_red_fitted_10"], 289.0)
        self.assertAlmostEqual(results[2024]["median_red_fitted_10"], 304.5)
        for key in ["field_id", "red_min", "red_max", "red_doy_min", "red_doy_max"]:
            self.assertEqual(results[2015][key], results[2024][key])

    def test_curve_medians_follow_the_observation_year(self):
        calculator = SpectralFeatureCalculator()
        calculator.x_values = np.array([90, 91, 120, 121, 273, 274, 304, 305])
        params = np.array([0.1, 0.9, 120.0, 260.0, 0.1, 0.1])
        days = np.linspace(90, 305, 6)
        values = double_logistic_function(days, *params)
        module = "cropgrm.feature.processors.spectral_calculator"

        for year, april_days in [(2015, [91, 120]), (2024, [120, 121])]:
            data = pl.DataFrame(
                {"field_id": [7] * 6, "year": [year] * 6, "DOY": days, "ndmi": values}
            )
            with (
                self.subTest(year=year),
                patch(f"{module}.fit_curve", return_value=params),
                patch(
                    f"{module}.get_fourth_derivative",
                    return_value=Mock(return_value=np.zeros_like(calculator.x_values)),
                ),
            ):
                result = calculator.process_spectral_features_curve(
                    data, "ndmi", calculator.bounds_config["ndmi"], values
                )

                april_values = double_logistic_function(np.array(april_days), *params)
                expected = np.median(april_values)
                self.assertAlmostEqual(result["median_ndmi_fitted_4"], expected)
