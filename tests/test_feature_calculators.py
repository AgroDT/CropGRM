import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from crop_classifier.feature.math.curves import double_logistic_function
from crop_classifier.feature.processors.add_indices import add_indices
from crop_classifier.feature.processors.meteo_calculator import MeteoFeatureCalculator
from crop_classifier.feature.processors.spectral_calculator import (
    SpectralFeatureCalculator,
)


class FeatureCalculatorTests(unittest.TestCase):
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
        module = "crop_classifier.feature.processors.spectral_calculator"

        for year, april_days in [(2015, [91, 120]), (2024, [120, 121])]:
            data = pl.DataFrame(
                {"field_id": [7] * 6, "year": [year] * 6, "DOY": days, "ndmi": values}
            )
            with self.subTest(year=year), patch(
                f"{module}.fit_curve", return_value=params
            ), patch(
                f"{module}.FOURTH_DERIVATIVE_LAMBDIFIED",
                return_value=np.zeros_like(calculator.x_values),
            ):
                result = calculator.process_spectral_features_curve(
                    data, "ndmi", calculator.bounds_config["ndmi"], values
                )

                april_values = double_logistic_function(np.array(april_days), *params)
                expected = np.median(april_values)
                self.assertAlmostEqual(result["median_ndmi_fitted_4"], expected)
