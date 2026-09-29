import unittest

import polars as pl

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

        result = MeteoFeatureCalculator().calculate_meteo_features(data).row(0, named=True)

        self.assertEqual(result["field_id"], "7")
        self.assertAlmostEqual(result["median_t_4"], 285.0)
        self.assertAlmostEqual(result["sum_t_4"], 290.0)
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
