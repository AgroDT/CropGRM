import unittest
from unittest.mock import patch

import numpy as np

from cropgrm.feature.math.curves import fit_curve


class CurveFittingTests(unittest.TestCase):
    def setUp(self):
        self.days = np.array([90, 120, 150, 180, 210, 240], dtype=float)
        self.values = np.array([-0.3, 0.1, 0.4, 1.2, 0.5, 0.2])
        self.bounds = ([-0.2, -0.2, 0, 0, 0, 0], [1, 1, 365, 365, 1, 1])

    def test_initial_guess_is_bounded_without_changing_observations(self):
        expected = np.array([0.1, 0.8, 120, 260, 0.1, 0.1])
        original = self.values.copy()
        with patch(
            "cropgrm.feature.math.curves.curve_fit",
            return_value=(expected, None),
        ) as optimize:
            result = fit_curve(self.days, self.values, self.bounds)
        self.assertIs(result, expected)
        np.testing.assert_array_equal(
            optimize.call_args.kwargs["p0"], [-0.2, 1, 0, 365, 0.1, 0.1]
        )
        np.testing.assert_array_equal(optimize.call_args.args[2], original)
        np.testing.assert_array_equal(self.values, original)

    def test_nonconvergence_returns_none(self):
        with patch(
            "cropgrm.feature.math.curves.curve_fit",
            side_effect=RuntimeError("Optimal parameters not found"),
        ):
            self.assertIsNone(fit_curve(self.days, self.values, self.bounds))

    def test_invalid_bounds_are_not_hidden(self):
        bounds = ([1, -0.2, 0, 0, 0, 0], [-0.2, 1, 365, 365, 1, 1])
        with self.assertRaises(ValueError):
            fit_curve(self.days, self.values, bounds)
