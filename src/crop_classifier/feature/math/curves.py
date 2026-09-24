import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    sigmoid1 = 1 / (1 + np.exp(-mS * (t - S)))
    sigmoid2 = 1 / (1 + np.exp(mA * (t - A)))
    seasonal_term = sigmoid1 + sigmoid2 - 1
    return wNDVI + (mNDVI - wNDVI) * seasonal_term

def fit_curve( t, ndvi_observed, bounds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        initial_guess = [np.min(ndvi_observed), np.max(ndvi_observed), 0, 365, 0.1, 0.1]
        try:
            params, _ = curve_fit(
                double_logistic_function,
                t,
                ndvi_observed,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )
            return params
        except Exception:
            return None

@staticmethod
def sort_extrema_points(extrema_points_x, doy_max):
    less_than_doy_max = [x for x in extrema_points_x if x < doy_max]
    greater_than_doy_max = [x for x in extrema_points_x if x > doy_max]

    return {
        'start_of_growth': min(less_than_doy_max) if less_than_doy_max else None,
        'end_of_growth': max(less_than_doy_max) if less_than_doy_max else None,
        'start_of_decay': min(greater_than_doy_max) if greater_than_doy_max else None,
        'end_of_decay': max(greater_than_doy_max) if greater_than_doy_max else None
    }