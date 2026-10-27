import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


class InsufficientObservationsError(ValueError):
    """Raised when fewer than six finite observations are available for fitting."""


def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    """Evaluate a double logistic seasonal curve.

    Args:
        t (float | numpy.ndarray): Time in days of year; arrays have shape (n,).
        wNDVI (float): Baseline index value.
        mNDVI (float): Upper index level used to scale the seasonal term.
        S (float): Growth transition time in days of year.
        A (float): Decay transition time in days of year.
        mS (float): Growth slope parameter in inverse days.
        mA (float): Decay slope parameter in inverse days.

    Returns:
        float | numpy.ndarray: Modeled index values with the same shape as t.
    """
    sigmoid1 = 1 / (1 + np.exp(-mS * (t - S)))
    sigmoid2 = 1 / (1 + np.exp(mA * (t - A)))
    seasonal_term = sigmoid1 + sigmoid2 - 1
    return wNDVI + (mNDVI - wNDVI) * seasonal_term


def fit_curve(t, ndvi_observed, bounds):
    """Fit a double logistic curve to observed index values.

    Args:
        t (numpy.ndarray): Observation days of year with shape (n,).
        ndvi_observed (numpy.ndarray): Nonempty observed index values with shape (n,),
            aligned with t.
        bounds (tuple[array-like, array-like]): Lower and upper bounds for the six
            parameters in wNDVI, mNDVI, S, A, mS, mA order.

    Returns:
        numpy.ndarray | None: Six fitted parameters in bounds order, or None if fitting
            fails or returns nonfinite parameters.

    Raises:
        InsufficientObservationsError: If fewer than six finite observations remain.
    """
    t = np.asarray(t)
    ndvi_observed = np.asarray(ndvi_observed)
    valid = np.isfinite(t) & np.isfinite(ndvi_observed)
    t, ndvi_observed = t[valid], ndvi_observed[valid]
    if len(t) < 6:
        raise InsufficientObservationsError(
            "Double-logistic fitting requires at least six finite observations; "
            f"received {len(t)}."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        initial_guess = [np.min(ndvi_observed), np.max(ndvi_observed), 0, 365, 0.1, 0.1]
        initial_guess = np.clip(initial_guess, bounds[0], bounds[1])
        params, _ = curve_fit(
            double_logistic_function,
            t,
            ndvi_observed,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )
        return params if np.isfinite(params).all() else None



@staticmethod
def sort_extrema_points(extrema_points_x, doy_max):
    """Assign extrema before and after the seasonal peak.

    Args:
        extrema_points_x (Sequence[float]): Extremum times in days of year.
        doy_max (float): Day of year of the seasonal maximum; equal times are excluded.

    Returns:
        dict[str, float | None]: Earliest and latest times before the peak as
            start_of_growth and end_of_growth, and after it as start_of_decay and
            end_of_decay; missing sides use None.
    """
    less_than_doy_max = [x for x in extrema_points_x if x < doy_max]
    greater_than_doy_max = [x for x in extrema_points_x if x > doy_max]

    return {
        "start_of_growth": min(less_than_doy_max) if less_than_doy_max else None,
        "end_of_growth": max(less_than_doy_max) if less_than_doy_max else None,
        "start_of_decay": min(greater_than_doy_max) if greater_than_doy_max else None,
        "end_of_decay": max(greater_than_doy_max) if greater_than_doy_max else None,
    }
