import numpy as np


def hampel_filter(y: np.ndarray, window_size: int, sigm: float = 3) -> np.ndarray:
    """Replace local outliers with the window median.

    Args:
        y (numpy.ndarray): One-dimensional numeric observations with shape (n,).
        window_size (int): Positive window half-width in samples; each window contains
            twice this many values.
        sigm (float): Multiplier of the unscaled median absolute deviation used as the
            outlier threshold.

    Returns:
        numpy.ndarray: Copy of y with detected interior outliers replaced; edge values
            are unchanged.
    """
    new_y = y.copy()
    for i in range(window_size, len(y) - window_size):
        window = y[(i - window_size) : (i + window_size)]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if np.abs(y[i] - med) > sigm * mad:
            new_y[i] = med
    return new_y
