import numpy as np

def hampel_filter(y: np.ndarray, window_size: int, sigm: float = 3) -> np.ndarray:
    """Filter outliers by Hampel filter"""
    new_y = y.copy()
    for i in range(window_size, len(y) - window_size):
        window = y[(i - window_size):(i + window_size)]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if np.abs(y[i] - med) > sigm * mad:
            new_y[i] = med
    return new_y