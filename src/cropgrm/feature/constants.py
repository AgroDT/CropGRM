import numpy as np

DATA_TYPE_SPECTRAL = "spectral"
DATA_TYPE_METEO = "meteorological"

CHUNK_SIZE = 50_000

FEATURE_CONFIG = {
    "red": {"pipeline": "mean"},
    "nir": {"pipeline": "mean"},
    "blue": {"pipeline": "mean"},
    "swir1": {"pipeline": "mean"},
    "green": {"pipeline": "mean"},
    "swir2": {"pipeline": "mean"},
    "ndyi": {"pipeline": "extrema"},
    "ndmi": {"pipeline": "curve"},
    "wrdvi": {"pipeline": "curve"},
}

BOUNDS_CONFIG = {
    "wrdvi": ([-1, -1, 0, 0, 0, 0], [1, 1, 365, 365, 1, 1]),
    "ndmi": ([-0.2, -0.2, 0, 0, 0, 0], [1, 1, 365, 365, 1, 1]),
}

X_VALUES = np.linspace(1, 365, 2000)
