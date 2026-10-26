# config.py
from enum import Enum
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).parent


def validate_n_jobs(n_jobs: int) -> int:
    """Validate a nonzero integer worker count without importing feature tools."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer.")
    return n_jobs


# Features configuration
BASIC_INDICES = ["blue", "green", "red", "nir", "swir1", "swir2"]
METEO_INDICES = ["temperature", "precipitation"]

CLASS_NAMES = {
    1: "winter wheat",
    2: "spring oats",
    3: "spring barley",
    4: "spring rye",
    5: "corn",
    6: "soybean",
    7: "sunflower",
    8: "sugar beet",
    9: "rapeseed",
    10: "sorghum",
    11: "potato",
    13: "spring wheat",
    # 14: "winter oats",
    # 15: "winter barley",
    # 16: "winter rye"
}


class ModelType(str, Enum):
    """Identify the available crop classification model variants."""

    SMALL = "small"
    OPTIMIZED = "optimized"
    LARGE = "large"
    FINETUNED = "finetuned"


class OutputFormat(str, Enum):
    """Identify the supported prediction export formats."""

    TABLE = "table"
    VECTOR = "vector"
    RASTER = "raster"


DEFAULT_MODEL_PATHS: Dict[ModelType, Path] = {
    ModelType.SMALL: ROOT_DIR / "models/CropGRM-small.cbm",
    ModelType.OPTIMIZED: ROOT_DIR / "models/CropGRM-optimized.cbm",
    ModelType.LARGE: ROOT_DIR / "models/CropGRM-large.cbm",
    ModelType.FINETUNED: ROOT_DIR / "models/CropGRM-finetuned.cbm",
}
