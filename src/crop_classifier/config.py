# config.py
from enum import Enum
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).parent

# Features configuration
BASIC_INDICES = ["blue", "green", "red", "nir", "swir1", "swir2"]
METEO_INDICES = ['temperature', 'precipitation']

CLASS_NAMES = {
    1: 'winter wheat',
    2: 'spring oats',
    3: 'spring barley',
    4: 'spring rye',
    5: 'corn',
    6: 'soybean',
    7: 'sunflower',
    8: 'sugar beet',
    9: 'rapeseed',
    10: 'sorghum',
    11: 'potato',
    13: 'spring wheat',
    14: 'winter oats',
    15: 'winter barley',
    16: 'winter rye'
}

class ModelType(str, Enum):
    SMALL = "small"
    OPTIMIZED = "optimized"
    LARGE = "large"
    FINETUNED = "finetuned"


class OutputFormat(str, Enum):
    TABLE = "table"
    VECTOR = "vector"
    RASTER = "raster"


DEFAULT_MODEL_PATHS: Dict[ModelType, Path] = {
    ModelType.SMALL: ROOT_DIR / "models/small_model.cbm",
    ModelType.OPTIMIZED: ROOT_DIR / "models/optimized_model.cbm",
    ModelType.LARGE: ROOT_DIR / "models/large_model.cbm",
    ModelType.FINETUNED: ROOT_DIR / "models/finetuned_model.cbm",
}