from typing import Tuple
import ee
from pathlib import Path
from crop_classifier.gee_extractor.data_downloading.base import _BaseGEEExtractor

class RasterExporter(_BaseGEEExtractor):
    """Save ee.Image locally."""

    def to_geotiff(
        self,
        image: ee.Image,
        region: ee.Geometry,
        output_path: Path,
        scale: int = 30,
        crs: str = "EPSG:4326",
        dtype: str = "float32",
    ) -> Path:
        try:
            import geedim  # noqa
        except ImportError:
            raise ImportError("Install geedim: pip install geedim")
        prep = image.gd.prepareForExport(crs=crs, scale=scale, region=region, dtype=dtype)
        prep.gd.toGeoTIFF(str(output_path))
        return output_path