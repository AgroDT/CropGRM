from pathlib import Path
from tempfile import TemporaryDirectory

import ee

from cropgrm._optional import import_optional
from cropgrm.gee_extractor.data_downloading.base import _BaseGEEExtractor


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
        """Download a GeoTIFF and replace the destination only after success.

        Args:
            image (ee.Image): Image containing the bands to export.
            region (ee.Geometry): Geographic export region.
            output_path (Path): Destination GeoTIFF path.
            scale (int): Export pixel scale in meters.
            crs (str): CRS identifier for the exported raster.
            dtype (str): Export pixel data type passed to geedim.

        Returns:
            Path: Destination path supplied by the caller.

        Raises:
            ImportError: If geedim is unavailable.
        """
        import_optional("geedim", "gee")
        prep = image.gd.prepareForExport(
            crs=crs, scale=scale, region=region, dtype=dtype
        )
        with TemporaryDirectory(dir=output_path.parent) as directory:
            downloaded_path = Path(directory) / output_path.name
            prep.gd.toGeoTIFF(str(downloaded_path))
            downloaded_path.replace(output_path)
        return output_path
