import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from crop_classifier.gee_extractor.constants import DATA_TYPE_SPECTRAL
from crop_classifier.gee_extractor.processor.tiff_transformer import TiffTransformator


class RasterTransformerTests(unittest.TestCase):
    def test_spectral_geotiff_becomes_georeferenced_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raster_path = folder / "fields_tile_0_2015-04-01.tif"
            parquet_path = folder / "spectral.parquet"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=6,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(40.0, 52.0, 0.01, 0.01),
            ) as raster:
                for band in range(1, 7):
                    raster.write(np.full((2, 2), band / 10, dtype="float32"), band)

            TiffTransformator(folder).process_raster_series(
                parquet_path, "fields", DATA_TYPE_SPECTRAL
            )
            result = pd.read_parquet(parquet_path)

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["date"]), {"2015-04-01"})
        np.testing.assert_allclose(result["blue"], 0.1, atol=1e-6)
        np.testing.assert_allclose(result["swir2"], 0.6, atol=1e-6)
        self.assertAlmostEqual(result.loc[0, "lon"], 40.005)
        self.assertAlmostEqual(result.loc[0, "lat"], 51.995)
