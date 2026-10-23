import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin

from crop_classifier.feature.processors.data_preparer import DataPreparer
from crop_classifier.gee_extractor.constants import DATA_TYPE_SPECTRAL
from crop_classifier.gee_extractor.processor.tiff_transformer import TiffTransformator


class RasterTransformerTests(unittest.TestCase):
    def write_scene(self, path, values, nodata=None, mask=None):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=6,
            dtype=str(values.dtype),
            nodata=nodata,
            crs="EPSG:4326",
            transform=from_origin(40, 52, 0.01, 0.01),
        ) as raster:
            raster.write(values)
            if mask is not None:
                raster.write_mask(mask)

    def test_explicit_scene_list_excludes_other_years_and_old_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            scenes = [
                folder / "fields_L8_2024-04-01.tif",
                folder / "fields_L9_2024-04-01.tif",
            ]
            for value, path in enumerate(
                [
                    *scenes,
                    folder / "fields_old_2023-04-01.tif",
                    folder / "fields_old_2024-04-01.tif",
                ],
                1,
            ):
                self.write_scene(path, np.full((6, 1, 1), value, dtype="float32"))
            output = folder / "result.parquet"
            TiffTransformator(folder).process_raster_series(
                output, "fields", DATA_TYPE_SPECTRAL, scene_paths=scenes
            )
            result = pd.read_parquet(output)
        self.assertEqual(result["date"].to_list(), ["2024-04-01"] * 2)
        self.assertEqual(result["blue"].to_list(), [1, 2])

    def test_band_nodata_and_dataset_masks_become_arrow_nulls(self):
        for dtype in ("float32", "int16"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                values = np.ones((6, 1, 2), dtype=dtype)
                values[0, 0, 0] = -9999
                nodata_scene = folder / "fields_nodata_2024-04-01.tif"
                self.write_scene(nodata_scene, values, nodata=-9999)
                masked_scene = folder / "fields_mask_2024-04-02.tif"
                self.write_scene(
                    masked_scene,
                    np.ones_like(values),
                    mask=np.array([[0, 255]], dtype="uint8"),
                )
                output = folder / "result.parquet"
                TiffTransformator(folder).process_raster_series(
                    output,
                    "fields",
                    DATA_TYPE_SPECTRAL,
                    scene_paths=[nodata_scene, masked_scene],
                )
                table = pq.read_table(output)
                self.assertEqual(table["blue"].to_pylist(), [None, 1, None, 1])
                self.assertEqual(table["green"].to_pylist(), [1, 1, None, 1])
                data = pl.read_parquet(output)
                prepared = DataPreparer().prepare_pixel_data(
                    data, ["blue", "green"], ["lat", "lon"]
                )
                self.assertEqual(prepared.height, 3)
                self.assertNotIn(-9999, prepared["blue"].to_list())

    def test_empty_explicit_scene_list_does_not_reuse_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "result.parquet"
            output.write_bytes(b"previous output")
            with self.assertRaisesRegex(ValueError, "No raster scenes"):
                TiffTransformator(folder).process_raster_series(
                    output, "fields", DATA_TYPE_SPECTRAL, scene_paths=[]
                )
            self.assertEqual(output.read_bytes(), b"previous output")

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
