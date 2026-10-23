import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from crop_classifier.prediction.exporters.polygon_raster import PolygonRasterExporter


class PolygonRasterExportTests(unittest.TestCase):
    def export(self, folder, geometries, classes, **kwargs):
        fields = gpd.GeoDataFrame(
            {"field_id": range(len(geometries))}, geometry=geometries, crs="EPSG:3857"
        )
        predictions = pd.DataFrame({"field_id": range(len(classes)), "class": classes})
        prefix = Path(folder) / "predictions"
        with patch(
            "crop_classifier.prediction.exporters.polygon_raster.gpd.read_file",
            return_value=fields,
        ):
            PolygonRasterExporter().export(
                predictions, prefix, fields_geometry_path="fields.geojson", **kwargs
            )
        return prefix.with_suffix(".tif")

    def test_partial_edges_keep_resolution_and_cover_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.export(folder, [box(0, 0, 65, 65)], [13])
            with rasterio.open(path) as raster:
                self.assertEqual(raster.shape, (3, 3))
                self.assertEqual(raster.res, (30.0, 30.0))
                self.assertEqual(tuple(raster.bounds), (0, -25, 90, 65))
                np.testing.assert_array_equal(raster.read(1), np.full((3, 3), 13))

    def test_subpixel_geometry_and_missing_class(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.export(folder, [box(0, 0, 1, 1)], [])
            with rasterio.open(path) as raster:
                self.assertEqual(raster.shape, (1, 1))
                self.assertEqual(raster.read(1)[0, 0], 0)
                self.assertEqual(raster.read_masks(1)[0, 0], 255)

    def test_palette_codes_zero_and_background_survive_reopening(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.export(
                folder,
                [box(0, 0, 20, 20), box(60, 0, 80, 20)],
                [0, 13],
                pixel_size=10.0,
            )
            with rasterio.open(path) as raster:
                self.assertEqual(raster.dtypes, ("uint8",))
                self.assertEqual(raster.nodata, 255)
                self.assertEqual(set(np.unique(raster.read(1))), {0, 13, 255})
                self.assertFalse(raster.read(1, masked=True).mask[0, 0])
                for code in (0, 13):
                    expected = tuple(
                        int(c * 255) for c in plt.get_cmap("tab20")(code)[:3]
                    )
                    self.assertEqual(raster.colormap(1)[code][:3], expected)
            path = self.export(folder, [box(0, 0, 20, 20)], [13])
            with rasterio.open(path) as raster:
                self.assertEqual(raster.colormap(1)[13][:3], expected)

    def test_bounds_are_computed_in_output_crs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.export(
                folder,
                [box(0, 0, 65, 65)],
                [1],
                output_epsg_code="EPSG:4326",
                pixel_size=0.0002,
            )
            with rasterio.open(path) as raster:
                self.assertEqual(raster.crs.to_epsg(), 4326)
                self.assertEqual(raster.shape, (3, 3))
                self.assertEqual(raster.res, (0.0002, 0.0002))

    def test_later_polygon_still_wins_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.export(folder, [box(0, 0, 65, 65)] * 2, [1, 7])
            with rasterio.open(path) as raster:
                self.assertTrue((raster.read(1) == 7).all())

    def test_rejects_empty_or_unusable_geometries(self):
        with tempfile.TemporaryDirectory() as folder:
            for geometries in ([], [None], [box(0, 0, 0, 0)]):
                with self.subTest(geometries=geometries):
                    with self.assertRaisesRegex(ValueError, "usable geometry"):
                        self.export(folder, geometries, [1] * len(geometries))

    def test_rejects_invalid_class_codes_before_casting(self):
        with tempfile.TemporaryDirectory() as folder:
            for code in (-1, 255, 256, 1.5, float("inf")):
                with self.subTest(code=code):
                    with self.assertRaisesRegex(ValueError, "Class codes"):
                        self.export(folder, [box(0, 0, 65, 65)], [code])

    def test_invalid_pixel_size(self):
        for size in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "pixel_size.*finite.*positive"):
                    PolygonRasterExporter().export(
                        pd.DataFrame(),
                        "unused",
                        fields_geometry_path="unused",
                        pixel_size=size,
                    )
