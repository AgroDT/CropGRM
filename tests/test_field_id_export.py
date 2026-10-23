import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import rasterio
from shapely.geometry import box

from crop_classifier.feature.processors.data_preparer import DataPreparer
from crop_classifier.prediction.exporters.polygon_raster import PolygonRasterExporter
from crop_classifier.prediction.exporters.vector import VectorExporter


class FieldIdExportTests(unittest.TestCase):
    def test_leading_zeros_reach_exports_and_only_unmatched_ids_warn(self):
        fields = gpd.GeoDataFrame(
            {"field_id": ["001", "1", "002"]},
            geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
            crs="EPSG:3857",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                "field_id,class,class_name\n001,13,spring wheat\n1,0,\n",
                encoding="utf8",
            )
            predictions = DataPreparer.read_dataset(path).collect().to_pandas()
            module = "crop_classifier.prediction.exporters"
            prefix = Path(directory) / "result"
            with (
                patch(f"{module}.polygon_raster.gpd.read_file", return_value=fields),
                self.assertLogs(f"{module}.polygon_raster", level="WARNING") as logs,
            ):
                PolygonRasterExporter().export(
                    predictions,
                    prefix,
                    fields_geometry_path="fields.fgb",
                    pixel_size=10,
                )
            self.assertEqual(len(logs.output), 1)
            self.assertIn(
                "1 field geometries have no matching prediction", logs.output[0]
            )
            with rasterio.open(prefix.with_suffix(".tif")) as raster:
                values = raster.read(1)
                self.assertEqual(values[0, 0], 13)
                self.assertEqual(values[0, 2], 0)
                self.assertEqual(values[0, 4], 0)
            with (
                patch(f"{module}.vector.gpd.read_file", return_value=fields),
                patch.object(gpd.GeoDataFrame, "to_file", autospec=True) as write,
                self.assertLogs(f"{module}.vector", level="WARNING") as logs,
            ):
                VectorExporter().export(
                    predictions, prefix, fields_geometry_path="fields.fgb"
                )
            self.assertEqual(len(logs.output), 1)
            self.assertIn(
                "1 field geometries have no matching prediction", logs.output[0]
            )
            exported = write.call_args.args[0].set_index("field_id")
            self.assertEqual(exported.loc["001", "class_name"], "spring wheat")
            self.assertTrue(exported.loc[["1", "002"], "class_name"].isna().all())
