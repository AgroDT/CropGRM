import tempfile
import unittest
from unittest.mock import call, patch

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from crop_classifier import CropClassifier, OutputFormat
from crop_classifier.prediction.processors.make_predictions import CropPredictor


class RasterPixelSizeAPITests(unittest.TestCase):
    def test_fractional_and_default_size_reach_real_raster_exporters(self):
        for zonal in (True, False):
            for size in (12.5, None):
                with self.subTest(zonal=zonal, size=size):
                    with tempfile.TemporaryDirectory() as folder:
                        classifier = CropClassifier(
                            "fields.geojson",
                            "unused",
                            2020,
                            folder,
                            use_zonal_spectral=zonal,
                            output_format=OutputFormat.RASTER,
                            epsg_code="EPSG:3857",
                        )
                        predictions = pd.DataFrame(
                            {
                                "field_id": [0, 1],
                                "class": [0, 13],
                                "lon": [0.0, 65.0],
                                "lat": [0.0, 65.0],
                            }
                        )
                        fields = gpd.GeoDataFrame(
                            {"field_id": [0, 1]},
                            geometry=[box(0, 0, 20, 20), box(45, 45, 65, 65)],
                            crs="EPSG:3857",
                        )
                        with (
                            patch("pandas.read_parquet", return_value=predictions),
                            patch.object(
                                CropPredictor, "predict", return_value=predictions
                            ),
                            patch("geopandas.read_file", return_value=fields),
                        ):
                            options = {} if size is None else {"pixel_size": size}
                            classifier.predict(predictions)
                            classifier.save(
                                **({} if zonal else {"input_epsg_code": "EPSG:3857"}),
                                **options,
                            )
                        with rasterio.open(
                            classifier.default_prediction_prefix.with_suffix(".tif")
                        ) as raster:
                            self.assertEqual(raster.res, (size or 30.0,) * 2)

    def test_run_forwards_fractional_size(self):
        with tempfile.TemporaryDirectory() as folder:
            classifier = CropClassifier(
                "fields.shp", "unused", 2020, folder, output_format="raster"
            )
            with (
                patch.object(classifier, "download_data"),
                patch.object(classifier, "build_features"),
                patch("pandas.read_parquet") as read,
                patch.object(classifier, "predict") as predict,
                patch.object(classifier, "save") as save,
            ):
                classifier.run(pixel_size=12.5)
            predict.assert_called_once_with(read.return_value)
            self.assertEqual(
                save.call_args_list,
                [
                    call(OutputFormat.TABLE, predictions=predict.return_value),
                    call(
                        OutputFormat.RASTER,
                        predictions=predict.return_value,
                        pixel_size=12.5,
                    ),
                ],
            )

    def test_api_rejects_invalid_size_before_reading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            classifier = CropClassifier(
                "fields.shp", "unused", 2020, folder, output_format="raster"
            )
            for size in (0, -1, float("nan"), float("inf")):
                with self.subTest(size=size):
                    with self.assertRaisesRegex(
                        ValueError, "pixel_size.*finite.*positive"
                    ):
                        classifier.save(pixel_size=size)
