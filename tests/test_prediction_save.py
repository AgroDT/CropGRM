import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import box

from cropgrm import CropClassifier, OutputFormat
from cropgrm.prediction.exporters import ExporterFactory
from cropgrm.prediction.exporters.point_raster import PointRasterExporter
from cropgrm.prediction.processors.make_predictions import CropPredictor


class PredictionSaveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.classifier = CropClassifier(output_dir=self.folder)
        self.features = pd.DataFrame(
            {"field_id": ["001", "002"], "feature": [0.12345678901234567, 2.0]}
        )
        self.table = self.features.assign(
            probability=[0.8, 0.9],
            **{"class": [1, 7]},
            class_name=["winter wheat", "sunflower"],
            prediction_status=["predicted", "predicted"],
        )
        self.fields = gpd.GeoDataFrame(
            {"field_id": ["001", "002"]},
            geometry=[box(0, 0, 20, 20), box(45, 45, 65, 65)],
            crs="EPSG:3857",
        )

    def test_one_inference_and_repeated_real_exports(self):
        model = Mock()
        model.feature_names_ = ["feature"]
        model.classes_ = np.array([1, 7])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        original = self.features.copy(deep=True)
        self.classifier.output_format = OutputFormat.VECTOR
        with (
            patch("pandas.read_parquet", return_value=self.features) as read_features,
            patch.object(CropPredictor, "load_model", return_value=model) as load,
            patch.object(self.classifier, "build_features") as build,
            patch.object(self.classifier, "download_data") as download,
        ):
            result = self.classifier.predict(self.features)
            read_features.assert_not_called()
            self.assertFalse(self.classifier.default_prediction_prefix.parent.exists())
            csv = self.classifier.save(OutputFormat.TABLE)
            pd.testing.assert_frame_equal(
                pd.read_csv(csv, dtype={"field_id": str}, float_precision="round_trip"),
                result[
                    [
                        "field_id",
                        "class_name",
                        "class",
                        "probability",
                        "prediction_status",
                    ]
                ],
            )
            with patch("geopandas.read_file", return_value=self.fields):
                raster = self.classifier.save(
                    OutputFormat.RASTER,
                    fields_geometry_path="fields.geojson",
                    pixel_size=12.5,
                    output_epsg_code="EPSG:3857",
                )
                vector = self.classifier.save(fields_geometry_path="fields.geojson")
            load.assert_called_once()
            model.predict_proba.assert_called_once()
            read_features.assert_not_called()
            build.assert_not_called()
            download.assert_not_called()
        with rasterio.open(raster) as dataset:
            self.assertEqual(dataset.res, (12.5, 12.5))
        self.assertTrue(vector.is_file())
        pd.testing.assert_frame_equal(self.features, original)
        pd.testing.assert_frame_equal(result, self.table)
        pd.testing.assert_frame_equal(self.classifier._predictions, self.table)

    def test_predict_needs_no_writable_output_or_table_io(self):
        blocked = self.folder / "file"
        blocked.write_text("not a directory", encoding="utf8")
        model = Mock()
        model.feature_names_ = ["feature"]
        model.classes_ = np.array([1, 7])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        original = self.features.copy(deep=True)
        with (
            patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")),
            patch("pandas.read_parquet", side_effect=AssertionError("read")),
            patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("write")),
            patch.object(CropClassifier, "save", side_effect=AssertionError("export")),
            patch.object(CropPredictor, "load_model", return_value=model),
        ):
            classifier = CropClassifier(output_dir=blocked / "output")
            result = classifier.predict(self.features)
        pd.testing.assert_frame_equal(result, self.table)
        pd.testing.assert_frame_equal(self.features, original)
        self.features.loc[0, "feature"] = -100.0
        self.assertEqual(result.loc[0, "feature"], original.loc[0, "feature"])
        result.loc[0, "class"] = 7
        result.loc[0, "feature"] = -200.0
        pd.testing.assert_frame_equal(classifier._predictions, self.table)

    def test_direct_exporters_leave_prediction_tables_unchanged(self):
        original = self.table.copy(deep=True)
        for fmt in (OutputFormat.TABLE, OutputFormat.VECTOR, OutputFormat.RASTER):
            with self.subTest(format=fmt):
                exporter = ExporterFactory.get_exporter(fmt, True, self.table)
                with patch("geopandas.read_file", return_value=self.fields):
                    exporter.export(
                        self.table,
                        self.folder / f"direct_{fmt.value}",
                        fields_geometry_path="fields.geojson",
                        pixel_size=10,
                    )
                pd.testing.assert_frame_equal(self.table, original)

    def test_direct_point_raster_keeps_input_and_grid_with_or_without_reprojection(
        self,
    ):
        for source_crs in ("EPSG:3857", "EPSG:4326"):
            with self.subTest(source_crs=source_crs):
                lons, lats = [0.0, 25.0], [0.0, 25.0]
                if source_crs != "EPSG:3857":
                    lons, lats = Transformer.from_crs(
                        "EPSG:3857", source_crs, always_xy=True
                    ).transform(lons, lats)
                points = self.table.assign(
                    lon=lons, lat=lats, lon_meter=[-1.0, -2.0], lat_meter=[-3.0, -4.0]
                )
                points.index = [17, 5]
                original = points.copy(deep=True)
                prefix = self.folder / f"direct_points_{source_crs[-4:]}"
                PointRasterExporter().export(
                    points,
                    prefix,
                    input_epsg_code=source_crs,
                    output_epsg_code="EPSG:3857",
                    pixel_size=10,
                )
                pd.testing.assert_frame_equal(points, original)
                expected = np.full((4, 4), -9999, dtype=np.int16)
                expected[0, 0], expected[2, 2] = 1, 7
                with rasterio.open(prefix.with_suffix(".tif")) as raster:
                    np.testing.assert_array_equal(raster.read(1), expected)
                    self.assertEqual(raster.res, (10.0, 10.0))
                    self.assertEqual(raster.crs.to_epsg(), 3857)
                    self.assertEqual(raster.nodata, -9999)

    def test_failed_point_export_does_not_modify_input_or_cached_predictions(self):
        points = self.table.assign(lon=[40.0, 40.001], lat=[52.0, 52.001])
        original = points.copy(deep=True)
        self.classifier.use_zonal_spectral = False
        self.classifier._predictions = points
        with patch(
            "rioxarray.raster_array.RasterArray.to_raster", side_effect=OSError("write")
        ):
            with self.assertRaisesRegex(OSError, "write"):
                PointRasterExporter().export(
                    points,
                    self.folder / "direct_failure",
                    output_epsg_code="EPSG:32637",
                )
            pd.testing.assert_frame_equal(points, original)
            with self.assertRaisesRegex(OSError, "write"):
                self.classifier.save(OutputFormat.RASTER, output_epsg_code="EPSG:32637")
        pd.testing.assert_frame_equal(points, original)
        pd.testing.assert_frame_equal(self.classifier._predictions, original)

    def test_predict_rejects_non_dataframe_input_without_reading_files(self):
        for data in ("features.parquet", Path("features.parquet"), None):
            with self.subTest(data=data):
                self.classifier._predictions = self.table
                with (
                    patch("pandas.read_parquet", side_effect=AssertionError("read")),
                    self.assertRaisesRegex(TypeError, "pandas DataFrame"),
                ):
                    self.classifier.predict(data)
                self.assertIsNone(self.classifier._predictions)

    def test_explicit_table_and_saved_csv_need_no_prediction_resources(self):
        original = self.table.copy(deep=True)
        with (
            patch.object(CropPredictor, "load_model", side_effect=AssertionError),
            patch.object(CropPredictor, "predict", side_effect=AssertionError),
            patch.object(CropClassifier, "build_features", side_effect=AssertionError),
            patch.object(CropClassifier, "download_data", side_effect=AssertionError),
            patch("pandas.read_parquet", side_effect=AssertionError),
        ):
            # An explicit source takes precedence over a different cached result.
            self.classifier._predictions = self.table.iloc[:0]
            csv = self.classifier.save(predictions=self.table)
            fresh = CropClassifier(output_dir=self.folder)
            with patch("geopandas.read_file", return_value=self.fields):
                vector = fresh.save(
                    OutputFormat.VECTOR,
                    self.folder / "from_csv",
                    predictions=csv,
                    fields_geometry_path="fields.geojson",
                )
            self.assertTrue(vector.is_file())
            exported = gpd.read_file(vector)
            self.assertEqual(set(exported["class_name"]), {"winter wheat", "sunflower"})
            copy = fresh.save(output_prefix=self.folder / "copy", predictions=csv)
            pd.testing.assert_frame_equal(
                pd.read_csv(
                    copy, dtype={"field_id": str}, float_precision="round_trip"
                ),
                original[
                    [
                        "field_id",
                        "class_name",
                        "class",
                        "probability",
                        "prediction_status",
                    ]
                ],
            )
        pd.testing.assert_frame_equal(self.table, original)
        self.assertIsNone(fresh._predictions)

    def test_point_reprojection_does_not_mutate_table(self):
        points = self.table.assign(lon=[40.0, 40.001], lat=[52.0, 52.001])
        original = points.copy(deep=True)
        self.classifier.use_zonal_spectral = False
        csv = self.classifier.save(OutputFormat.TABLE, predictions=points)
        pd.testing.assert_frame_equal(
            pd.read_csv(csv, dtype={"field_id": str}, float_precision="round_trip"),
            original[
                [
                    "field_id",
                    "class_name",
                    "class",
                    "lat",
                    "lon",
                    "probability",
                    "prediction_status",
                ]
            ],
        )
        path = self.classifier.save(
            OutputFormat.RASTER,
            predictions=points,
            output_epsg_code="EPSG:32637",
            pixel_size=12.5,
        )
        with rasterio.open(path) as dataset:
            self.assertEqual(dataset.res, (12.5, 12.5))
            self.assertEqual(dataset.crs.to_epsg(), 32637)
        pd.testing.assert_frame_equal(points, original)

    def test_failed_prediction_clears_previous_result(self):
        self.classifier._predictions = self.table
        with patch.object(
            CropPredictor, "predict", side_effect=RuntimeError("inference")
        ):
            with self.assertRaisesRegex(RuntimeError, "inference"):
                self.classifier.predict(self.features)
        with self.assertRaisesRegex(ValueError, "No predictions"):
            self.classifier.save()

    def test_run_keeps_cache_only_when_inference_succeeded(self):
        for step in ("read", "export"):
            with self.subTest(step=step):
                self.classifier._predictions = self.table
                with (
                    patch.object(self.classifier, "download_data"),
                    patch.object(
                        self.classifier,
                        "build_features",
                        return_value=self.folder / "features.parquet",
                    ),
                    patch(
                        "pandas.read_parquet",
                        side_effect=OSError("read") if step == "read" else None,
                        return_value=self.features,
                    ),
                    patch.object(CropPredictor, "predict", return_value=self.table),
                    patch.object(
                        self.classifier, "save", side_effect=OSError("export")
                    ),
                ):
                    with self.assertRaises(OSError):
                        self.classifier.run()
                if step == "read":
                    self.assertIsNone(self.classifier._predictions)
                else:
                    pd.testing.assert_frame_equal(
                        self.classifier._predictions, self.table
                    )
                    with patch.object(
                        CropPredictor,
                        "predict",
                        side_effect=AssertionError("inference"),
                    ):
                        self.assertTrue(self.classifier.save().is_file())

    def test_missing_sources_columns_geometry_and_crs(self):
        with self.assertRaisesRegex(ValueError, "No predictions"):
            self.classifier.save()
        for fmt, table, missing in (
            (OutputFormat.TABLE, pd.DataFrame(), "class"),
            (OutputFormat.VECTOR, self.table.drop(columns="field_id"), "field_id"),
            (OutputFormat.VECTOR, self.table.drop(columns="class_name"), "class_name"),
            (OutputFormat.RASTER, self.table.drop(columns="class"), "class"),
        ):
            with self.subTest(fmt=fmt, missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    self.classifier.save(fmt, predictions=table)
        for fmt in (OutputFormat.VECTOR, OutputFormat.RASTER):
            with self.assertRaisesRegex(ValueError, "fields_geometry_path"):
                self.classifier.save(fmt, predictions=self.table)
            with patch(
                "geopandas.read_file", return_value=self.fields.drop(columns="field_id")
            ):
                with self.assertRaisesRegex(ValueError, "field_id"):
                    self.classifier.save(
                        fmt,
                        predictions=self.table,
                        fields_geometry_path="fields.geojson",
                    )
        self.classifier.use_zonal_spectral = False
        with self.assertRaisesRegex(ValueError, "lat, lon"):
            self.classifier.save(OutputFormat.RASTER, predictions=self.table)
        with self.assertRaisesRegex(ValueError, "output_epsg_code"):
            self.classifier.save(
                OutputFormat.RASTER, predictions=self.table.assign(lat=0, lon=0)
            )
        with self.assertRaisesRegex(ValueError, "CSV"):
            self.classifier.save(predictions="table.fgb")

    def test_export_options_are_not_silently_ignored(self):
        for options in (
            {"pixel_size": 12.5},
            {"output_epsg_code": "EPSG:3857"},
            {"input_epsg_code": "EPSG:3857"},
            {"fields_geometry_path": "fields.geojson"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.classifier.save(predictions=self.table, **options)

    def test_run_predicts_once_and_reuses_result_for_final_export(self):
        self.classifier.output_format = OutputFormat.VECTOR
        with (
            patch.object(self.classifier, "download_data") as download,
            patch.object(self.classifier, "build_features") as build,
            patch("pandas.read_parquet", return_value=self.features),
            patch.object(
                CropPredictor, "predict", return_value=self.table
            ) as inference,
            patch("geopandas.read_file", return_value=self.fields),
        ):
            self.classifier.input_file = Path("fields.geojson")
            result = self.classifier.run()
            inference.assert_called_once()
            download.assert_called_once()
            build.assert_called_once()
        self.assertTrue((self.folder / "final/fields.csv").exists())
        self.assertTrue((self.folder / "final/fields.fgb").exists())
        pd.testing.assert_frame_equal(result, self.table)
