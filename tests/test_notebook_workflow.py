import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from cropgrm import CropClassifier, ModelType, OutputFormat
from cropgrm.prediction.processors.make_predictions import CropPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NotebookWorkflowTests(unittest.TestCase):
    def test_local_feature_building_and_tabular_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            classifier = CropClassifier(
                shapefile_path=PROJECT_ROOT / "data/raw/fields.shp",
                gee_project="not-used-in-offline-test",
                year=2015,
                output_dir=output_dir,
                model_type=ModelType.FINETUNED,
                output_format=OutputFormat.TABLE,
                n_jobs=1,
            )

            feature_path = classifier.build_features(
                spectral_path=PROJECT_ROOT / "data/raw/fields_spectral.csv",
                meteo_path=PROJECT_ROOT / "data/raw/fields_meteo.csv",
            )

            features = pd.read_parquet(feature_path)
            self.assertEqual(len(features), 29)
            self.assertEqual(features["field_id"].nunique(), 29)
            for model_type in ModelType:
                with self.subTest(model=model_type.value):
                    model = CropPredictor().load_model(model_type)
                    self.assertTrue(
                        set(model.feature_names_).issubset(features.columns)
                    )

            result = classifier.predict(features)
            classifier.save()
            self.assertIsInstance(result, pd.DataFrame)
            prediction_path = output_dir / "final/fields.csv"
            predictions = pd.read_csv(prediction_path, dtype={"field_id": str})

            self.assertEqual(len(predictions), len(features))
            self.assertEqual(set(predictions["field_id"]), set(features["field_id"]))
            self.assertTrue({"class", "class_name"}.issubset(predictions.columns))
            self.assertTrue(predictions["class"].between(0, 16).all())
            self.assertEqual(
                list(predictions.columns),
                ["field_id", "class_name", "class", "probability", "prediction_status"],
            )

            classifier.model_type = ModelType.SMALL
            classifier.predict(features)
            classifier.save(output_prefix=output_dir / "final/fields_small")
            small_predictions = pd.read_csv(output_dir / "final/fields_small.csv")
            self.assertEqual(len(small_predictions), 29)


class NotebookScenarioTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch(
                "cropgrm.gee_extractor.gee_pipeline.GEEDownloadingDataPipeline",
                side_effect=AssertionError("Earth Engine must not be used."),
            )
        )
        self.enterContext(
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Network access must not be used."),
            )
        )

    def write_field_inputs(self, spectral_path, meteo_path, nrows=2):
        for destination, source in (
            (spectral_path, "fields_spectral.csv"),
            (meteo_path, "fields_meteo.csv"),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(
                PROJECT_ROOT / "data/raw" / source,
                nrows=nrows,
                dtype={"field_id": str},
            ).to_csv(destination, index=False)

    def test_run_builds_features_and_predicts_once_with_default_paths(self):
        classifier = CropClassifier(
            "fields.shp", "unused", 2024, self.folder, model_type=ModelType.FINETUNED
        )
        self.write_field_inputs(
            classifier.default_spectral_path,
            classifier.default_meteo_path,
            nrows=None,
        )
        expected_ids = set(
            pd.read_csv(
                classifier.default_spectral_path,
                usecols=["field_id"],
                dtype={"field_id": str},
            )["field_id"]
        )
        load_model = CropPredictor().load_model

        def load_checked_model(model_type):
            model = load_model(model_type)
            features = pd.read_parquet(classifier.default_processed_path)
            missing = set(model.feature_names_).difference(features.columns)
            self.assertFalse(
                missing, f"Missing model features in test dataset: {sorted(missing)}"
            )
            return model

        with (
            patch.object(classifier, "download_data") as download,
            patch.object(
                classifier, "build_features", wraps=classifier.build_features
            ) as build,
            patch.object(classifier, "predict", wraps=classifier.predict) as predict,
            patch.object(
                CropPredictor, "load_model", side_effect=load_checked_model
            ) as load,
        ):
            result = classifier.run()
        download.assert_called_once_with()
        build.assert_called_once_with()
        predict.assert_called_once()
        load.assert_called_once_with(ModelType.FINETUNED)
        features = pd.read_parquet(classifier.default_processed_path)
        pd.testing.assert_frame_equal(predict.call_args.args[0], features)
        saved = pd.read_csv(
            classifier.default_prediction_prefix.with_suffix(".csv"),
            dtype={
                "field_id": result["field_id"].dtype,
                "class_name": result["class_name"].dtype,
            },
        )
        self.assertEqual(len(features), len(expected_ids))
        self.assertEqual(set(features["field_id"]), expected_ids)
        self.assertEqual(set(features["field_id"]), set(result["field_id"]))
        pd.testing.assert_frame_equal(
            saved,
            result[
                ["field_id", "class_name", "class", "probability", "prediction_status"]
            ],
        )
        pd.testing.assert_frame_equal(classifier._predictions, result)

    def test_custom_paths_and_model_comparison_reuse_one_feature_file(self):
        classifier = CropClassifier("fields.shp", output_dir=self.folder / "output")
        spectral = self.folder / "custom_data/spectral_2024.csv"
        meteo = self.folder / "custom_data/meteo_2024.csv"
        self.write_field_inputs(spectral, meteo)
        destination = self.folder / "custom_data/processed_features.parquet"
        models = []
        for probabilities in ([[0.9, 0.1], [0.1, 0.9]], [[0.8, 0.2], [0.8, 0.2]]):
            model = Mock()
            model.feature_names_ = ["median_red_fitted_4"]
            model.classes_ = np.array([1, 7])
            model.predict_proba.return_value = np.array(probabilities)
            models.append(model)
        results = []
        with (
            patch.object(
                classifier, "build_features", wraps=classifier.build_features
            ) as build,
            patch.object(CropPredictor, "load_model", side_effect=models) as load,
        ):
            feature_path = classifier.build_features(
                spectral_path=spectral, meteo_path=meteo, output_path=destination
            )
            self.assertEqual(feature_path, destination)
            original_features = feature_path.read_bytes()
            features = pd.read_parquet(feature_path)
            for model_type in (ModelType.FINETUNED, ModelType.SMALL):
                classifier.model_type = model_type
                prefix = (
                    self.folder
                    / "custom_predictions"
                    / f"{classifier.input_file.stem}_{model_type.value}"
                )
                predicted = classifier.predict(features)
                classifier.save(output_prefix=prefix)
                saved = pd.read_csv(prefix.with_suffix(".csv"), dtype={"field_id": str})
                pd.testing.assert_frame_equal(
                    saved,
                    predicted[
                        [
                            "field_id",
                            "class_name",
                            "class",
                            "probability",
                            "prediction_status",
                        ]
                    ],
                )
                results.append(saved)
            build.assert_called_once_with(
                spectral_path=spectral, meteo_path=meteo, output_path=destination
            )
            self.assertEqual(
                load.call_args_list, [call(ModelType.FINETUNED), call(ModelType.SMALL)]
            )
        for model in models:
            model.predict_proba.assert_called_once()
            pd.testing.assert_frame_equal(
                model.predict_proba.call_args.args[0], features[model.feature_names_]
            )
        self.assertEqual(feature_path.read_bytes(), original_features)
        self.assertFalse(classifier.default_processed_path.exists())
        comparison = pd.DataFrame(
            {
                "field_id": results[0]["field_id"],
                "finetuned_pred": results[0]["class_name"],
                "base_pred": results[1]["class_name"],
            }
        )
        self.assertEqual(
            comparison["field_id"].tolist(), results[1]["field_id"].tolist()
        )
        agreement = comparison["finetuned_pred"] == comparison["base_pred"]
        self.assertEqual(agreement.tolist(), [True, False])
        self.assertEqual(agreement.mean(), 0.5)

    def test_pixel_features_prediction_and_csv_raster_reexport(self):
        classifier = CropClassifier(
            output_dir=self.folder / "output",
            use_zonal_spectral=False,
            epsg_code="EPSG:32637",
        )
        dates = pd.date_range("2024-04-01", "2024-10-28", freq="7D")
        days = dates.dayofyear.to_numpy()
        index = 0.1 + 0.6 * (
            1 / (1 + np.exp(-0.05 * (days - 140)))
            + 1 / (1 + np.exp(0.05 * (days - 260)))
            - 1
        )
        spectral_rows = []
        meteo_rows = []
        coordinates = [(40.0, 52.0), (40.001, 52.001)]
        for pixel, (lon, lat) in enumerate(coordinates):
            for date, value in zip(dates, index):
                spatial = {"lon": lon, "lat": lat, "date": date.strftime("%Y-%m-%d")}
                nir = 0.6 + pixel * 0.01
                spectral_rows.append(
                    dict(
                        spatial,
                        blue=0.1,
                        green=0.2,
                        nir=nir,
                        red=0.1 * nir * (1 - value) / (1 + value),
                        swir1=nir * (1 - value) / (1 + value),
                        swir2=0.3,
                    )
                )
                meteo_rows.append(
                    dict(spatial, temperature=280.0 + date.month, precipitation=0.001)
                )
        spectral = self.folder / "fields_spectral.parquet"
        meteo = self.folder / "fields_meteo.parquet"
        pd.DataFrame(spectral_rows).to_parquet(spectral, index=False)
        pd.DataFrame(meteo_rows).to_parquet(meteo, index=False)
        feature_path = classifier.build_features(
            spectral, meteo, self.folder / "processed/fields_input_pixel.parquet"
        )
        features = pd.read_parquet(feature_path)
        self.assertEqual(len(features), 2)
        self.assertNotIn("field_id", features.columns)
        self.assertEqual(set(zip(features["lon"], features["lat"])), set(coordinates))
        np.testing.assert_allclose(features["median_t_4"], 284.0)
        np.testing.assert_allclose(features["median_prec_4"], 0.001)
        prefix = self.folder / "final/fields_pixel_prediction"
        predictions = classifier.predict(features)
        classifier.save(output_prefix=prefix)
        original = predictions.copy(deep=True)
        csv = prefix.with_suffix(".csv")
        saved = pd.read_csv(
            csv,
            dtype={"class_name": predictions["class_name"].dtype},
            float_precision="round_trip",
        )
        pd.testing.assert_frame_equal(
            saved,
            predictions[
                [
                    "class_name",
                    "class",
                    "lat",
                    "lon",
                    "probability",
                    "prediction_status",
                ]
            ],
        )
        with (
            patch(
                "cropgrm.api.CropClassifierPipeline",
                side_effect=AssertionError("Export must not invoke prediction."),
            ),
            patch.object(
                CropClassifier,
                "build_features",
                side_effect=AssertionError("Export must not calculate features."),
            ),
        ):
            raster = classifier.save(OutputFormat.RASTER, prefix, pixel_size=12.5)
            fresh = CropClassifier(
                output_dir=self.folder / "new_session", use_zonal_spectral=False
            )
            reexported = fresh.save(
                OutputFormat.RASTER,
                self.folder / "final/fields_pixel_reexported",
                predictions=csv,
                input_epsg_code="EPSG:4326",
                output_epsg_code="EPSG:32637",
                pixel_size=12.5,
            )
        with rasterio.open(raster) as first, rasterio.open(reexported) as second:
            self.assertEqual(first.crs.to_epsg(), 32637)
            self.assertEqual(first.res, (12.5, 12.5))
            self.assertEqual(first.crs, second.crs)
            self.assertEqual(first.transform, second.transform)
            np.testing.assert_array_equal(first.read(1), second.read(1))
            self.assertEqual(first.read(1, masked=True).count(), len(predictions))
        pd.testing.assert_frame_equal(predictions, original)
        self.assertIsNone(fresh._predictions)

    def test_field_exports_geometry_defaults_and_interactive_map(self):
        fields = gpd.GeoDataFrame(
            {"field_id": [101, 202]},
            geometry=[
                box(500000, 5700000, 500020, 5700020),
                box(500045, 5700045, 500065, 5700065),
            ],
            crs="EPSG:32637",
        )
        geometry_path = self.folder / "fields.geojson"
        fields.to_file(geometry_path, driver="GeoJSON")
        table = pd.DataFrame(
            {
                "field_id": [101, 202],
                "class": [1, 7],
                "class_name": ["winter wheat", "sunflower"],
            }
        )
        classifier = CropClassifier(
            shapefile_path=geometry_path,
            output_dir=self.folder / "output",
            epsg_code="EPSG:32637",
        )
        with patch(
            "cropgrm.api.CropClassifierPipeline",
            side_effect=AssertionError("Export must not invoke prediction."),
        ):
            prefix = classifier.output_dir / "final/CropMap_fields"
            csv = classifier.save(OutputFormat.TABLE, prefix, predictions=table)
            raster = classifier.save(
                OutputFormat.RASTER, prefix, predictions=table, pixel_size=12.5
            )
            vector = classifier.save(OutputFormat.VECTOR, prefix, predictions=table)
            fresh = CropClassifier(output_dir=self.folder / "new_session")
            reexported = fresh.save(
                OutputFormat.VECTOR,
                self.folder / "final/CropMap_fields_reexported",
                predictions=csv,
                fields_geometry_path=geometry_path,
            )
        expected = {"101": "winter wheat", "202": "sunflower"}
        expected_geometry = dict(zip(fields["field_id"].astype(str), fields.geometry))
        for path in (vector, reexported):
            exported = gpd.read_file(path)
            self.assertEqual(exported.crs, fields.crs)
            self.assertEqual(
                exported.set_index("field_id")["class_name"].to_dict(), expected
            )
            for field_id, geometry in exported.set_index("field_id").geometry.items():
                self.assertTrue(geometry.equals(expected_geometry[field_id]))
        with rasterio.open(raster) as dataset:
            self.assertEqual(dataset.res, (12.5, 12.5))
            self.assertEqual(dataset.crs.to_epsg(), 32637)
            self.assertEqual(set(dataset.read(1, masked=True).compressed()), {1, 7})
        # Rendering HTML does not fetch tiles, scripts, or styles.
        map_view = gpd.read_file(vector).explore(column="class_name", cmap="tab20")
        html = map_view.get_root().render()
        self.assertIn("L.geoJson(", html)
        for value in ("class_name", "winter wheat", "sunflower"):
            self.assertIn(value, html)
