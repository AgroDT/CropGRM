import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from crop_classifier import CropClassifier, ModelType, OutputFormat
from crop_classifier.prediction.processors.make_predictions import CropPredictor

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
            )

            with patch(
                "crop_classifier.feature.processors.spectral_calculator.N_JOBS", 1
            ):
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

            classifier.predict(processed_path=feature_path)
            prediction_path = output_dir / "final/fields.csv"
            predictions = pd.read_csv(prediction_path)

            self.assertEqual(len(predictions), len(features))
            self.assertEqual(set(predictions["field_id"]), set(features["field_id"]))
            self.assertTrue({"class", "class_name"}.issubset(predictions.columns))
            self.assertTrue(predictions["class"].between(0, 16).all())

            classifier.model_type = ModelType.SMALL
            classifier.predict(
                processed_path=feature_path,
                output_prefix=output_dir / "final/fields_small",
            )
            small_predictions = pd.read_csv(output_dir / "final/fields_small.csv")
            self.assertEqual(len(small_predictions), 29)
