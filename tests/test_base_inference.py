import os
import subprocess
import sys
import textwrap
import unittest


class BaseInferenceTests(unittest.TestCase):
    def test_real_model_and_csv_work_without_optional_capabilities(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys
            import tempfile
            from pathlib import Path

            blocked = {
                'ee', 'geedim', 'geemap', 'geopandas', 'polars', 'joblib',
                'rasterio', 'rioxarray', 'pyproj', 'xarray', 'sympy',
            }

            class BlockOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.split('.')[0]
                    if root in blocked:
                        raise ModuleNotFoundError('Blocked optional dependency', name=root)

            sys.meta_path.insert(0, BlockOptional())
            import pandas as pd
            from cropgrm import CropClassifier, ModelType
            from cropgrm.prediction.processors.make_predictions import CropPredictor

            model = CropPredictor().load_model(ModelType.FINETUNED)
            data = pd.DataFrame(0.0, index=range(2), columns=model.feature_names_)
            data['field_id'] = ['001', '002']
            with tempfile.TemporaryDirectory() as folder:
                classifier = CropClassifier(output_dir=folder)
                result = classifier.predict(data)
                assert len(result) == 2
                assert result['field_id'].tolist() == ['001', '002']
                assert {'class', 'class_name', 'probability'} <= set(result.columns)
                output = classifier.save()
                saved = pd.read_csv(output, dtype={'field_id': str})
                assert saved['field_id'].tolist() == ['001', '002']
                assert Path(output).is_file()

            assert not blocked.intersection(sys.modules)
            assert not any(
                name.startswith(('cropgrm.feature', 'cropgrm.gee_extractor'))
                for name in sys.modules
            )
            assert not any(
                name in sys.modules for name in (
                    'cropgrm.prediction.exporters.vector',
                    'cropgrm.prediction.exporters.point_raster',
                    'cropgrm.prediction.exporters.polygon_raster',
                )
            )
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
