import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from cropgrm._optional import import_optional


class OptionalDependencyTests(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_operations_explain_the_missing_extra(self):
        for dependency, extra, operation in (
            ("polars", "features", "classifier.build_features()"),
            ("ee", "gee", "classifier.download_data()"),
            ("geopandas", "geo", "classifier.save('vector', predictions=data)"),
            (
                "rioxarray",
                "geo",
                "classifier.use_zonal_spectral = False; "
                "classifier.save('raster', predictions=data, "
                "output_epsg_code='EPSG:3857')",
            ),
        ):
            with self.subTest(dependency=dependency):
                self.run_script(f"""
                    import importlib.abc
                    import sys
                    import tempfile

                    class BlockDependency(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname.split('.')[0] == {dependency!r}:
                                raise ModuleNotFoundError('Missing', name={dependency!r})

                    sys.meta_path.insert(0, BlockDependency())
                    import pandas as pd
                    from cropgrm import CropClassifier

                    data = pd.DataFrame({{
                        'field_id': ['001'], 'class': [1], 'class_name': ['winter wheat'],
                        'lon': [40.0], 'lat': [52.0],
                    }})
                    with tempfile.TemporaryDirectory() as folder:
                        classifier = CropClassifier(
                            'fields.geojson', 'unused', 2024, output_dir=folder
                        )
                        try:
                            {operation}
                        except ModuleNotFoundError as exc:
                            assert exc.name == {dependency!r}, str(exc)
                            assert '.[{extra}]' in str(exc), str(exc)
                        else:
                            raise AssertionError('Expected missing extra error')
                """)

    def test_unrelated_import_errors_are_not_relabelled(self):
        error = ModuleNotFoundError("Implementation error", name="internal_typo")
        with patch("cropgrm._optional.import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                import_optional("cropgrm.feature.feature_pipeline", "features")
        self.assertIs(caught.exception, error)

    def test_missing_geedim_explains_the_download_extra(self):
        error = ModuleNotFoundError("Missing", name="geedim")
        with patch("cropgrm._optional.import_module", side_effect=error):
            with self.assertRaisesRegex(ModuleNotFoundError, r"\.\[gee\]"):
                import_optional("geedim", "gee")

    def test_feature_import_does_not_build_symbolic_derivatives(self):
        self.run_script("""
            import importlib.abc
            import sys

            class BlockSympy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] == 'sympy':
                        raise ModuleNotFoundError('Missing', name='sympy')

            sys.meta_path.insert(0, BlockSympy())
            from cropgrm.feature import SpectralFeatureCalculator
            from cropgrm.feature.math.derivatives import get_fourth_derivative
            SpectralFeatureCalculator()
            assert 'sympy' not in sys.modules
            assert get_fourth_derivative.cache_info().currsize == 0
            try:
                get_fourth_derivative()
            except ModuleNotFoundError as exc:
                assert '.[features]' in str(exc)
            else:
                raise AssertionError('Expected missing SymPy error')
        """)

    def test_standalone_tiff_conversion_does_not_import_earth_engine(self):
        self.run_script("""
            import importlib.abc
            import sys

            class BlockEE(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'ee', 'geemap', 'geedim'}:
                        raise ModuleNotFoundError('Missing', name=fullname)

            sys.meta_path.insert(0, BlockEE())
            from cropgrm.gee_extractor import TiffTransformator
            assert TiffTransformator.__name__ == 'TiffTransformator'
            assert not {'ee', 'geemap', 'geedim'}.intersection(sys.modules)
        """)
