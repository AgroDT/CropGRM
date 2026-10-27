import unittest
from unittest.mock import patch

from cropgrm.gee_extractor.data_downloading.base import _BaseGEEExtractor
from cropgrm.gee_extractor.data_downloading.exporter import RasterExporter
from cropgrm.gee_extractor.data_downloading.meteo import MeteoExtractor


class GEEInitializationTests(unittest.TestCase):
    def test_session_is_shared_only_for_matching_project_and_endpoint(self):
        with (
            patch.object(_BaseGEEExtractor, "_initialized_config", None),
            patch(
                "cropgrm.gee_extractor.data_downloading.base.ee.Initialize"
            ) as initialize,
        ):
            MeteoExtractor("project-a")
            RasterExporter("project-a")
            initialize.assert_called_once()
            with self.assertRaisesRegex(ValueError, "already initialized"):
                MeteoExtractor("project-b")
            with self.assertRaisesRegex(ValueError, "already initialized"):
                MeteoExtractor("project-a", "https://earthengine.googleapis.com")
            initialize.assert_called_once()

    def test_failed_initialization_can_be_retried(self):
        with (
            patch.object(_BaseGEEExtractor, "_initialized_config", None),
            patch(
                "cropgrm.gee_extractor.data_downloading.base.ee.Initialize",
                side_effect=[RuntimeError("unavailable"), None],
            ) as initialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                MeteoExtractor("project-a")
            MeteoExtractor("project-b")
            self.assertEqual(initialize.call_count, 2)
