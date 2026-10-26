import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cropgrm.gee_extractor.data_downloading.exporter import RasterExporter
from cropgrm.gee_extractor.gee_pipeline import GEEDownloadingDataPipeline


class GEESceneFileTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = object.__new__(GEEDownloadingDataPipeline)
        pipeline.year = 2024
        pipeline.start_date = "2024-04-01"
        pipeline.end_date = "2024-11-01"
        pipeline.spectral = MagicMock()
        pipeline.meteo = MagicMock()
        pipeline.raster_exporter = MagicMock()
        return pipeline

    def test_reruns_reuse_existing_scenes_and_download_only_missing_files(self):
        for method, extractor_method in (
            ("_export_spectral_time_series", "get_time_series_collection"),
            ("_export_meteo_time_series", "get_collection"),
        ):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                pipeline = self.make_pipeline()
                extractor = (
                    pipeline.spectral if "spectral" in method else pipeline.meteo
                )
                collection = getattr(extractor, extractor_method).return_value
                collection.size.return_value.getInfo.return_value = 2
                scene_ids = ["LC08/scene_a", "LC09/scene_b"]
                timestamp = 1711929600000
                timestamps = (
                    [timestamp, timestamp]
                    if "spectral" in method
                    else [timestamp, timestamp + 86400000]
                )

                def aggregate(name):
                    result = MagicMock()
                    result.getInfo.return_value = (
                        scene_ids if name == "system:index" else timestamps
                    )
                    return result

                collection.aggregate_array.side_effect = aggregate

                def download(image, geometry, path, scale):
                    path.write_bytes(b"downloaded")

                pipeline.raster_exporter.to_geotiff.side_effect = download
                folder = Path(directory) / "2024"
                with patch("cropgrm.gee_extractor.gee_pipeline.ee.Image") as image:
                    first = getattr(pipeline, method)(MagicMock(), "fields", folder)
                    self.assertEqual(len(set(first)), 2)
                    self.assertEqual(pipeline.raster_exporter.to_geotiff.call_count, 2)
                    first[0].write_bytes(b"old contents")
                    pipeline.raster_exporter.to_geotiff.reset_mock()
                    image.reset_mock()
                    scene_ids.reverse()
                    timestamps.reverse()
                    second = getattr(pipeline, method)(MagicMock(), "fields", folder)
                    pipeline.raster_exporter.to_geotiff.assert_not_called()
                    image.assert_not_called()
                    self.assertEqual(first[0].read_bytes(), b"old contents")
                    first[1].unlink()
                    third = getattr(pipeline, method)(MagicMock(), "fields", folder)
                    pipeline.raster_exporter.to_geotiff.assert_called_once()
                    self.assertEqual(
                        pipeline.raster_exporter.to_geotiff.call_args.args[2], first[1]
                    )
                self.assertEqual(second, list(reversed(first)))
                self.assertEqual(third, second)
                self.assertEqual(set(folder.glob("*.tif")), set(first))
                self.assertEqual(first[0].read_bytes(), b"old contents")
                self.assertEqual(first[1].read_bytes(), b"downloaded")
                if "spectral" in method:
                    self.assertTrue(
                        all(path.name.endswith("_2024-04-01.tif") for path in first)
                    )

    def test_run_passes_year_directories_and_exact_scene_lists(self):
        pipeline = self.make_pipeline()
        pipeline.use_zonal_spectral = False
        with tempfile.TemporaryDirectory() as directory:
            pipeline.output_dir = Path(directory)
            spectral = [Path(directory) / "spectral.tif"]
            meteo = [Path(directory) / "meteo.tif"]
            module = "cropgrm.gee_extractor.gee_pipeline"
            with (
                patch(f"{module}.gpd.read_file", return_value=[1]),
                patch(f"{module}.geemap.shp_to_ee"),
                patch.object(
                    pipeline, "_export_spectral_time_series", return_value=spectral
                ) as export_spectral,
                patch.object(
                    pipeline, "_export_meteo_time_series", return_value=meteo
                ) as export_meteo,
                patch(f"{module}.TiffTransformator") as transform,
            ):
                pipeline.run("fields.shp")
            self.assertEqual(
                export_spectral.call_args.args[2],
                Path(directory) / "spectral_timeseries" / "2024",
            )
            self.assertEqual(
                export_meteo.call_args.args[2],
                Path(directory) / "meteo_timeseries" / "2024",
            )
            calls = transform.return_value.process_raster_series.call_args_list
            self.assertEqual(
                [call.kwargs["scene_paths"] for call in calls], [spectral, meteo]
            )

    def test_empty_weather_collection_returns_no_scene_paths(self):
        pipeline = self.make_pipeline()
        collection = pipeline.meteo.get_collection.return_value
        collection.size.return_value.getInfo.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            result = pipeline._export_meteo_time_series(
                MagicMock(), "fields", Path(directory)
            )
        self.assertEqual(result, [])
        pipeline.raster_exporter.to_geotiff.assert_not_called()

    def test_raster_download_replaces_existing_file_only_after_success(self):
        exporter = object.__new__(RasterExporter)
        image = MagicMock()
        prepared = image.gd.prepareForExport.return_value
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scene.tif"
            output.write_bytes(b"original")

            def download(path):
                Path(path).write_bytes(b"replacement")

            prepared.gd.toGeoTIFF.side_effect = download
            with patch.dict("sys.modules", {"geedim": MagicMock()}):
                exporter.to_geotiff(image, MagicMock(), output)
                self.assertEqual(output.read_bytes(), b"replacement")
                prepared.gd.toGeoTIFF.side_effect = RuntimeError("download failed")
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    exporter.to_geotiff(image, MagicMock(), output)
            self.assertEqual(output.read_bytes(), b"replacement")
