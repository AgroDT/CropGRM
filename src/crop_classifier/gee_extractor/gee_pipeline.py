import logging
from pathlib import Path
from typing import Union
from datetime import datetime, timezone

import ee
import geemap
import geedim
import geopandas as gpd

from crop_classifier.gee_extractor import SpectralExtractor, MeteoExtractor, RasterExporter
from crop_classifier.gee_extractor import TiffTransformator
from crop_classifier.gee_extractor.constants import DATA_TYPE_SPECTRAL, DATA_TYPE_METEO

logger = logging.getLogger(__name__)


class GEEDownloadingDataPipeline:
    def __init__(
        self,
        project_id: str,
        year: int,
        output_dir: Union[str, Path],
        use_zonal_spectral: bool = False,
        opt_url: str = "https://earthengine-highvolume.googleapis.com",
    ):
        self.year = year
        self.output_dir = Path(output_dir) / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = f"{year}-04-01"
        self.end_date = f"{year}-10-31"
        self.use_zonal_spectral = use_zonal_spectral

        self.spectral = SpectralExtractor(project_id, opt_url)
        self.meteo = MeteoExtractor(project_id, opt_url)
        self.raster_exporter = RasterExporter(project_id, opt_url)

    def run(
        self,
        shapefile_path: Union[str, Path],
    ) -> None:
        """Pipeline for downloading GEE data depending on input options."""
        logger.info(f"Starting pipeline for {shapefile_path}, year {self.year}")

        gdf = gpd.read_file(shapefile_path)
        shape_ee = geemap.shp_to_ee(shapefile_path)

        meteo_out_path = self.output_dir / f'{Path(shapefile_path).stem}_meteo'
        spectral_out_path = self.output_dir / f'{Path(shapefile_path).stem}_spectral'

        if self.use_zonal_spectral:
            self._fetch_spectral_stats(shape_ee, f'{spectral_out_path}.csv')
            self._fetch_meteo_stats(shape_ee, f'{meteo_out_path}.csv')
        else:
            combined_geom = shape_ee.geometry() if len(gdf) > 1 else shape_ee.geometry()

            spectral_dir = self.output_dir / "spectral_timeseries"
            meteo_dir = self.output_dir / "meteo_timeseries"

            self._export_spectral_time_series(combined_geom, Path(shapefile_path).stem, spectral_dir)
            self._export_meteo_time_series(combined_geom, Path(shapefile_path).stem, meteo_dir)

            TiffTransformator(spectral_dir).process_raster_series(
                f'{spectral_out_path}.parquet', Path(shapefile_path).stem, data_type=DATA_TYPE_SPECTRAL
            )
            TiffTransformator(meteo_dir).process_raster_series(
                f'{meteo_out_path}.parquet', Path(shapefile_path).stem, data_type=DATA_TYPE_METEO
            )

        logger.info("Downloading completed successfully.")

    def _fetch_spectral_stats(self, shape_ee: ee.FeatureCollection, output_csv: Path) -> None:
        collection = self.spectral.get_time_series_collection(
            geometry=shape_ee.geometry(),
            start_date=self.start_date,
            end_date=self.end_date,
            cloud_cover_limit=50,
            satellites=("L5", "L8", "L9"),
        )
        if collection.size().getInfo() == 0:
            logger.warning("No spectral data found for the given period.")
            raise ValueError("No spectral data found for the given period.")

        geemap.zonal_statistics(
            collection, shape_ee, output_csv, statistics_type="median", scale=30
        )
        logger.info(f"Spectral zonal stats saved to {output_csv}")

    def _export_spectral_time_series(
        self, geometry: ee.Geometry, base_name: str, export_dir: Path
    ) -> None:
        collection = self.spectral.get_time_series_collection(
            geometry=geometry,
            start_date=self.start_date,
            end_date=self.end_date,
            cloud_cover_limit=50,
            satellites=("L5", "L8", "L9"),
        )
        count = collection.size().getInfo()
        if count == 0:
            logger.warning("No spectral data for raster export.")
            raise ValueError("No spectral data found for the given period.")

        timestamps = collection.aggregate_array("system:time_start").getInfo()
        export_dir.mkdir(parents=True, exist_ok=True)

        for i, ts in enumerate(timestamps):
            date_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            tile = 0
            while True:
                ts_filename = export_dir / f"{base_name}_tile_{tile}_{date_str}.tif"
                if not ts_filename.exists():
                    break
                tile += 1

            img = ee.Image(collection.toList(count).get(i)).clip(geometry)
            self.raster_exporter.to_geotiff(img, geometry, ts_filename, scale=30)

        logger.info(f"Exported {count} spectral GeoTIFF scenes.")

    def _fetch_meteo_stats(self, shape_ee: ee.FeatureCollection, output_csv: Path) -> None:
        collection = self.meteo.get_collection(
            geometry=shape_ee.geometry(),
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if collection.size().getInfo() == 0:
            logger.warning("No meteo data found for the given period.")
            return

        geemap.zonal_statistics(
            collection, shape_ee, output_csv, statistics_type="median", scale=100
        )
        logger.info(f"Meteo zonal stats saved to {output_csv}")

    def _export_meteo_time_series(
        self, geometry: ee.Geometry, base_name: str, export_dir: Path
    ) -> None:
        collection = self.meteo.get_collection(
            geometry=geometry,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        count = collection.size().getInfo()
        if count == 0:
            logger.warning("No meteo data for raster export.")
            return

        timestamps = collection.aggregate_array("system:time_start").getInfo()
        export_dir.mkdir(parents=True, exist_ok=True)

        for i, ts in enumerate(timestamps):
            date_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            tile = 0
            while True:
                ts_filename = export_dir / f"{base_name}_tile_{tile}_{date_str}.tif"
                if not ts_filename.exists():
                    break
                tile += 1

            img = ee.Image(collection.toList(count).get(i)).clip(geometry)
            self.raster_exporter.to_geotiff(img, geometry, ts_filename, scale=30)

        logger.info(f"Exported {count} meteo GeoTIFF scenes.")
