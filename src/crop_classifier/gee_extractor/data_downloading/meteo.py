from typing import Tuple

import ee

from crop_classifier.config import METEO_INDICES
from crop_classifier.gee_extractor.data_downloading.base import _BaseGEEExtractor


class MeteoExtractor(_BaseGEEExtractor):
    """Downloading ECMWF/ERA5_LAND/DAILY_AGGR."""

    @staticmethod
    def _rename_bands(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        return image.select(
            ["temperature_2m", "total_precipitation_sum"],
            METEO_INDICES,
        )

    def get_collection(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
    ) -> ee.ImageCollection:
        return (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(self._rename_bands)
        )