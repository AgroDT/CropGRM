import ee

from cropgrm.config import METEO_INDICES
from cropgrm.gee_extractor.data_downloading.base import _BaseGEEExtractor


class MeteoExtractor(_BaseGEEExtractor):
    """Select daily ERA5-Land temperature and precipitation data."""

    @staticmethod
    def _rename_bands(image: ee.Image) -> ee.Image:
        """Select and rename the meteorological input bands.

        Args:
            image (ee.Image): ERA5-Land image containing temperature_2m and
                total_precipitation_sum.

        Returns:
            ee.Image: Image with temperature in kelvins and precipitation in meters;
                no unit conversion is applied.
        """
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
        """Select daily weather images for a region and date range.

        Args:
            geometry (ee.Geometry): Region used to filter intersecting images.
            start_date (str): Inclusive start date in YYYY-MM-DD format.
            end_date (str): Exclusive end date in YYYY-MM-DD format.

        Returns:
            ee.ImageCollection: ERA5-Land images with temperature in kelvins and
                precipitation in meters; these units are retained in subsequent
                processing.
        """
        return (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(self._rename_bands)
        )
