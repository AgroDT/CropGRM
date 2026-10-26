from typing import Tuple

import ee

from cropgrm.config import BASIC_INDICES
from cropgrm.gee_extractor.data_downloading.base import _BaseGEEExtractor


class SpectralExtractor(_BaseGEEExtractor):
    """Prepare Landsat Collection 2 Level 2 spectral time series."""

    @staticmethod
    def _mask_and_scale_landsat(image: ee.Image) -> ee.Image:
        """Mask flagged pixels and scale Landsat surface reflectance.

        Args:
            image (ee.Image): Landsat image with QA_PIXEL and SR_B* bands.

        Returns:
            ee.Image: Image with scaled reflectance bands and a mask excluding dilated
                cloud, cloud, and cloud-shadow flags; acquisition time is retained.
        """
        image = ee.Image(image)
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(1 << 1)
            .eq(0)
            .And(qa.bitwiseAnd(1 << 4).eq(0))
            .And(qa.bitwiseAnd(1 << 3).eq(0))
        )
        scaled = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        return ee.Image(
            image.addBands(scaled, overwrite=True)
            .updateMask(mask)
            .copyProperties(image, ["system:time_start"])
        )

    @staticmethod
    def _process_landsat5(image: ee.Image) -> ee.Image:
        """Prepare harmonized Landsat 5 reflectance bands.

        Args:
            image (ee.Image): Landsat 5 Collection 2 Level 2 image.

        Returns:
            ee.Image: Masked, scaled, and linearly adjusted blue, green, red, nir,
                swir1, and swir2 bands with acquisition time.
        """
        image = ee.Image(image)
        img = SpectralExtractor._mask_and_scale_landsat(image)
        bands = img.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            BASIC_INDICES,
        )
        slopes = ee.Image.constant([1.0946, 1.0043, 1.0529, 1.0045, 1.0001, 1.0006])
        offsets = ee.Image.constant([0.0004, 0.0041, 0.0011, 0.0019, 0.0039, 0.0016])
        return ee.Image(
            bands.multiply(slopes)
            .add(offsets)
            .copyProperties(image, ["system:time_start"])
        )

    @staticmethod
    def _process_landsat89(image: ee.Image) -> ee.Image:
        """Prepare Landsat 8 or 9 reflectance bands.

        Args:
            image (ee.Image): Landsat 8 or 9 Collection 2 Level 2 image.

        Returns:
            ee.Image: Masked and scaled blue, green, red, nir, swir1, and swir2 bands
                with acquisition time.
        """
        image = ee.Image(image)
        img = SpectralExtractor._mask_and_scale_landsat(image)
        bands = img.select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            BASIC_INDICES,
        )
        return ee.Image(bands.copyProperties(image, ["system:time_start"]))

    def get_time_series_collection(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_cover_limit: int = 30,
        satellites: Tuple[str, ...] = ("L5", "L8", "L9"),
    ) -> ee.ImageCollection:
        """Merge prepared Landsat imagery for a region and date range.

        Args:
            geometry (ee.Geometry): Region used to filter intersecting images.
            start_date (str): Inclusive start date in YYYY-MM-DD format.
            end_date (str): Exclusive end date in YYYY-MM-DD format.
            cloud_cover_limit (int): Exclusive scene cloud-cover threshold in percent.
            satellites (tuple[str, ...]): Nonempty sequence containing only L5, L8,
                or L9 sensor codes.

        Returns:
            ee.ImageCollection: Merged images containing the configured six reflectance
                bands.

        Raises:
            ValueError: If satellites is empty or contains an unsupported sensor code.
        """
        if not satellites or any(
            satellite not in ("L5", "L8", "L9") for satellite in satellites
        ):
            raise ValueError(
                "satellites must be nonempty and contain only L5, L8, or L9."
            )

        collections = []
        if "L5" in satellites:
            collections.append(
                ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_limit))
                .map(SpectralExtractor._process_landsat5)
            )
        if "L8" in satellites:
            collections.append(
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_limit))
                .map(SpectralExtractor._process_landsat89)
            )
        if "L9" in satellites:
            collections.append(
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_limit))
                .map(SpectralExtractor._process_landsat89)
            )

        merged = collections[0]
        for col in collections[1:]:
            merged = merged.merge(col)
        return merged.select(BASIC_INDICES)
