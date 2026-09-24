from typing import Tuple
import ee
import geedim
from crop_classifier.gee_extractor.data_downloading.base import _BaseGEEExtractor
from crop_classifier.config import BASIC_INDICES

class SpectralExtractor(_BaseGEEExtractor):
    """Processing Landsat Collection 2 Level 2."""

    @staticmethod
    def _mask_and_scale_landsat(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)
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
        image = ee.Image(image)
        img = SpectralExtractor._mask_and_scale_landsat(image)
        bands = img.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            BASIC_INDICES,
        )
        slopes = ee.Image.constant([1.0946, 1.0043, 1.0529, 1.0045, 1.0001, 1.0006])
        offsets = ee.Image.constant([0.0004, 0.0041, 0.0011, 0.0019, 0.0039, 0.0016])
        return ee.Image(
            bands.multiply(slopes).add(offsets).copyProperties(image, ["system:time_start"])
        )

    @staticmethod
    def _process_landsat89(image: ee.Image) -> ee.Image:
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