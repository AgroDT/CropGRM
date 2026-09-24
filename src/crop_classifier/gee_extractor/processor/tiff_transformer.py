import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

from crop_classifier.config import (
    BASIC_INDICES,
    METEO_INDICES
)
from crop_classifier.gee_extractor.constants import (
    DATA_TYPE_SPECTRAL,
    DATA_TYPE_METEO
)

logger = logging.getLogger(__name__)


class TiffTransformator:
    """
    Transform tiff timeseries from GEE to .parquet file suitable for analysis
    """
    def __init__(self, folder: Path | str):
        self.basic_indices = BASIC_INDICES
        self.meteo_bands = METEO_INDICES
        self.folder = Path(folder)

    @staticmethod
    def get_pixel_coords_and_geo(
        height: int,
        width: int,
        transform: rasterio.transform.Affine,
        src_crs: Optional[CRS],
        target_crs: str = "EPSG:4326"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vector coordinate calculation:
        Return flat arrays of geographical coordinates (lat, lon).
        """
        rows, cols = np.indices((height, width))
        rows_flat = rows.ravel()
        cols_flat = cols.ravel()

        xs, ys = rasterio.transform.xy(transform, rows_flat, cols_flat)

        if src_crs and str(src_crs) != target_crs:
            transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            lons, lats = transformer.transform(xs, ys)
        else:
            lons, lats = xs, ys

        return rows_flat, cols_flat, np.array(lons), np.array(lats)

    def process_raster_series(self, output_parquet_path: str | Path, shp_name: str, data_type: str) -> None:
        files = sorted(list(self.folder.glob(f"{shp_name}*.tif")))

        if data_type == DATA_TYPE_SPECTRAL:
            band_names = self.basic_indices
        elif data_type == DATA_TYPE_METEO:
            band_names = self.meteo_bands
        else:
            raise ValueError(
                f'Invalid data_type: "{data_type}". '
                f'Only "{DATA_TYPE_SPECTRAL}" or "{DATA_TYPE_METEO}" are available.'
            )

        writer = None
        output_path = Path(output_parquet_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            for file_path in files:
                # logger.info(f'Processing {file_path.name}...')
                date_str = file_path.stem.split('_')[-1]

                with rasterio.open(file_path, 'r') as ds:
                    height, width = ds.height, ds.width
                    num_pixels = height * width

                    rows, cols, lons, lats = self.get_pixel_coords_and_geo(
                        height, width, ds.transform, ds.crs
                    )

                    bands_data = ds.read(list(range(1, len(band_names) + 1)))
                    bands_flat = bands_data.reshape(len(band_names), -1).T

                    data_dict = {
                        'date': np.full(num_pixels, date_str),
                        'lon': lons.astype(np.float64),
                        'lat': lats.astype(np.float64),
                    }

                    for i, b_name in enumerate(band_names):
                        data_dict[b_name] = bands_flat[:, i]

                    arrow_table = pa.Table.from_pydict(data_dict)

                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path,
                            arrow_table.schema
                        )

                    writer.write_table(arrow_table)

                    del bands_data, bands_flat, arrow_table, data_dict

        finally:
            if writer:
                writer.close()
                logger.info(f'Data saved to {output_path}')