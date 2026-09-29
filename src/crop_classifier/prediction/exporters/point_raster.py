import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

from crop_classifier.prediction.exporters.base import BaseExporter


class PointRasterExporter(BaseExporter):
    """Raster creation from pixel data (lat/lon)."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        input_epsg_code: str = "EPSG:4326",
        output_epsg_code: str = None,
        pixel_size: int = 30,
        nodata: int = -9999,
        **kwargs,
    ) -> None:

        if output_epsg_code and input_epsg_code != output_epsg_code:
            transformer = Transformer.from_crs(
                input_epsg_code, output_epsg_code, always_xy=True
            )
            df["lon_meter"], df["lat_meter"] = transformer.transform(
                df["lon"], df["lat"]
            )
        else:
            df = df.rename(columns={"lon": "lon_meter", "lat": "lat_meter"})

        min_lon, min_lat = df["lon_meter"].min(), df["lat_meter"].min()
        max_lon, max_lat = df["lon_meter"].max(), df["lat_meter"].max()

        nrows = int(np.ceil((max_lat - min_lat) / pixel_size)) + 1
        ncols = int(np.ceil((max_lon - min_lon) / pixel_size)) + 1

        raster_array = np.full((nrows, ncols), nodata, dtype=np.int16)

        row_indices = ((df["lat_meter"] - min_lat) / pixel_size).astype(int)
        col_indices = ((df["lon_meter"] - min_lon) / pixel_size).astype(int)

        raster_array[row_indices, col_indices] = df["class"].values

        latitudes = np.arange(min_lat, min_lat + pixel_size * nrows, pixel_size)[:nrows]
        longitudes = np.arange(min_lon, min_lon + pixel_size * ncols, pixel_size)[
            :ncols
        ]

        data_array = xr.DataArray(
            raster_array,
            dims=("y", "x"),
            coords={
                "y": latitudes,
                "x": longitudes,
            },
        )

        data_array = data_array.rio.write_crs(output_epsg_code)
        data_array = data_array.rio.write_nodata(nodata)
        data_array.rio.to_raster(f"{output_prefix}.tif")
