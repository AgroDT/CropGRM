from pathlib import Path
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from crop_classifier.prediction.exporters.base import BaseExporter


class PolygonRasterExporter(BaseExporter):
    """Field geometry rasterization on the base field_id."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        fields_geometry_path: Union[str, Path] = None,
        output_epsg_code: str = None,
        pixel_size: int = 30,
        nodata: int = -9999,
        **kwargs,
    ) -> None:
        if not fields_geometry_path:
            raise ValueError(
                "For polygon rasterization 'fields_geometry_path' required."
            )

        gdf = gpd.read_file(fields_geometry_path).merge(
            df[["field_id", "class"]], on="field_id", how="left"
        )
        gdf["class"] = gdf["class"].fillna(0).astype(int)

        if output_epsg_code and gdf.crs != output_epsg_code:
            gdf = gdf.to_crs(output_epsg_code)

        xmin, ymin, xmax, ymax = gdf.total_bounds
        width = int((xmax - xmin) / pixel_size)
        height = int((ymax - ymin) / pixel_size)

        transform = from_origin(
            west=xmin, north=ymax, xsize=pixel_size, ysize=pixel_size
        )
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=nodata,
            transform=transform,
            dtype="int16",
            all_touched=True,
        )

        cmap = plt.get_cmap("tab20")
        classes = sorted(gdf["class"].unique())

        colormap = {
            int(cls): tuple(int(c * 255) for c in cmap(i)[:3])
            for i, cls in enumerate(classes)
        }

        with rasterio.open(
            f"{output_prefix}.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            nodata=nodata,
            crs=gdf.crs,
            transform=transform,
            photometric="Palette",
            COMPRESS="ZSTD",
        ) as dst:
            dst.write(raster, 1)
            dst.write_colormap(1, colormap)
