from pathlib import Path
from typing import Dict, Union
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
        **kwargs
    ) -> None:
        if not fields_geometry_path:
            raise ValueError("For polygon rasterization 'fields_geometry_path' required.")

        gdf = (
            gpd.read_file(fields_geometry_path)
            .merge(df[["field_id", "class"]], on="field_id", how="left")
        )
        gdf["class"] = gdf["class"].fillna(0).astype(int)
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs("epsg:3857")
        
        nodata = 0
        pixel_size = 30
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width = int((xmax - xmin) / pixel_size)
        height = int((ymax - ymin) / pixel_size)

        transform = from_origin(west=xmin, north=ymax, xsize=pixel_size, ysize=pixel_size)
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=nodata,
            transform=transform,
            dtype="uint8",
            all_touched=True,
        )

        cmap = plt.get_cmap("tab20")
        colormap = {
            i: tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(cmap.N)
        }

        with rasterio.open(
            f'{output_prefix}.tif',
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