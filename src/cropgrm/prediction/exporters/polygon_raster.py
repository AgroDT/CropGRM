import logging
from math import ceil, isfinite
from pathlib import Path
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from cropgrm.prediction.exporters.base import (
    BaseExporter,
    validate_geometry_path,
    validate_pixel_size,
)

logger = logging.getLogger(__name__)


class PolygonRasterExporter(BaseExporter):
    """Rasterize crop labels joined to field polygons by field_id."""

    def export(
        self,
        df: pd.DataFrame,
        output_prefix: str,
        fields_geometry_path: Union[str, Path] = None,
        output_epsg_code: str = None,
        pixel_size: float = 30.0,
        nodata: int = 255,
        **kwargs,
    ) -> None:
        """Rasterize field classes to a single-band GeoTIFF.

        Grid dimensions include partial edge cells at the requested resolution.

        Args:
            df (pandas.DataFrame): Predictions containing field_id and class; missing
                joined classes are filled with 0.
            output_prefix (str | Path): Destination prefix; .tif is appended.
            fields_geometry_path (str | Path | None): Polygon file with geometry and
                field_id; a path is required.
            output_epsg_code (str | None): Target CRS; None retains the polygon CRS.
            pixel_size (float): Finite positive spacing in output CRS units: metres
                for a metric CRS and degrees for a geographic CRS.
            nodata (int): uint8 background value, distinct from class codes and 0.
            **kwargs (object): Additional export options, ignored by this exporter.

        Raises:
            ValueError: If inputs, geometries, bounds, or class codes are invalid.
        """
        validate_geometry_path(fields_geometry_path)
        validate_pixel_size(pixel_size)
        self._export_prepared(
            df,
            output_prefix,
            fields_geometry_path,
            output_epsg_code,
            pixel_size,
            nodata,
        )

    def _export_prepared(
        self,
        df,
        output_prefix,
        fields_geometry_path,
        output_epsg_code=None,
        pixel_size=30.0,
        nodata=255,
        **kwargs,
    ):
        """Rasterize field geometry using validated export options."""
        if not isfinite(nodata) or not 1 <= nodata <= 255 or nodata != int(nodata):
            raise ValueError("nodata must be an integer in 1..255.")

        gdf = gpd.read_file(fields_geometry_path)
        if "field_id" not in gdf.columns or "geometry" not in gdf.columns:
            raise ValueError("Field geometry requires field_id and geometry columns.")
        unmatched = (
            ~gdf["field_id"].astype(str).isin(df["field_id"].astype(str))
        ).sum()
        if unmatched:
            logger.warning(
                "%d field geometries have no matching prediction field_id; "
                "their raster classes will be filled with 0.",
                unmatched,
            )
        gdf = gdf.assign(field_id=gdf["field_id"].astype(str)).merge(
            df[["field_id", "class"]].assign(field_id=df["field_id"].astype(str)),
            on="field_id",
            how="left",
        )
        if output_epsg_code and gdf.crs != output_epsg_code:
            gdf = gdf.to_crs(output_epsg_code)

        gdf = gdf.loc[
            gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid
        ].copy()
        if gdf.empty:
            raise ValueError("At least one usable geometry is required.")

        classes = gdf["class"].fillna(0)
        if not classes.map(
            lambda value: (
                isfinite(value)
                and 0 <= value <= 254
                and value == int(value)
                and value != nodata
            )
        ).all():
            raise ValueError(
                "Class codes must be integers in 0..254, distinct from nodata."
            )
        gdf["class"] = classes.astype("uint8")

        xmin, ymin, xmax, ymax = gdf.total_bounds
        if not all(isfinite(value) for value in (xmin, ymin, xmax, ymax)) or not (
            xmin < xmax and ymin < ymax
        ):
            raise ValueError(
                "Geometry bounds must be finite with positive width and height."
            )
        width = ceil((xmax - xmin) / pixel_size)
        height = ceil((ymax - ymin) / pixel_size)

        transform = from_origin(
            west=xmin, north=ymax, xsize=pixel_size, ysize=pixel_size
        )
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
        classes = sorted(gdf["class"].unique())

        colormap = {
            int(cls): tuple(int(c * 255) for c in cmap(int(cls))[:3]) for cls in classes
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
