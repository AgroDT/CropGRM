import xarray as xr
import pandas as pd


def raster_creation(df, output_tif):
    pixel_size = 30
    min_lon, min_lat = df['lon'].min(), df['lat'].min()
    max_lon, max_lat = df['lon'].max(), df['lat'].max()

    nrows = int(np.ceil((max_lat - min_lat) / pixel_size)) + 1
    ncols = int(np.ceil((max_lon - min_lon) / pixel_size)) + 1

    raster_array = np.full((nrows, ncols), np.nan, dtype=np.float32)

    row_indices = ((df['lat'] - min_lat) / pixel_size).astype(int)
    col_indices = ((df['lon'] - min_lon) / pixel_size).astype(int)

    raster_array[row_indices, col_indices] = df['class'].values

    latitudes = np.arange(min_lat, min_lat + pixel_size * nrows, pixel_size)[:nrows]
    longitudes = np.arange(min_lon, min_lon + pixel_size * ncols, pixel_size)[:ncols]

    data_array = xr.DataArray(
        raster_array,
        dims=("y", "x"),
        coords={
            "y": latitudes,
            "x": longitudes,
        },
    )

    data_array = data_array.rio.write_crs("EPSG:32637")
    data_array.rio.to_raster(output_tif)

    print(f"GeoTIFF файл '{output_tif}' успешно создан!")