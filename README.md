# CropGRM

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

CropGRM is a general model for crop spatial recognition based on time series of
remote sensing and climate data. It can predict 12 crop types globally.

## Features

- Recognizes 12 crops: winter wheat, spring wheat, spring oats, spring barley,
  spring rye, spring canola, sunflower, corn, soybean, sorghum, beet, potato.
- Uses spectral (Landsat 5, 8, 9), phenological, and climate (ERA5) features.
- Trained on global crop data from Europe, the USA, Canada, and China.
- Gradient boosting model implemented with [Catboost](https://catboost.ai/).

## Project structure

```sh
CropGRM
├── data # test data
│   ├── final # output files after prediction
│   ├── processed # preprocessed data for model predictions
│   └── raw # input files
├── notebooks # examples and instructions
├── src/cropgrm # classification package and models
├── README.md
├── pyproject.toml # package metadata and dependencies
└── uv.lock # locked dependencies
```

## Models

The project includes three base models with different numbers of predictors and
a fine-tuned model (available in [models](./src/cropgrm/models)):

- [CropGRM-large.cbm](./src/cropgrm/models/CropGRM-large.cbm) - 134 features
- [CropGRM-optimized.cbm](./src/cropgrm/models/CropGRM-optimized.cbm) - 82 features
- [CropGRM-small.cbm](./src/cropgrm/models/CropGRM-small.cbm) - 24 features
- [CropGRM-finetuned.cbm](./src/cropgrm/models/CropGRM-finetuned.cbm) - fine-tuned
  CropGRM-small on local scientific center data


## Installation

Requires Python 3.12 or newer and Git. Clone the repository, enter its root
directory, and create a virtual environment:

```sh
git clone https://github.com/AgroDT/CropGRM.git
cd CropGRM
python -m venv .venv
```

Activate the environment on Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```sh
source .venv/bin/activate
```

If your Python command is `python3`, use it in place of `python` above.
The installation commands below must be run from the repository root (the folder
containing `pyproject.toml`).

Choose the command matching your workflow:

```sh
# Inference on prepared features, Parquet input, and CSV export
python -m pip install .

# Feature preparation from local spectral and weather tables
python -m pip install ".[features]"

# Earth Engine downloading, including TIFF-to-Parquet conversion
python -m pip install ".[gee]"

# Vector and raster prediction export; standalone TIFF conversion
python -m pip install ".[geo]"

# Complete downloading, feature preparation, inference, and export workflow
python -m pip install ".[features,gee,geo]"
```

Extras can be combined: `.[features,geo]` prepares local observations and exports
maps without requiring Earth Engine. `gee` includes its own geospatial download
dependencies but does not include feature preparation. A complete `run()` needs
`features` and `gee`; add `geo` when selecting geographic prediction output.

## Examples

Detailed setup and usage instructions are available in [notebooks](./notebooks).
Open them in Jupyter or a compatible editor using the Python environment where
CropGRM is installed:

- [inference_tutorial.ipynb](./notebooks/inference_tutorial.ipynb) - instructions
  and examples for making field-based and pixel-based predictions

- [model_experimentation.ipynb](./notebooks/model_experimentation.ipynb) - 
  example for model comparison

- [inference_geospatial.ipynb](./notebooks/inference_geospatial.ipynb) -
  instructions and examples for making geospatial predictions

Earth Engine downloading requires an authenticated session, an accessible GEE
project, and an input shapefile. The extraction period is April 1 through
October 31 of the selected year. Prepared local features can be used without
Earth Engine access.

## Input data

- Vector dataset with field geometries (example available at
  [data/raw/fields.fgb](./data/raw/fields.fgb))
- Meteorological data (ERA5) downloaded from Google Earth Engine (example
  available at [data/raw/fields_meteo.csv](./data/raw/fields_meteo.csv)
  [data/raw/fields_meteo.parquet](./data/raw/fields_meteo.parquet)
  for field-based and pixel-based detection respectively)
- Spectral data (Landsat 5, 8, 9) downloaded from Google Earth Engine (example
  available at [data/raw/fields_spectral.csv](./data/raw/fields_spectral.csv)
  and [data/raw/fields_spectral.parquet](./data/raw/fields_spectral.parquet)
  for field-based and pixel-based detection respectively)
- Preprocessed feature dataset from Google Earth Engine for each field (example
  available at [data/processed/fields_input.parquet](./data/processed/fields_input.parquet)
  and [data/processed/fields_input_pixel.parquet](./data/processed/fields_input_pixel.parquet)
  for field-based and pixel-based detection respectively)

The spectral and meteorological datasets must share spatial identifiers:
`field_id` for field statistics, or matching `lat`/`lon` for pixels (used to
generate `field_id` when absent). Only shared spatial IDs are processed.
Use `use_zonal_spectral=True` for field CSV downloads and `False` for pixel
Parquet downloads. Features are always saved as Parquet; the default path is
`<output_dir>/processed/<input_stem>_input.parquet` in both modes, so use separate
output directories or explicit paths to retain both.

Results use `OutputFormat.TABLE` (CSV), `VECTOR` (FlatGeobuf), or `RASTER`
(GeoTIFF). The default prefix is `<output_dir>/final/<input_stem>`;
`output_prefix` omits the extension. CSV contains available spatial identifiers,
`class`, and `class_name`; features and `probability` remain in the returned
DataFrame. Vector and field raster export require field geometry with `field_id`.
Pixel raster export uses `lat`/`lon` and requires `output_epsg_code` in `save()`
or `epsg_code` in the constructor; `input_epsg_code` defaults to `EPSG:4326`.
Raster `pixel_size` defaults to 30 in output CRS units (metres for a metric CRS,
degrees for a geographic CRS); it can also be passed to `run()` for raster output.

## License and Attribution

All trained models, datasets, and example notebooks in this repository are
released under the Creative Commons Attribution 4.0 International (CC BY 4.0)
license (see [LICENSE](./LICENSE)).

Any use of the models - academic, commercial, or derivative - must include
proper attribution. Please cite this repository when using, modifying, or
distributing the models or derivative works.

Example attribution:

> This work uses models published in the repository
> https://github.com/AgroDT/CropGRM, licensed under CC BY 4.0.
