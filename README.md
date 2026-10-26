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
├── src/crop_classifier # classification package and models
├── README.md
├── pyproject.toml # package metadata and dependencies
└── uv.lock # locked dependencies
```

## Models

The project includes three base models with different numbers of predictors and
a fine-tuned model (available in [models](./src/crop_classifier/models)):

- [CropGRM-large.cbm](./src/crop_classifier/models/CropGRM-large.cbm) - 134 features
- [CropGRM-optimized.cbm](./src/crop_classifier/models/CropGRM-optimized.cbm) - 82 features
- [CropGRM-small.cbm](./src/crop_classifier/models/CropGRM-small.cbm) - 24 features
- [finetuned_model.cbm](./src/crop_classifier/models/CropGRM-finetuned.cbm) - fine-tuned
  CropGRM-small on local scientific center data


## Installation

Run these commands from the checked-out repository directory with Python 3.12
or newer. They install the local project; publication on PyPI is not required.

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

The base package declares CatBoost, NumPy, pandas, and PyArrow. CatBoost itself
also depends on libraries such as SciPy and Matplotlib. Base inference does not
import CropGRM feature preparation, Earth Engine, geographic exporters, or SymPy.
Optional capabilities are loaded when used; a missing dependency reports the
corresponding extra. The symbolic seasonal derivative is initialized on demand
and cached per process during feature calculation.

For development with uv, use `uv sync --locked --all-extras`, then
`uv run --no-sync python -m unittest discover -s tests -v`. For a base-only
environment, use `uv sync --locked --no-dev`.

## Examples

Examples and instructions are available in [notebooks](./notebooks):

- [inference_tutorial.ipynb](./notebooks/inference_tutorial.ipynb) - instructions
  and examples for making field-based an pixel-based predictions

- [model_experimentation.ipynb](./notebooks/model_experimentation.ipynb) - 
  example for model comparison

- [inference_geospatial.ipynb](./notebooks/inference_geospatial.ipynb) -
  instructions and examples for making geospatial predictions

Tested on Python 3.12.

To predict from the included field features without downloading data:

```python
import pandas as pd

from crop_classifier import CropClassifier, ModelType, OutputFormat

classifier = CropClassifier(
    shapefile_path="data/raw/fields.shp",
    output_dir="data",
    use_zonal_spectral=True,
    model_type=ModelType.FINETUNED,
    threshold=0.5,
)
data = pd.read_parquet("data/processed/fields_input.parquet")
predictions = classifier.predict(data)
classifier.save(OutputFormat.TABLE)
classifier.save(OutputFormat.VECTOR)
classifier.save(OutputFormat.RASTER, output_epsg_code="EPSG:32637", pixel_size=30)
```

`predict(data)` accepts a pandas DataFrame and returns all input columns plus
`class`, `class_name`, and `probability`, without reading feature files or exporting
results. It still loads the selected model. A maximum probability at or below `threshold`
produces class 0. `save()` reuses the last predictions or accepts a DataFrame or
saved CSV through `predictions=`, without running inference again.
An export failure leaves the cached predictions available for another save attempt.
The previous `processed_path` and `output_prefix` arguments of `predict()` have
been removed: read Parquet explicitly and pass the output prefix to `save()`.

Set `CropClassifier(..., n_jobs=4)` to use four workers for spectral feature
preparation in `build_features()` and `run()`. The default is `n_jobs=1`
(sequential); `-1` explicitly uses all available CPUs. Other negative integers
follow joblib semantics, while zero, booleans, and nonintegers are rejected.
`PredictorPipeline(n_jobs=...)` and `SpectralFeatureCalculator(n_jobs=...)` also
accept this setting when used directly. This controls joblib workers rather than
CatBoost prediction threads or a global CPU limit for other libraries.

For the full workflow, also set `gee_project` and `year`, then call `run()`:
`download_data()` → `build_features()` → read Parquet → `predict(data)` → save CSV
and export to `output_format` when another format is selected.
Downloads require an authenticated Earth Engine session and an input shapefile;
the extraction period is April 1 through October 31 of `year`.
To reuse raw data, call `build_features(spectral_path=..., meteo_path=...)` and
read the returned Parquet path with `pd.read_parquet()` before calling `predict(data)`.

## Input data

- Vector dataset with field geometries (example available at
  [data/raw/fields.fgb](./data/raw/fields.fgb))
- Meteorological data (ERA5) downloaded from Google Earth Engine (example
  available at [data/raw/fields_meteo_data.csv](./data/raw/fields_meteo_data.csv)
  [data/raw/fields_meteo_data.parquet](./data/raw/fields_meteo_data.parquet)
  for field-based and pixel-based detection respectively)
- Spectral data (Landsat 5, 8, 9) downloaded from Google Earth Engine (example
  available at [data/raw/fields_spectral_data.csv](./data/raw/fields_spectral_data.csv)
  and [data/raw/fields_spectral_data.parquet](./data/raw/fields_spectral_data.parquet)
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
