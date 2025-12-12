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
├── models # pre-trained models
├── notebooks # examples and instructions
├── README.md
└── requirements.txt
```

## Models

The project includes three base models with different numbers of predictors and
a fine-tuned model (available in [models](./models)):

- [CropGRM-large.cbm](./models/CropGRM-large.cbm) - 134 features
- [CropGRM-optimized.cbm](./models/CropGRM-optimized.cbm) - 82 features
- [CropGRM-small.cbm](./models/CropGRM-small.cbm) - 24 features
- [finetuned_model.cbm](./models/finetuned_model.cbm) - fine-tuned
  CropGRM-small on local scientific center data

Feature lists are in [notebooks/inference_tabular.ipynb](./notebooks/inference_tabular.ipynb).

## Examples

Examples and instructions are available in [notebooks](./notebooks):

- [inference_tabular.ipynb](./notebooks/inference_tabular.ipynb) - instructions
  and examples for making tabular predictions including feature lists for
  different models
- [inference_geospatial.ipynb](./notebooks/inference_geospatial.ipynb) -
  instructions and examples for making geospatial predictions

Tested on Python 3.12.

## Input data

- Vector dataset with field geometries, (example available at
  [data/raw/fields.fgb](./data/raw/fields.fgb)).
- Preprocessed feature dataset from Google Earth Engine for each field (example
  available at [data/processed/input_data_for_model.parquet](./data/processed/input_data_for_model.parquet))

The two input datasets should have the same unique identifier for joining data.

The results can be obtained in tabular, vector, or raster format.

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
