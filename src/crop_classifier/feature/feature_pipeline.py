import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Union
import polars as pl
from joblib import Parallel, delayed
import gc

from crop_classifier.config import BASIC_INDICES, METEO_INDICES
from crop_classifier.feature import (
    CHUNK_SIZE,
    DataPreparer,
    MeteoFeatureCalculator,
    SpectralFeatureCalculator,

)

import logging

logger = logging.getLogger(__name__)

class PredictorPipeline:
    """Predictor calculation of meteorological and spectral features."""
    def __init__(
        self
    ):
        self.preparer = DataPreparer()
        self.spectral_extractor = SpectralFeatureCalculator()
        self.meteo_extractor = MeteoFeatureCalculator()
        
        self.basic_indices = BASIC_INDICES
        self.meteo_bands = METEO_INDICES
        self.chunk_size = CHUNK_SIZE


    def build_features(
        self, 
        spectral_path: Union[str, Path], 
        meteo_path: Union[str, Path], 
        output_file: Union[str, Path],
    ):
        """
        Building features from meteorological and spectral data from GEE.
        Use bands 'temperature' and 'precipitaion" (ERA5) 
                            and 'red', 'nir', 'blue', 'swir1', 'green', 'swir2' (Landsat 5, 8, 9)
        """
        spectral_path = Path(spectral_path)
        meteo_path = Path(meteo_path)

        if spectral_path.suffix == ".parquet":
            id_cols = ["lat", "lon"]
        else:
            id_cols = ["field_id"]

        logger.info(f"Reading dataset with spatial keys: {id_cols}...")
        
        lazy_spectral = self.preparer.read_dataset(spectral_path)
        lazy_meteo = self.preparer.read_dataset(meteo_path)

        lazy_spectral = self.preparer.ensure_field_id(lazy_spectral, id_cols)
        lazy_meteo = self.preparer.ensure_field_id(lazy_meteo, id_cols)

        spectral_ids = set(
            lazy_spectral.select("field_id").unique().collect()["field_id"]
        )
        meteo_ids = set(
            lazy_meteo.select("field_id").unique().collect()["field_id"]
        )

        common_ids = list(spectral_ids.intersection(meteo_ids))
        total_entities = len(common_ids)

        # logger.info(f"Total matching unique spatial entities: {total_entities}")

        if total_entities == 0:
            raise ValueError("No common spatial IDs found between datasets!")

        chunk_results = []
        total_chunks = (
            total_entities + self.chunk_size - 1
        ) // self.chunk_size

        for chunk_idx, i in enumerate(
            range(0, total_entities, self.chunk_size), 1
        ):
            logger.info(f"Processing chunk {chunk_idx} / {total_chunks}...")
            batch_ids = common_ids[i : i + self.chunk_size]

            spec_df = lazy_spectral.filter(
                pl.col("field_id").is_in(batch_ids)
            ).collect()
            meteo_df = lazy_meteo.filter(
                pl.col("field_id").is_in(batch_ids)
            ).collect()

            prepared_spectral = self.preparer.prepare_data_chunk(
                spec_df, self.basic_indices, id_cols
            )
            prepared_meteo = self.preparer.prepare_data_chunk(
                meteo_df, self.meteo_bands, id_cols
            )

            spectral_features = self.spectral_extractor.calculate_spectral_features(
                prepared_spectral
            )
            meteo_features = self.meteo_extractor.calculate_meteo_features(prepared_meteo)

            merged_chunk = spectral_features.join(
                meteo_features, on="field_id", how="left"
            )

            if set(id_cols) == {"lat", "lon"}:
                merged_chunk = merged_chunk.drop("field_id")

            chunk_results.append(merged_chunk)

            del (
                spec_df,
                meteo_df,
                prepared_spectral,
                prepared_meteo,
                spectral_features,
                meteo_features,
                merged_chunk,
            )
            gc.collect()

        final_df = pl.concat(chunk_results, how="diagonal")
        final_df.write_parquet(output_file)
        logger.info(f"Successfully saved full dataset to {output_file}")