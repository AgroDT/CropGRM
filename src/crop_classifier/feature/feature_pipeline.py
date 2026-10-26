import gc
import logging
from pathlib import Path
from typing import Union

import polars as pl

from crop_classifier.config import BASIC_INDICES, METEO_INDICES
from crop_classifier.feature import (
    CHUNK_SIZE,
    DataPreparer,
    MeteoFeatureCalculator,
    SpectralFeatureCalculator,
)

logger = logging.getLogger(__name__)


class PredictorPipeline:
    """Build model features from spectral and meteorological time series."""

    def __init__(self, n_jobs: int = 1):
        """Initialize feature tools with the requested spectral worker count.

        Args:
            n_jobs (int): Joblib worker count; 1 is sequential, -1 uses all CPUs.
        """
        self.preparer = DataPreparer()
        self.spectral_extractor = SpectralFeatureCalculator(n_jobs=n_jobs)
        self.meteo_extractor = MeteoFeatureCalculator()

        self.basic_indices = BASIC_INDICES
        self.meteo_bands = METEO_INDICES
        self.chunk_size = CHUNK_SIZE

    def build_features(
        self,
        use_zonal_spectral,
        spectral_path: Union[str, Path],
        meteo_path: Union[str, Path],
        output_file: Union[str, Path],
    ):
        """Write features for spatial entities shared by both datasets.

        Args:
            use_zonal_spectral (bool): Whether inputs contain field statistics instead
                of pixel observations.
            spectral_path (str | Path): Spectral CSV or Parquet input path.
            meteo_path (str | Path): Meteorological CSV or Parquet input path.
            output_file (str | Path): Destination for the combined feature Parquet file.

        Raises:
            ValueError: If required input bands or shared spatial IDs are missing,
                an input file format is unsupported, or observations do not belong
                to one common calendar year.
        """
        spectral_path = Path(spectral_path)
        meteo_path = Path(meteo_path)

        if use_zonal_spectral:
            id_cols = ["field_id"]
        else:
            id_cols = ["lat", "lon"]

        logger.info(f"Reading dataset with spatial keys: {id_cols}...")

        lazy_spectral = self.preparer.read_dataset(spectral_path)
        lazy_meteo = self.preparer.read_dataset(meteo_path)

        self.preparer.validate_bands(
            lazy_spectral, self.basic_indices, "spectral", spectral_path
        )
        self.preparer.validate_bands(
            lazy_meteo, self.meteo_bands, "meteorological", meteo_path
        )

        spectral_years = self.preparer.observation_years(
            lazy_spectral, self.basic_indices
        )
        meteo_years = self.preparer.observation_years(lazy_meteo, self.meteo_bands)
        if (
            len(spectral_years) != 1
            or len(meteo_years) != 1
            or spectral_years != meteo_years
        ):
            raise ValueError(
                "Feature preparation requires one common calendar year: "
                f"spectral years={sorted(spectral_years)}, "
                f"meteorological years={sorted(meteo_years)}."
            )

        lazy_spectral = self.preparer.ensure_field_id(lazy_spectral, id_cols)
        lazy_meteo = self.preparer.ensure_field_id(lazy_meteo, id_cols)

        spectral_ids = set(
            lazy_spectral.select("field_id").unique().collect()["field_id"]
        )
        meteo_ids = set(lazy_meteo.select("field_id").unique().collect()["field_id"])

        common_ids = list(spectral_ids.intersection(meteo_ids))
        total_entities = len(common_ids)

        # logger.info(f"Total matching unique spatial entities: {total_entities}")

        if total_entities == 0:
            raise ValueError("No common spatial IDs found between datasets!")

        chunk_results = []
        total_chunks = (total_entities + self.chunk_size - 1) // self.chunk_size

        if not use_zonal_spectral:
            processing_id_cols = ["lat", "lon", "field_id"]
        else:
            processing_id_cols = ["field_id"]

        for chunk_idx, i in enumerate(range(0, total_entities, self.chunk_size), 1):
            logger.info(f"Processing chunk {chunk_idx} / {total_chunks}...")
            batch_ids = common_ids[i : i + self.chunk_size]

            spec_df = lazy_spectral.filter(
                pl.col("field_id").is_in(batch_ids)
            ).collect()
            meteo_df = lazy_meteo.filter(pl.col("field_id").is_in(batch_ids)).collect()

            prepared_spectral = self.preparer.prepare_data_chunk(
                spec_df, self.basic_indices, processing_id_cols
            )
            prepared_meteo = self.preparer.prepare_data_chunk(
                meteo_df, self.meteo_bands, processing_id_cols
            )

            spectral_features = self.spectral_extractor.calculate_spectral_features(
                prepared_spectral
            )
            meteo_features = self.meteo_extractor.calculate_meteo_features(
                prepared_meteo
            )

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
