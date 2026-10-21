import logging
import warnings
from calendar import monthrange
from typing import Callable, Dict

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from crop_classifier.config import BASIC_INDICES
from crop_classifier.feature.constants import (
    BOUNDS_CONFIG,
    CHUNK_SIZE,
    FEATURE_CONFIG,
    FOURTH_DERIVATIVE_LAMBDIFIED,
    N_JOBS,
    X_VALUES,
)
from crop_classifier.feature.math.curves import (
    double_logistic_function,
    fit_curve,
    sort_extrema_points,
)
from crop_classifier.feature.math.filtering_outliers import hampel_filter
from crop_classifier.feature.processors.add_indices import add_indices

logger = logging.getLogger(__name__)


class SpectralFeatureCalculator:
    """Extract spectral summaries and seasonal curve features.

    Attributes:
        pipelines (dict[str, Callable]): Registered per-index feature processors.
    """

    def __init__(self):
        """Initialize spectral feature settings and built-in processors."""
        self.feature_config = FEATURE_CONFIG
        self.bounds_config = BOUNDS_CONFIG
        self.x_values = X_VALUES
        self.basic_indices = BASIC_INDICES
        self.n_jobs = N_JOBS
        self.chunk_size = CHUNK_SIZE

        self.pipelines: Dict[str, Callable] = {
            "mean": self.process_spectral_features_means,
            "curve": self.process_spectral_features_curve,
            "extrema": self.process_ndyi_features,
        }

    def _months_for_doy(self, year: int) -> np.ndarray:
        """Map the configured day grid to calendar months for a year.

        Args:
            year (int): Calendar year of the observation group.

        Returns:
            numpy.ndarray: Month numbers with the same shape as x_values; fractional
                days use their integer part.
        """
        month_ends = np.cumsum([monthrange(year, month)[1] for month in range(1, 13)])
        return np.searchsorted(month_ends, self.x_values.astype(int)) + 1

    def process_spectral_features_means(self, group_df, index, precomputed_values=None):
        """Summarize a linearly interpolated spectral time series.

        Extrema use the configured day grid. Monthly medians use the calendar
        year of the observation group.

        Args:
            group_df (polars.DataFrame): One spatial entity in a single year, with
                field_id, year, DOY, and the selected index, ordered by date.
            index (str): Name of the numeric spectral band or index column.
            precomputed_values (numpy.ndarray | None): Optional filtered values with
                shape (n,), aligned with rows remaining after DOY >= 90 filtering.

        Returns:
            dict[str, object] | None: Field ID, index extrema and their days of year,
                and April-to-October interpolated medians; None if fewer than two
                observations remain.
        """
        group_df = group_df.filter(pl.col("DOY") >= 90)
        if len(group_df) < 2:
            return None

        x = group_df["DOY"].to_numpy().astype(float)
        y = (
            precomputed_values
            if precomputed_values is not None
            else hampel_filter(group_df[index].to_numpy().astype(float), 3)
        )

        f = interp1d(x, y, kind="linear", fill_value="extrapolate")
        y_interp = f(self.x_values)

        result = {"field_id": group_df["field_id"][0]}
        result[f"{index}_min"] = float(np.min(y_interp))
        result[f"{index}_max"] = float(np.max(y_interp))
        result[f"{index}_doy_min"] = float(self.x_values[np.argmin(y_interp)])
        result[f"{index}_doy_max"] = float(self.x_values[np.argmax(y_interp)])

        synth = pl.DataFrame(
            {"month": self._months_for_doy(group_df["year"][0]), index: y_interp}
        )

        monthly = synth.group_by("month").agg(pl.col(index).median().alias("median"))
        for month in range(4, 11):
            row = monthly.filter(pl.col("month") == month)
            result[f"median_{index}_fitted_{month}"] = (
                row["median"][0] if len(row) else None
            )

        return result

    def process_spectral_features_curve(
        self, group_df, index, bounds, precomputed_values=None
    ):
        """Extract features from a fitted double logistic curve.

        Extrema use the configured day grid. Monthly medians use the calendar
        year of the observation group.

        Args:
            group_df (polars.DataFrame): One spatial entity in a single year, with
                field_id, year, DOY, and the selected index, ordered by date.
            index (str): Name of the numeric spectral band or index column.
            bounds (tuple[array-like, array-like]): Lower and upper bounds for wNDVI,
                mNDVI, S, A, mS, and mA.
            precomputed_values (numpy.ndarray | None): Optional filtered values with
                shape (n,), aligned with rows remaining after DOY >= 90 filtering.

        Returns:
            dict[str, object] | None: Field ID, fitted parameters, extrema, growth and
                decay features, and monthly medians; times are days of year. None if
                fewer than six observations remain or the fit is invalid.
        """
        group_df = group_df.filter(pl.col("DOY") >= 90)
        if len(group_df) < 6:
            return None

        x = group_df["DOY"].to_numpy().astype(float)
        y = (
            precomputed_values
            if precomputed_values is not None
            else hampel_filter(group_df[index].to_numpy().astype(float), 3)
        )

        params = fit_curve(x, y, bounds)
        if params is None or np.isnan(params).any():
            return None

        wNDVI, mNDVI, S, A, mS, mA = params

        finer_values = double_logistic_function(self.x_values, *params)
        doy_max = float(self.x_values[np.argmax(finer_values)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fourth_derivative_values = FOURTH_DERIVATIVE_LAMBDIFIED(
                self.x_values, *params
            )
        zero_crossings_fourth = np.where(np.diff(np.sign(fourth_derivative_values)))[0]
        extrema_points_third_x = self.x_values[zero_crossings_fourth]
        extrema_points_third_x = sorted(set(extrema_points_third_x))

        sorted_points = sort_extrema_points(extrema_points_third_x, doy_max)

        result = {
            "field_id": group_df["field_id"][0],
            f"{index}_wNDVI": params[0],
            f"{index}_mNDVI": params[1],
            f"{index}_S": params[2],
            f"{index}_A": params[3],
            f"{index}_mS": params[4],
            f"{index}_mA": params[5],
            f"{index}_doy_max": doy_max,
            f"{index}_max": float(np.max(finer_values)),
            f"{index}_start_of_growth": sorted_points.get("start_of_growth"),
            f"{index}_end_of_growth": sorted_points.get("end_of_growth"),
            f"{index}_start_of_decay": sorted_points.get("start_of_decay"),
            f"{index}_end_of_decay": sorted_points.get("end_of_decay"),
            f"{index}_max_growth": double_logistic_function(
                sorted_points["end_of_growth"], *params
            )
            if sorted_points["end_of_growth"]
            else None,
            f"{index}_mean_growth": double_logistic_function(S, *params),
            f"{index}_min_growth": double_logistic_function(
                sorted_points["start_of_growth"], *params
            )
            if sorted_points["start_of_growth"]
            else None,
            f"{index}_max_decay": double_logistic_function(
                sorted_points["start_of_decay"], *params
            )
            if sorted_points["start_of_decay"]
            else None,
            f"{index}_min_decay": double_logistic_function(
                sorted_points["end_of_decay"], *params
            )
            if sorted_points["end_of_decay"]
            else None,
            f"{index}_mean_decay": double_logistic_function(A, *params),
        }

        synth = pl.DataFrame(
            {"month": self._months_for_doy(group_df["year"][0]), index: finer_values}
        )
        monthly = synth.group_by("month").agg(pl.col(index).median().alias("median"))
        for month in range(4, 11):
            row = monthly.filter(pl.col("month") == month)
            result[f"median_{index}_fitted_{month}"] = (
                row["median"][0] if len(row) else None
            )

        return result

    def process_ndyi_features(self, group_df, index, precomputed_values=None):
        """Find filtered spectral extrema and their timing.

        Args:
            group_df (polars.DataFrame): One spatial entity with field_id, DOY, and the
                selected index; observations must be ordered by DOY.
            index (str): Name of the numeric spectral band or index column.
            precomputed_values (numpy.ndarray | None): Optional filtered values with
                shape (n,), aligned with rows remaining after DOY >= 90 filtering.

        Returns:
            dict[str, object] | None: Field ID, filtered index extrema, and their
                days of year; None if no observations remain after DOY >= 90 filtering.
        """
        group_df = group_df.filter(pl.col("DOY") >= 90)
        if len(group_df) == 0:
            return None
        x = group_df["DOY"].to_numpy().astype(float)
        y = (
            precomputed_values
            if precomputed_values is not None
            else hampel_filter(group_df[index].to_numpy().astype(float), 3)
        )

        result = {"field_id": group_df["field_id"][0]}
        result[f"{index}_min"] = float(np.min(y))
        result[f"{index}_max"] = float(np.max(y))
        result[f"{index}_doy_min"] = float(x[np.argmin(y)])
        result[f"{index}_doy_max"] = float(x[np.argmax(y)])

        return result

    def register_pipeline(self, name: str, func: Callable):
        """Register or replace a named spectral feature processor.

        Args:
            name (str): Processor name referenced by the feature configuration.
            func (Callable): Processor accepting (group, index, cached_values), or
                (group, index, bounds, cached_values) for curve; returns a feature
                dictionary or None.
        """
        self.pipelines[name] = func

    def _run_pipeline(self, pipeline_name, index, grouped_data):
        """Apply a registered processor to each spatial group.

        Args:
            pipeline_name (str): Registered processor name; curve receives configured
                bounds.
            index (str): Name of the numeric spectral band or index column.
            grouped_data (list[tuple]): Pairs of a polars.DataFrame spatial group
                and a cache mapping index names to filtered numpy.ndarray values
                with shape (n,), or None.

        Returns:
            list[dict[str, object] | None]: Processor results in input group order,
                including skipped results as None.
        """
        func = self.pipelines[pipeline_name]

        if pipeline_name == "curve":
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(group, index, self.bounds_config[index], cache.get(index))
                for group, cache in grouped_data
            )

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(group, index, cache.get(index))
            for group, cache in grouped_data
        )

    def calculate_spectral_features(self, df_prepared: pl.DataFrame) -> pl.DataFrame:
        """Build configured spectral features for all spatial entities.

        Args:
            df_prepared (polars.DataFrame): Prepared observations for one year, with
                field_id, date, year, DOY, and bands required by add_indices; optional
                lat and lon are retained.

        Returns:
            polars.DataFrame: Features joined by field_id, with available spatial
                coordinates and nulls for missing feature results. Curve failures are
                logged; features with no successful results have no output columns.
        """
        df = add_indices(df_prepared)

        id_columns = [c for c in df.columns if c in ["lat", "lon", "field_id"]]
        mapping_df = df.select(id_columns).unique()

        grouped_data = []
        for _, group in df.group_by("field_id"):
            filtered_group = group.filter(pl.col("DOY") >= 90)
            if len(filtered_group) == 0:
                continue

            hampel_cache = {}
            for index in self.feature_config.keys():
                if index in filtered_group.columns:
                    values = filtered_group[index].to_numpy().astype(float)
                    hampel_cache[index] = (
                        hampel_filter(values, 3) if len(values) > 0 else None
                    )

            grouped_data.append((filtered_group, hampel_cache))

        all_field_ids = df["field_id"].unique().to_frame()
        all_features = []

        for index, cfg in self.feature_config.items():
            pipeline = cfg["pipeline"]

            results = self._run_pipeline(pipeline, index, grouped_data)
            successful = [r for r in results if r is not None]
            if pipeline == "curve" and len(successful) < len(results):
                too_short = sum(len(group) < 6 for group, _ in grouped_data)
                failed = len(results) - len(successful) - too_short
                logger.warning(
                    "Double-logistic features unavailable for index '%s': "
                    "%d groups have fewer than six observations after DOY >= 90; "
                    "%d curve fits failed or returned invalid parameters "
                    "(for example, due to noisy data).",
                    index,
                    too_short,
                    failed,
                )
            results = successful
            if results:
                all_features.append(pl.DataFrame(results))

        merged = all_field_ids
        for feat_df in all_features:
            merged = merged.join(feat_df, on="field_id", how="left")

        merged = merged.join(mapping_df, on="field_id", how="left")
        return merged
