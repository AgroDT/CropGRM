import logging
import warnings
from calendar import monthrange
from typing import Callable, Dict

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from cropgrm.config import BASIC_INDICES, validate_n_jobs
from cropgrm.feature.constants import (
    BOUNDS_CONFIG,
    CHUNK_SIZE,
    FEATURE_CONFIG,
    X_VALUES,
)
from cropgrm.feature.math.curves import (
    InsufficientObservationsError,
    double_logistic_function,
    fit_curve,
    sort_extrema_points,
)
from cropgrm.feature.math.derivatives import get_fourth_derivative
from cropgrm.feature.math.filtering_outliers import hampel_filter
from cropgrm.feature.processors.add_indices import add_indices

logger = logging.getLogger(__name__)


class SpectralFeatureCalculator:
    """Extract spectral summaries and seasonal curve features.

    Attributes:
        pipelines (dict[str, Callable]): Registered per-index feature processors.
    """

    def __init__(self, n_jobs: int = 1):
        """Initialize spectral settings and built-in processors.

        Args:
            n_jobs (int): Nonzero joblib worker count; 1 is sequential, -1 uses all
                available CPUs. Defaults to 1.
        """
        self.feature_config = FEATURE_CONFIG
        self.bounds_config = BOUNDS_CONFIG
        self.x_values = X_VALUES
        self.basic_indices = BASIC_INDICES
        self.n_jobs = validate_n_jobs(n_jobs)
        self.chunk_size = CHUNK_SIZE

        self.pipelines: Dict[str, Callable] = {
            "mean": self._process_spectral_features_means,
            "curve": self._process_spectral_features_curve,
            "extrema": self._process_ndyi_features,
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

    @staticmethod
    def _prepare_index_group(group_df, precomputed_values):
        """Select finite observations from a prepared group and its required cache."""
        valid = np.isfinite(precomputed_values)
        return group_df.filter(pl.Series(valid)), precomputed_values[valid]

    @staticmethod
    def _prepare_public_index_group(group_df, index, precomputed_values):
        """Prepare a standalone processor call, retaining aligned cached values."""
        group_df = group_df.filter(pl.col("DOY") >= 90)
        if precomputed_values is None:
            values = group_df[index].to_numpy().astype(float)
            valid = np.isfinite(values)
            precomputed_values = np.full(values.shape, np.nan)
            precomputed_values[valid] = hampel_filter(values[valid], 3)
        else:
            precomputed_values = np.where(
                np.isfinite(group_df[index].to_numpy().astype(float)),
                precomputed_values,
                np.nan,
            )
        return group_df, precomputed_values

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
                finite observations remain after DOY filtering.
        """
        group_df, values = self._prepare_public_index_group(
            group_df, index, precomputed_values
        )
        return self._process_spectral_features_means(group_df, index, values)

    def _process_spectral_features_means(self, group_df, index, precomputed_values):
        """Calculate interpolation features from a prepared group and cache."""
        group_df, y = self._prepare_index_group(group_df, precomputed_values)
        if len(group_df) < 2:
            return None

        x = group_df["DOY"].to_numpy().astype(float)

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
                fewer than six finite observations remain or the fit is invalid.
        """
        group_df, values = self._prepare_public_index_group(
            group_df, index, precomputed_values
        )
        return self._process_spectral_features_curve(group_df, index, bounds, values)

    def _process_spectral_features_curve(
        self, group_df, index, bounds, precomputed_values
    ):
        """Fit a prepared group, leaving insufficient data unclassified."""
        field_id = group_df["field_id"][0] if len(group_df) else None
        group_df, y = self._prepare_index_group(group_df, precomputed_values)
        x = group_df["DOY"].to_numpy().astype(float)

        try:
            params = fit_curve(x, y, bounds)
        except InsufficientObservationsError as error:
            logger.warning("Field %s: %s", field_id, error)
            return None
        except RuntimeError:
            warnings.warn(
                f"The feature calculation failed for {field_id}, "
                "probably due to unsuitable/noisy data.",
                UserWarning,
                stacklevel=2,
            )
            return None

        wNDVI, mNDVI, S, A, mS, mA = params

        finer_values = double_logistic_function(self.x_values, *params)
        doy_max = float(self.x_values[np.argmax(finer_values)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fourth_derivative_values = get_fourth_derivative()(self.x_values, *params)
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
                days of year; None if no finite observations remain after DOY filtering.
        """
        group_df, values = self._prepare_public_index_group(
            group_df, index, precomputed_values
        )
        return self._process_ndyi_features(group_df, index, values)

    def _process_ndyi_features(self, group_df, index, precomputed_values):
        """Calculate extrema from a prepared group and cache."""
        group_df, y = self._prepare_index_group(group_df, precomputed_values)
        if len(group_df) == 0:
            return None
        x = group_df["DOY"].to_numpy().astype(float)

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
                with shape (n,). The internal cache contains every configured index.

        Returns:
            list[dict[str, object] | None]: Processor results in input group order,
                including skipped results as None.
        """
        func = self.pipelines[pipeline_name]

        if pipeline_name == "curve":
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(group, index, self.bounds_config[index], cache[index])
                for group, cache in grouped_data
            )

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(group, index, cache[index]) for group, cache in grouped_data
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
                logged. Built-in feature columns are always present. Each curve index
                includes fit_status: success, insufficient_observations, or fit_failed.
        """
        df = add_indices(df_prepared)

        id_columns = [c for c in df.columns if c in ["lat", "lon", "field_id"]]
        mapping_df = df.select(id_columns).unique()

        grouped_data = []
        group_ids = []
        for _, group in df.group_by("field_id"):
            group_ids.append(group["field_id"][0])
            filtered_group = group.filter(pl.col("DOY") >= 90)
            hampel_cache = {}
            for index in self.feature_config.keys():
                values = filtered_group[index].to_numpy().astype(float)
                valid = np.isfinite(values)
                filtered_values = np.full(values.shape, np.nan)
                filtered_values[valid] = hampel_filter(values[valid], 3)
                hampel_cache[index] = filtered_values

            grouped_data.append((filtered_group, hampel_cache))

        all_field_ids = df["field_id"].unique().to_frame()
        all_features = []
        feature_columns = []

        for index, cfg in self.feature_config.items():
            pipeline = cfg["pipeline"]

            results = self._run_pipeline(pipeline, index, grouped_data)
            successful = [r for r in results if r is not None]
            if pipeline == "curve":
                statuses = []
                for (_, cache), result in zip(grouped_data, results):
                    status = "success"
                    if result is None:
                        status = (
                            "insufficient_observations"
                            if np.isfinite(cache[index]).sum() < 6
                            else "fit_failed"
                        )
                    statuses.append(status)
                all_features.append(
                    pl.DataFrame(
                        {
                            "field_id": group_ids,
                            f"{index}_fit_status": statuses,
                        },
                        schema_overrides={"field_id": df.schema["field_id"]},
                    )
                )
            if pipeline == "curve" and len(successful) < len(results):
                too_short = statuses.count("insufficient_observations")
                failed = len(results) - len(successful) - too_short
                logger.warning(
                    "Double-logistic features unavailable for index '%s': "
                    "%d groups have fewer than six observations after DOY >= 90 "
                    "and finite-value filtering; "
                    "%d curve fits failed or returned invalid parameters "
                    "(for example, due to noisy data).",
                    index,
                    too_short,
                    failed,
                )
            results = successful
            if results:
                all_features.append(pl.DataFrame(results))

            suffixes = ["min", "max", "doy_min", "doy_max"]
            if pipeline == "curve":
                suffixes = [
                    "wNDVI",
                    "mNDVI",
                    "S",
                    "A",
                    "mS",
                    "mA",
                    "doy_max",
                    "max",
                    "start_of_growth",
                    "end_of_growth",
                    "start_of_decay",
                    "end_of_decay",
                    "max_growth",
                    "mean_growth",
                    "min_growth",
                    "max_decay",
                    "min_decay",
                    "mean_decay",
                ]
            if pipeline in ("mean", "curve", "extrema"):
                columns = [f"{index}_{suffix}" for suffix in suffixes]
                if pipeline in ("mean", "curve"):
                    columns += [f"median_{index}_fitted_{m}" for m in range(4, 11)]
                feature_columns.extend(columns)
                available = set().union(*(result.keys() for result in results))
                missing = [name for name in columns if name not in available]
                if missing:
                    all_features.append(
                        all_field_ids.with_columns(
                            pl.lit(None, dtype=pl.Float64).alias(name)
                            for name in missing
                        )
                    )

        merged = all_field_ids
        for feat_df in all_features:
            merged = merged.join(feat_df, on="field_id", how="left")

        merged = merged.join(mapping_df, on="field_id", how="left")
        merged = merged.with_columns(
            pl.col(name).cast(pl.Float64) for name in feature_columns
        )
        coordinates = [name for name in id_columns if name != "field_id"]
        return merged.select(
            ["field_id", *sorted(set(merged.columns) - set(id_columns)), *coordinates]
        )
