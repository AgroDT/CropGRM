from pathlib import Path
from typing import List, Union

import polars as pl


class DataPreparer:
    """Normalize field and pixel observations for feature calculation."""

    @staticmethod
    def read_dataset(path: Union[str, Path]) -> pl.LazyFrame:
        """Open a supported table as a lazy dataset.

        Args:
            path (str | Path): Input path ending in .parquet or .csv, with
                case-sensitive matching.

        Returns:
            polars.LazyFrame: Lazy scan of the input table.

        Raises:
            ValueError: If the path has an unsupported suffix.
        """
        path_str = str(path)
        if path_str.endswith(".parquet"):
            return pl.scan_parquet(path)
        elif path_str.endswith(".csv"):
            return pl.scan_csv(path)
        else:
            raise ValueError(f"Unsupported data format: {path}")

    @staticmethod
    def add_doy_and_month(df: pl.DataFrame) -> pl.DataFrame:
        """Normalize dates and derive calendar fields.

        Args:
            df (polars.DataFrame): Table with a date column; strings must use YYYY-MM-DD
                and other values must be castable to Date.

        Returns:
            polars.DataFrame: Table with date normalized to Date and year, DOY
                (day of year), and month columns added or replaced.

        Raises:
            ValueError: If date is missing, invalid, or contains null values.
        """
        if "date" not in df.columns:
            raise ValueError("Input data must contain a 'date' column.")

        date_dtype = df.schema["date"]
        if date_dtype == pl.Utf8:
            date_expr = pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        elif date_dtype == pl.Date:
            date_expr = pl.col("date")
        else:
            date_expr = pl.col("date").cast(pl.Date)

        try:
            df = df.with_columns(date_expr.alias("date"))
        except pl.exceptions.PolarsError as exc:
            raise ValueError("Column 'date' must contain valid dates.") from exc
        if df["date"].null_count():
            raise ValueError("Column 'date' must not contain null values.")

        return df.with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.ordinal_day().alias("DOY"),
                pl.col("date").dt.month().alias("month"),
            ]
        )

    def prepare_field_data(
        self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]
    ) -> pl.DataFrame:
        """Convert wide field statistics into dated band observations.

        Args:
            df (polars.DataFrame): Wide table with field_id and band columns whose final
                name components are YYYYMMDD_band.
            var_group (list[str]): Band names used to select and cast measurement
                columns.
            id_cols (list[str]): Identifier columns retained during unpivoting; must
                include field_id.

        Returns:
            polars.DataFrame: Rows sorted by field_id and date, with date, year, DOY,
                month, and available numeric band columns.
        """
        cols = [c for c in df.columns if any(band in c for band in var_group)]

        df_long = df.unpivot(
            index=id_cols, on=cols, variable_name="variable", value_name="value"
        ).filter(pl.col("value").is_not_null())

        df_long = df_long.with_columns(
            [
                pl.col("variable").str.split("_").list.get(-2).alias("date"),
                pl.col("variable").str.split("_").list.get(-1).alias("band"),
            ]
        )

        df_long = df_long.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))

        result_df = df_long.pivot(
            values="value",
            index=["field_id"] + ["date"],
            on="band",
            aggregate_function="first",
        )

        available_bands = [b for b in var_group if b in result_df.columns]
        result_df = result_df.with_columns([pl.col(available_bands).cast(pl.Float64)])

        result_df = self.add_doy_and_month(result_df)
        return result_df.sort(["field_id", "date"])

    @staticmethod
    def ensure_field_id(df: Union[pl.DataFrame, pl.LazyFrame], id_cols: List[str]):
        """Add a spatial identifier when field_id is absent.

        Args:
            df (polars.DataFrame | polars.LazyFrame): Table containing the requested
                identifier columns.
            id_cols (list[str]): Nonempty list of columns converted to strings and
                joined with underscores if needed.

        Returns:
            polars.DataFrame | polars.LazyFrame: Table of the same kind with field_id;
                an existing field_id is preserved.
        """
        cols = (
            df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        )
        if "field_id" in cols:
            return df

        if len(id_cols) == 1:
            id_expr = pl.col(id_cols[0]).cast(pl.Utf8).alias("field_id")
        else:
            id_expr = pl.concat_str(
                [pl.col(c).cast(pl.Utf8) for c in id_cols], separator="_"
            ).alias("field_id")

        return df.with_columns(id_expr)

    def prepare_pixel_data(
        self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]
    ) -> pl.DataFrame:
        """Aggregate valid pixel measurements by spatial key and date.

        Args:
            df (polars.DataFrame): Pixel table with date, spatial identifiers, and
                numeric band columns.
            var_group (list[str]): Bands to aggregate; at least one must be present.
            id_cols (list[str]): Columns identifying each spatial entity.

        Returns:
            polars.DataFrame: Median bands per spatial key and calendar date, with
                year, DOY, and month, sorted by spatial keys and date.

        Raises:
            ValueError: If date is missing, invalid, or contains null values.
        """
        df = self.add_doy_and_month(df)

        available_bands = [b for b in var_group if b in df.columns]

        result_df = df.with_columns(
            [
                pl.when(pl.col(b).is_finite()).then(pl.col(b)).otherwise(None).alias(b)
                for b in available_bands
            ]
        )

        result_df = result_df.filter(
            pl.any_horizontal([pl.col(b).is_not_null() for b in available_bands])
        )

        result_df = result_df.group_by([*id_cols, "date", "year", "DOY", "month"]).agg(
            [pl.col(b).drop_nulls().median().alias(b) for b in available_bands]
        )

        return result_df.sort([*id_cols, "date"])

    def prepare_data_chunk(
        self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]
    ) -> pl.DataFrame:
        """Prepare pixel observations or wide field statistics.

        Args:
            df (polars.DataFrame): Input table; a date column selects pixel preparation,
                otherwise wide field preparation is used.
            var_group (list[str]): Names of the bands to prepare.
            id_cols (list[str]): Spatial identifier columns; field statistics require
                field_id.

        Returns:
            polars.DataFrame: Prepared observations with band values, date, year,
                DOY, month, and spatial identifiers.
        """
        if "date" in df.columns:
            return self.prepare_pixel_data(df, var_group, id_cols)
        else:
            return self.prepare_field_data(df, var_group, id_cols)
