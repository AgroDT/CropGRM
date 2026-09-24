from pathlib import Path
from typing import List, Union
import polars as pl

class DataPreparer:
    """Data preparing for calculation."""
    @staticmethod
    def read_dataset(path: Union[str, Path]) -> pl.LazyFrame:
        path_str = str(path)
        if path_str.endswith(".parquet"):
            return pl.scan_parquet(path)
        elif path_str.endswith(".csv"):
            return pl.scan_csv(path)
        else:
            raise ValueError(f"Unsupported data format: {path}")

    @staticmethod
    def add_doy_and_month(df: pl.DataFrame) -> pl.DataFrame:
        date_dtype = df.schema["date"]
        if date_dtype == pl.Utf8:
            date_expr = pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        elif date_dtype == pl.Date:
            date_expr = pl.col("date")
        else:
            date_expr = pl.col("date").cast(pl.Date)
        
        return df.with_columns([
            date_expr.dt.ordinal_day().alias("DOY"),
            date_expr.dt.month().alias("month"),
        ])

    def prepare_field_data(self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]) -> pl.DataFrame:
        """
        Transform the field data table from wide format with normalization 'field_id'.
        """
        cols = [c for c in df.columns if any(band in c for band in var_group)]
    
        df_long = df.unpivot(
            index=id_cols,
            on=cols,
            variable_name="variable",
            value_name="value"
        ).filter(pl.col("value").is_not_null())
    
        df_long = df_long.with_columns([
            pl.col("variable").str.split("_").list.get(-2).alias("date"),
            pl.col("variable").str.split("_").list.get(-1).alias("band"),
        ])
    
        df_long = df_long.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y%m%d")
        )
    
        result_df = df_long.pivot(
            values="value",
            index=["field_id"] + id_cols + ["date"],
            on="band",
            aggregate_function="first"
        )
    
        available_bands = [b for b in var_group if b in result_df.columns]
        result_df = result_df.with_columns([
            pl.col(available_bands).cast(pl.Float64)
        ])
    
        result_df = self.add_doy_and_month(result_df)
        return result_df.sort(["field_id", "DOY"])

    @staticmethod
    def ensure_field_id(
        df: Union[pl.DataFrame, pl.LazyFrame], id_cols: List[str]
    ):
        """Создает единую колонку field_id для фильтрации и джойна."""
        cols = (
            df.collect_schema().names()
            if isinstance(df, pl.LazyFrame)
            else df.columns
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

    def prepare_pixel_data(self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]) -> pl.DataFrame:
        """
        Processing pixel-data get from tiff.
        """
        if "date" not in df.columns:
            raise ValueError("Parquet файл должен содержать колонку 'date'")
        
        result_df = self.add_doy_and_month(df)
        
        return result_df.sort(["field_id", "DOY"])
    
    def prepare_data_chunk(
            self, df: pl.DataFrame, var_group: List[str], id_cols: List[str]
        ) -> pl.DataFrame:
        """Method choosing depends on data format."""
        if "date" in df.columns:
            return self.prepare_pixel_data(df, var_group, id_cols)
        else:
            return self.prepare_field_data(df, var_group, id_cols)
            