from pathlib import Path
from typing import Dict, List, Optional, Union
import polars as pl

class MeteoFeatureCalculator():
    def calculate_meteo_features(
        self, 
        df, 
    ) -> pl.DataFrame:
    
        condition = df["temperature"] > 283
    
        com_calculated = (
            df.group_by(["field_id", "month"])
            .agg([
                pl.col("temperature").median().alias("median_t"),
                pl.col("precipitation").median().alias("median_prec"),
                pl.col("precipitation").sum().alias("sum_prec")
            ])
        )
    
        temp_calculated = (
            df.filter(condition)
            .group_by(["field_id", "month"])
            .agg([
                pl.col("temperature").sum().alias("sum_t")
            ])
        )
    
        calculated_df = com_calculated.join(
            temp_calculated,
            on=["field_id", "month"],
            how="left"
        )
    
        result_df = None
    
        for col in ["median_t", "sum_t", "median_prec", "sum_prec"]:
            pivot_df = calculated_df.pivot(
                index=["field_id"],
                on="month",
                values=col,
                aggregate_function=None
            )
    
            pivot_df = pivot_df.rename({
                str(m): f"{col}_{m}"
                for m in calculated_df["month"].unique().to_list()
            })
    
            if result_df is None:
                result_df = pivot_df
            else:
                result_df = result_df.join(pivot_df, on="field_id", how="left")
    
        return result_df