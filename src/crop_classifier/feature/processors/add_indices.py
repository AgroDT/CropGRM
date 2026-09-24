import polars as pl

def add_indices(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate indices"""
    return df.with_columns([
        ((pl.col("green") - pl.col("blue")) / (pl.col("green") + pl.col("blue"))).alias("ndyi"),
        ((pl.col("nir") - pl.col("swir1")) / (pl.col("nir") + pl.col("swir1"))).alias("ndmi"),
        ((pl.col("nir") * 0.1 - pl.col("red")) / (pl.col("nir") * 0.1 + pl.col("red"))).alias("wrdvi"),
    ])