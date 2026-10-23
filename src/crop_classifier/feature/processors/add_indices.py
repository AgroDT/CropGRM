import polars as pl


def add_indices(df: pl.DataFrame) -> pl.DataFrame:
    """Add spectral indices to a band table.

    Args:
        df (polars.DataFrame): Table containing green, blue, nir, swir1, and red
            reflectance columns on the same numeric scale.

    Returns:
        polars.DataFrame: Input columns with dimensionless ndyi, ndmi, and wrdvi
            index columns added or replaced.
    """
    return df.with_columns(
        [
            (
                (pl.col("green") - pl.col("blue")) / (pl.col("green") + pl.col("blue"))
            ).alias("ndyi"),
            (
                (pl.col("nir") - pl.col("swir1")) / (pl.col("nir") + pl.col("swir1"))
            ).alias("ndmi"),
            (
                (pl.col("nir") * 0.1 - pl.col("red"))
                / (pl.col("nir") * 0.1 + pl.col("red"))
            ).alias("wrdvi"),
        ]
    )
