# Data summary tools for big data analysis

import polars as pl


def sm(df):
    """
    Create comprehensive summary statistics for all columns in a single pass.

    Args:
        df: pl.DataFrame or pl.LazyFrame

    Returns:
        pl.DataFrame with summary statistics
    """
    is_lazy = isinstance(df, pl.LazyFrame)
    schema = df.collect_schema() if is_lazy else df.schema

    # Separate columns by type
    numeric_cols = [
        c
        for c, t in schema.items()
        if t
        in [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
            pl.Decimal,
        ]
    ]
    date_cols = [c for c, t in schema.items() if t in [pl.Date, pl.Datetime]]
    categorical_cols = [
        c for c in schema.keys() if c not in numeric_cols and c not in date_cols
    ]

    # Build all statistics in ONE select operation
    all_stats = []

    # Numeric columns - all at once
    for col in numeric_cols:
        all_stats.extend(
            [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("numeric").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                pl.col(col).drop_nans().mean().alias(f"{col}_mean"),
                pl.col(col).drop_nans().std().alias(f"{col}_sd"),
                pl.col(col).drop_nans().quantile(0.01).alias(f"{col}_p1"),
                pl.col(col).drop_nans().quantile(0.05).alias(f"{col}_p5"),
                pl.col(col).drop_nans().quantile(0.25).alias(f"{col}_p25"),
                pl.col(col).drop_nans().quantile(0.50).alias(f"{col}_p50"),
                pl.col(col).drop_nans().quantile(0.75).alias(f"{col}_p75"),
                pl.col(col).drop_nans().quantile(0.95).alias(f"{col}_p95"),
                pl.col(col).drop_nans().quantile(0.99).alias(f"{col}_p99"),
                pl.col(col).drop_nulls().n_unique().cast(pl.Int64).alias(f"{col}_n_unique"),
            ]
        )

    # Date columns
    for col in date_cols:
        all_stats.extend(
            [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("date").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                *[
                    pl.lit(None).cast(pl.Float64).alias(f"{col}_{stat}")
                    for stat in [
                        "mean",
                        "sd",
                        "p1",
                        "p5",
                        "p25",
                        "p50",
                        "p75",
                        "p95",
                        "p99",
                    ]
                ],
                pl.col(col).drop_nulls().n_unique().cast(pl.Int64).alias(f"{col}_n_unique"),
            ]
        )

    # Categorical columns
    for col in categorical_cols:
        all_stats.extend(
            [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("categorical").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                *[
                    pl.lit(None).cast(pl.Float64).alias(f"{col}_{stat}")
                    for stat in [
                        "mean",
                        "sd",
                        "p1",
                        "p5",
                        "p25",
                        "p50",
                        "p75",
                        "p95",
                        "p99",
                    ]
                ],
                pl.col(col).drop_nulls().n_unique().cast(pl.Int64).alias(f"{col}_n_unique"),
            ]
        )

    # Single pass through data
    result = df.select(all_stats)

    # Collect if lazy
    if is_lazy:
        result = result.collect(engine="streaming")

    # Reshape from wide to long format
    all_cols = numeric_cols + date_cols + categorical_cols
    rows = []

    for col in all_cols:
        row_data = {
            "variable": result[f"{col}_variable"][0],
            "type": result[f"{col}_type"][0],
            "nobs": result[f"{col}_nobs"][0],
            "mean": result[f"{col}_mean"][0]
            if f"{col}_mean" in result.columns
            else None,
            "sd": result[f"{col}_sd"][0] if f"{col}_sd" in result.columns else None,
            "p1": result[f"{col}_p1"][0],
            "p5": result[f"{col}_p5"][0],
            "p25": result[f"{col}_p25"][0],
            "p50": result[f"{col}_p50"][0],
            "p75": result[f"{col}_p75"][0],
            "p95": result[f"{col}_p95"][0],
            "p99": result[f"{col}_p99"][0],
            "n_unique": result[f"{col}_n_unique"][0],
        }
        rows.append(row_data)

    summary_df = pl.DataFrame(rows)

    # Sort by type
    type_order = {"date": 0, "categorical": 1, "numeric": 2}
    summary_df = (
        summary_df.with_columns(pl.col("type").replace(type_order).alias("type_order"))
        .sort("type_order")
        .drop("type_order")
    )

    return summary_df