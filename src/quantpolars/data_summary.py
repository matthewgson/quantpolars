# Data summary tools for big data analysis

import polars as pl
from typing import Union
try:
    from great_tables import GT, html
    GT_AVAILABLE = True
except ImportError:
    GT_AVAILABLE = False


class DataSummary:
    """
    A class to hold summary statistics for a DataFrame and provide various display options.
    
    Attributes:
        df: pl.DataFrame containing the summary statistics
    """
    
    def __init__(self, summary_df: pl.DataFrame):
        """
        Initialize DataSummary with a summary statistics DataFrame.
        
        Args:
            summary_df: DataFrame containing summary statistics
        """
        self.df = summary_df
    
    def to_gt(self) -> "GT":
        """
        Convert the summary statistics to a styled GT table.
        
        Returns:
            GT table with formatted display
            
        Raises:
            ImportError: If great_tables is not available
        """
        if not GT_AVAILABLE:
            raise ImportError("great_tables is required for styled output. Install with: pip3 install great-tables")
        
        # Create a copy of the dataframe for GT formatting
        gt_df = self.df.clone()
        
        # Convert integer date representations back to date objects for date-type rows
        from datetime import date, timedelta
        epoch = date(1970, 1, 1)
        
        # Create formatted string representations for date columns
        min_formatted = []
        max_formatted = []
        
        for row in gt_df.rows():
            var_type = row[1]  # type column
            min_val = row[6]   # min column
            max_val = row[7]   # max column
            
            if var_type == "date":
                if min_val is not None:
                    # min_val should already be a date object
                    if isinstance(min_val, date):
                        min_val = f"{min_val.month}/{min_val.day}/{min_val.year}"
                    else:
                        min_val = ""
                else:
                    min_val = ""
                if max_val is not None:
                    # max_val should already be a date object
                    if isinstance(max_val, date):
                        max_val = f"{max_val.month}/{max_val.day}/{max_val.year}"
                    else:
                        max_val = ""
                else:
                    max_val = ""
            elif var_type == "numeric":
                # Convert numeric values to strings
                min_val = str(min_val) if min_val is not None else ""
                max_val = str(max_val) if max_val is not None else ""
            else:
                # For categorical and other types
                min_val = ""
                max_val = ""
            
            min_formatted.append(min_val)
            max_formatted.append(max_val)
        
        # Replace the columns with the formatted string values
        gt_df = gt_df.with_columns([
            pl.Series("min", min_formatted, dtype=pl.Utf8).alias("min"),
            pl.Series("max", max_formatted, dtype=pl.Utf8).alias("max")
        ])
        
        # Convert to GT table with formatting - GT supports Polars DataFrames natively
        gt_table = (
            GT(gt_df)
            .tab_header(
                title="Data Summary Statistics",
                subtitle=f"Analysis of {len(gt_df)} variables"
            )
            .fmt_number(
                columns=["mean", "sd", "p1", "p5", "p25", "p50", "p75", "p95", "p99"],
                decimals=2
            )
            .fmt_percent(
                columns=["pct_missing"],
                decimals=1
            )
            .cols_label(
                variable="Variable",
                type="Type", 
                nobs="N Obs",
                pct_missing="% Missing",
                mean="Mean",
                sd="Std Dev",
                min="Min",
                max="Max",
                p1="1%",
                p5="5%",
                p25="25%",
                p50="50%",
                p75="75%",
                p95="95%",
                p99="99%",
                n_unique="N Unique"
            )
        )
        return gt_table
    
    def __repr__(self) -> str:
        """Return string representation of the summary DataFrame."""
        return self.df.__repr__()
    
    def __str__(self) -> str:
        """Return string representation of the summary DataFrame."""
        return self.df.__str__()


def sm(df: Union[pl.DataFrame, pl.LazyFrame]) -> DataSummary:
    """
    Create comprehensive summary statistics for all columns in a single pass.

    Args:
        df: pl.DataFrame or pl.LazyFrame

    Returns:
        DataSummary object containing the summary statistics DataFrame.
        Use .df to access the raw DataFrame or .to_gt() for styled output.
    """
    is_lazy = isinstance(df, pl.LazyFrame)
    schema = df.collect_schema() if is_lazy else df.schema
    total_rows = df.select(pl.len()).collect().item() if is_lazy else len(df)

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

    # Build statistics expressions using list comprehensions (more efficient than for loops)
    all_stats = []

    # Numeric columns - vectorized operations
    if numeric_cols:
        numeric_stats = [
            stat_expr
            for col in numeric_cols
            for stat_expr in [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("numeric").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                pl.col(col).drop_nulls().mean().alias(f"{col}_mean"),
                pl.col(col).drop_nulls().std().alias(f"{col}_sd"),
                pl.col(col).drop_nulls().min().alias(f"{col}_min"),
                pl.col(col).drop_nulls().max().alias(f"{col}_max"),
                pl.col(col).drop_nulls().quantile(0.01).alias(f"{col}_p1"),
                pl.col(col).drop_nulls().quantile(0.05).alias(f"{col}_p5"),
                pl.col(col).drop_nulls().quantile(0.25).alias(f"{col}_p25"),
                pl.col(col).drop_nulls().quantile(0.50).alias(f"{col}_p50"),
                pl.col(col).drop_nulls().quantile(0.75).alias(f"{col}_p75"),
                pl.col(col).drop_nulls().quantile(0.95).alias(f"{col}_p95"),
                pl.col(col).drop_nulls().quantile(0.99).alias(f"{col}_p99"),
                pl.col(col).drop_nulls().n_unique().cast(pl.Int64).alias(f"{col}_n_unique"),
            ]
        ]
        all_stats.extend(numeric_stats)

    # Date columns - vectorized operations
    if date_cols:
        date_stats = [
            stat_expr
            for col in date_cols
            for stat_expr in [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("date").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_mean"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_sd"),
                pl.col(col).drop_nulls().min().alias(f"{col}_min"),
                pl.col(col).drop_nulls().max().alias(f"{col}_max"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p1"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p5"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p25"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p50"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p75"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p95"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_p99"),
                pl.col(col).drop_nulls().n_unique().cast(pl.Int64).alias(f"{col}_n_unique"),
            ]
        ]
        all_stats.extend(date_stats)

    # Categorical columns - vectorized operations
    if categorical_cols:
        categorical_stats = [
            stat_expr
            for col in categorical_cols
            for stat_expr in [
                pl.lit(col).alias(f"{col}_variable"),
                pl.lit("categorical").alias(f"{col}_type"),
                pl.col(col).count().cast(pl.Int64).alias(f"{col}_nobs"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_mean"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_sd"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_min"),
                pl.lit(None).cast(pl.Float64).alias(f"{col}_max"),
                *[
                    pl.lit(None).cast(pl.Float64).alias(f"{col}_{stat}")
                    for stat in [
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
        ]
        all_stats.extend(categorical_stats)

    # Single pass through data
    result = df.select(all_stats)

    # Collect if lazy
    if is_lazy:
        result = result.collect(engine="streaming")

    # Reshape from wide to long format using Polars operations
    all_cols = numeric_cols + date_cols + categorical_cols

    # Create the summary dataframe more efficiently
    summary_data = []
    for col in all_cols:
        row_dict = {
            "variable": result[f"{col}_variable"][0],
            "type": result[f"{col}_type"][0],
            "nobs": result[f"{col}_nobs"][0],
            "pct_missing": round((total_rows - result[f"{col}_nobs"][0]) / total_rows, 4) if total_rows > 0 else 0.0,
            "mean": result[f"{col}_mean"][0] if col in numeric_cols else None,
            "sd": result[f"{col}_sd"][0] if col in numeric_cols else None,
            "min": result[f"{col}_min"][0] if col in numeric_cols or col in date_cols else None,
            "max": result[f"{col}_max"][0] if col in numeric_cols or col in date_cols else None,
            "p1": result[f"{col}_p1"][0] if col in numeric_cols or col in date_cols else None,
            "p5": result[f"{col}_p5"][0] if col in numeric_cols or col in date_cols else None,
            "p25": result[f"{col}_p25"][0] if col in numeric_cols or col in date_cols else None,
            "p50": result[f"{col}_p50"][0] if col in numeric_cols or col in date_cols else None,
            "p75": result[f"{col}_p75"][0] if col in numeric_cols or col in date_cols else None,
            "p95": result[f"{col}_p95"][0] if col in numeric_cols or col in date_cols else None,
            "p99": result[f"{col}_p99"][0] if col in numeric_cols or col in date_cols else None,
            "n_unique": result[f"{col}_n_unique"][0],
        }
        summary_data.append(row_dict)

    summary_df = pl.DataFrame(summary_data)

    # Sort by type using Polars' when/then for efficiency
    type_order = {"date": 0, "categorical": 1, "numeric": 2}
    summary_df = (
        summary_df.with_columns(
            pl.col("type").replace(type_order).alias("type_order")
        )
        .sort("type_order")
        .drop("type_order")
    )

    return DataSummary(summary_df)