# Welch's t-test implementations for Polars DataFrames
#
# Each test is one vectorised Polars aggregation (all groups at once) followed by
# a Rust Student-t kernel for the p-values, so cost is a single pass over the data
# regardless of the number of groups.

import polars as pl
from typing import List, Literal, Optional, Union

from ._plugin import call

Alternative = Literal["two-sided", "greater", "less"]


def t_pvalue(t: pl.Expr, df: pl.Expr, alternative: Alternative = "two-sided") -> pl.Expr:
    """Student-t p-value expression for a t statistic and degrees of freedom."""
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
    return call("t_pvalue", t, df, kwargs={"alternative": alternative})


def _prepare(df, columns: List[str], group_by) -> tuple:
    lf = df.lazy()
    names = lf.collect_schema().names()
    for col in columns:
        if col not in names:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    by = [group_by] if isinstance(group_by, str) else list(group_by or [])
    for col in by:
        if col not in names:
            raise ValueError(f"Group column '{col}' not found in DataFrame")
    return lf, by


def _aggregate(lf: pl.LazyFrame, by: List[str], aggs: List[pl.Expr]) -> pl.LazyFrame:
    if by:
        return lf.group_by(by, maintain_order=True).agg(aggs)
    return lf.select(aggs)


def _finish(stats: pl.LazyFrame, ok: pl.Expr, t: pl.Expr, dof: pl.Expr,
            nullable: List[str], alternative: Alternative, order: List[str]) -> pl.DataFrame:
    """Mask insufficient samples, add p-values, and order columns."""
    return (
        stats.with_columns(
            t_statistic=t,
            df=dof.cast(pl.Float64),
        )
        .with_columns([pl.when(ok).then(pl.col(c)).alias(c) for c in nullable + ["t_statistic", "df"]])
        .with_columns(p_value=t_pvalue(pl.col("t_statistic"), pl.col("df"), alternative))
        .with_columns(
            alternative=pl.lit(alternative),
            **{"significant_at_0.05": pl.col("p_value") < 0.05},
        )
        .select(order)
        .collect()
    )


_RESULT_TAIL = ["t_statistic", "df", "p_value", "alternative", "significant_at_0.05"]
_TWO_SAMPLE = ["n1", "n2", "mean1", "mean2", "std1", "std2"]


def one_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
    mu: float = 0.0,
    alternative: Alternative = "two-sided",
    group_by: Optional[Union[str, list[str]]] = None,
) -> pl.DataFrame:
    """
    Perform one-sample Welch's t-test on a single column.

    Tests whether the mean of the population differs from a hypothesized value (mu).

    Args:
        df: Polars DataFrame or LazyFrame
        column: Name of the column to test
        mu: Hypothesized population mean (default: 0.0)
        alternative: Alternative hypothesis:
            - "two-sided": mean != mu
            - "greater": mean > mu
            - "less": mean < mu
        group_by: Optional column name(s) to group by before testing

    Returns:
        DataFrame with columns:
            - group columns (if group_by specified)
            - n: sample size
            - mean: sample mean
            - std: sample standard deviation
            - t_statistic: Welch's t-statistic
            - df: degrees of freedom
            - p_value: p-value
            - alternative: direction of test
            - significant_at_0.05: boolean indicator
    """
    lf, by = _prepare(df, [column], group_by)
    x = pl.col(column)
    stats = _aggregate(lf, by, [
        x.count().cast(pl.Int64).alias("n"),
        x.mean().alias("mean"),
        x.std().alias("std"),
    ])
    n, mean, std = pl.col("n"), pl.col("mean"), pl.col("std")
    return _finish(
        stats,
        ok=n >= 2,
        t=(mean - mu) / (std / n.sqrt()),
        dof=n - 1,
        nullable=["mean", "std"],
        alternative=alternative,
        order=by + ["n", "mean", "std"] + _RESULT_TAIL,
    )


def _welch(stats: pl.LazyFrame, alternative: Alternative, order: List[str]) -> pl.DataFrame:
    n1, n2 = pl.col("n1"), pl.col("n2")
    a = pl.col("std1").pow(2) / n1
    b = pl.col("std2").pow(2) / n2
    welch_df = (a + b).pow(2) / (a.pow(2) / (n1 - 1) + b.pow(2) / (n2 - 1))
    dof = pl.when((a > 0) & (b > 0)).then(welch_df).otherwise(n1 + n2 - 2)
    return _finish(
        stats,
        ok=(n1 >= 2) & (n2 >= 2),
        t=(pl.col("mean1") - pl.col("mean2")) / (a + b).sqrt(),
        dof=dof,
        nullable=["mean1", "mean2", "std1", "std2"],
        alternative=alternative,
        order=order,
    )


def two_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column1: str,
    column2: Optional[str] = None,
    group_column: Optional[str] = None,
    alternative: Alternative = "two-sided",
    group_by: Optional[Union[str, list[str]]] = None,
) -> pl.DataFrame:
    """
    Perform two-sample Welch's t-test.

    Two modes of operation:
    1. Two columns mode: Compare values in column1 vs column2
    2. Grouping mode: Compare two groups defined by group_column

    Args:
        df: Polars DataFrame or LazyFrame
        column1: Name of first column (or the value column in grouping mode)
        column2: Name of second column (required in two-columns mode)
        group_column: Column defining groups (required in grouping mode, must have exactly 2 unique values)
        alternative: Alternative hypothesis:
            - "two-sided": mean1 != mean2
            - "greater": mean1 > mean2
            - "less": mean1 < mean2
        group_by: Optional column name(s) to group by before testing. In grouping
            mode, groups without exactly 2 levels of group_column are skipped.

    Returns:
        DataFrame with columns:
            - group columns (if group_by specified)
            - group1, group2: group labels, sorted (grouping mode only)
            - n1, n2: sample sizes
            - mean1, mean2: sample means
            - std1, std2: sample standard deviations
            - t_statistic: Welch's t-statistic
            - df: Welch-Satterthwaite degrees of freedom
            - p_value: p-value
            - alternative: direction of test
            - significant_at_0.05: boolean indicator
    """
    if column2 is None and group_column is None:
        raise ValueError("Must specify either column2 or group_column")
    if column2 is not None and group_column is not None:
        raise ValueError("Cannot specify both column2 and group_column")

    # Two columns mode
    if column2 is not None:
        lf, by = _prepare(df, [column1, column2], group_by)
        x1, x2 = pl.col(column1), pl.col(column2)
        stats = _aggregate(lf, by, [
            x1.count().cast(pl.Int64).alias("n1"),
            x2.count().cast(pl.Int64).alias("n2"),
            x1.mean().alias("mean1"),
            x2.mean().alias("mean2"),
            x1.std().alias("std1"),
            x2.std().alias("std2"),
        ])
        return _welch(stats, alternative, by + _TWO_SAMPLE + _RESULT_TAIL)

    # Grouping mode
    lf, by = _prepare(df, [column1], group_by)
    if group_column not in lf.collect_schema().names():
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
    x, level = pl.col(column1), pl.col(group_column)

    # Stage 1: stats per (by-group, level). Stage 2: pivot the two sorted levels side by side.
    per_level = (
        lf.with_row_index("__row")
        .filter(level.is_not_null())
        .group_by(by + [group_column])
        .agg(
            x.count().cast(pl.Int64).alias("__n"),
            x.mean().alias("__mean"),
            x.std().alias("__std"),
            pl.col("__row").min().alias("__first"),
        )
    )
    pick = lambda c, i: pl.col(c).sort_by(group_column).get(i)
    wide = _aggregate(per_level, by, [
        pl.len().alias("__levels"),
        pl.col("__first").min(),
        pick(group_column, 0).cast(pl.String).alias("group1"),
        pick(group_column, -1).cast(pl.String).alias("group2"),
        pick("__n", 0).alias("n1"),
        pick("__n", -1).alias("n2"),
        pick("__mean", 0).alias("mean1"),
        pick("__mean", -1).alias("mean2"),
        pick("__std", 0).alias("std1"),
        pick("__std", -1).alias("std2"),
    ]).collect()  # one row per by-group
    if by:
        wide = wide.filter(pl.col("__levels") == 2).sort("__first")
    elif wide["__levels"][0] != 2:
        raise ValueError(f"Group column must have exactly 2 unique values, found {wide['__levels'][0]}")

    return _welch(wide.lazy(), alternative, by + ["group1", "group2"] + _TWO_SAMPLE + _RESULT_TAIL)
