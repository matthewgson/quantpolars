# Welch's t-test implementations for Polars DataFrames

import polars as pl
from typing import Union, Optional, Literal
import math


def one_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
    mu: float = 0.0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
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
    is_lazy = isinstance(df, pl.LazyFrame)
    working_df = df if not is_lazy else df.collect()

    # Validate column exists
    if column not in working_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Define the test function
    def perform_test(data: pl.DataFrame) -> dict:
        """Perform one-sample t-test on a group."""
        # Filter out nulls and get values
        values = data.select(pl.col(column).drop_nulls()).to_series()
        n = len(values)

        if n < 2:
            return {
                "n": n,
                "mean": None,
                "std": None,
                "t_statistic": None,
                "df": None,
                "p_value": None,
                "alternative": alternative,
                "significant_at_0.05": None,
            }

        # Calculate statistics
        mean = values.mean()
        std = values.std()
        se = std / math.sqrt(n)

        # Calculate t-statistic
        t_stat = (mean - mu) / se if se > 0 else float("inf")

        # Degrees of freedom
        df_val = n - 1

        # Calculate p-value using Student's t-distribution approximation
        from scipy import stats

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
        elif alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df_val)
        else:  # less
            p_value = stats.t.cdf(t_stat, df_val)

        return {
            "n": n,
            "mean": float(mean),
            "std": float(std),
            "t_statistic": float(t_stat),
            "df": float(df_val),
            "p_value": float(p_value),
            "alternative": alternative,
            "significant_at_0.05": p_value < 0.05,
        }

    # Perform test(s)
    if group_by is None:
        # Single test on entire dataset
        result = perform_test(working_df)
        result_df = pl.DataFrame([result])
    else:
        # Group by and perform tests
        if isinstance(group_by, str):
            group_by = [group_by]

        # Validate group columns exist
        for col in group_by:
            if col not in working_df.columns:
                raise ValueError(f"Group column '{col}' not found in DataFrame")

        groups = working_df.group_by(group_by, maintain_order=True)
        results = []

        for group_values, group_df in groups:
            result = perform_test(group_df)
            # Add group values
            if len(group_by) == 1:
                result[group_by[0]] = group_values
            else:
                for i, col in enumerate(group_by):
                    result[col] = group_values[i]
            results.append(result)

        result_df = pl.DataFrame(results)
        # Reorder columns to put group columns first
        other_cols = [c for c in result_df.columns if c not in group_by]
        result_df = result_df.select(group_by + other_cols)

    return result_df


def two_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column1: str,
    column2: Optional[str] = None,
    group_column: Optional[str] = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
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
        group_by: Optional column name(s) to group by before testing

    Returns:
        DataFrame with columns:
            - group columns (if group_by specified)
            - n1, n2: sample sizes
            - mean1, mean2: sample means
            - std1, std2: sample standard deviations
            - t_statistic: Welch's t-statistic
            - df: Welch-Satterthwaite degrees of freedom
            - p_value: p-value
            - alternative: direction of test
            - significant_at_0.05: boolean indicator
    """
    is_lazy = isinstance(df, pl.LazyFrame)
    working_df = df if not is_lazy else df.collect()

    # Validate inputs
    if column2 is None and group_column is None:
        raise ValueError("Must specify either column2 or group_column")
    if column2 is not None and group_column is not None:
        raise ValueError("Cannot specify both column2 and group_column")

    # Two columns mode
    if column2 is not None:
        if column1 not in working_df.columns:
            raise ValueError(f"Column '{column1}' not found in DataFrame")
        if column2 not in working_df.columns:
            raise ValueError(f"Column '{column2}' not found in DataFrame")

        def perform_test(data: pl.DataFrame) -> dict:
            """Perform two-sample t-test comparing two columns."""
            values1 = data.select(pl.col(column1).drop_nulls()).to_series()
            values2 = data.select(pl.col(column2).drop_nulls()).to_series()

            n1, n2 = len(values1), len(values2)

            if n1 < 2 or n2 < 2:
                return {
                    "n1": n1,
                    "n2": n2,
                    "mean1": None,
                    "mean2": None,
                    "std1": None,
                    "std2": None,
                    "t_statistic": None,
                    "df": None,
                    "p_value": None,
                    "alternative": alternative,
                    "significant_at_0.05": None,
                }

            # Calculate statistics
            mean1, mean2 = values1.mean(), values2.mean()
            std1, std2 = values1.std(), values2.std()
            var1, var2 = std1**2, std2**2

            # Welch's standard error
            se = math.sqrt(var1 / n1 + var2 / n2)

            # Welch's t-statistic
            t_stat = (mean1 - mean2) / se if se > 0 else float("inf")

            # Welch-Satterthwaite degrees of freedom
            if var1 > 0 and var2 > 0:
                numerator = (var1 / n1 + var2 / n2) ** 2
                denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
                df_val = numerator / denominator if denominator > 0 else n1 + n2 - 2
            else:
                df_val = n1 + n2 - 2

            # Calculate p-value
            from scipy import stats

            if alternative == "two-sided":
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
            elif alternative == "greater":
                p_value = 1 - stats.t.cdf(t_stat, df_val)
            else:  # less
                p_value = stats.t.cdf(t_stat, df_val)

            return {
                "n1": n1,
                "n2": n2,
                "mean1": float(mean1),
                "mean2": float(mean2),
                "std1": float(std1),
                "std2": float(std2),
                "t_statistic": float(t_stat),
                "df": float(df_val),
                "p_value": float(p_value),
                "alternative": alternative,
                "significant_at_0.05": p_value < 0.05,
            }

    # Grouping mode
    else:
        if column1 not in working_df.columns:
            raise ValueError(f"Column '{column1}' not found in DataFrame")
        if group_column not in working_df.columns:
            raise ValueError(f"Group column '{group_column}' not found in DataFrame")

        def perform_test(data: pl.DataFrame) -> dict:
            """Perform two-sample t-test comparing two groups."""
            groups = data.select(pl.col(group_column).drop_nulls().unique()).to_series()

            if len(groups) != 2:
                raise ValueError(
                    f"Group column must have exactly 2 unique values, found {len(groups)}"
                )

            group1_val, group2_val = sorted(groups.to_list())

            values1 = (
                data.filter(pl.col(group_column) == group1_val)
                .select(pl.col(column1).drop_nulls())
                .to_series()
            )
            values2 = (
                data.filter(pl.col(group_column) == group2_val)
                .select(pl.col(column1).drop_nulls())
                .to_series()
            )

            n1, n2 = len(values1), len(values2)

            if n1 < 2 or n2 < 2:
                return {
                    "group1": str(group1_val),
                    "group2": str(group2_val),
                    "n1": n1,
                    "n2": n2,
                    "mean1": None,
                    "mean2": None,
                    "std1": None,
                    "std2": None,
                    "t_statistic": None,
                    "df": None,
                    "p_value": None,
                    "alternative": alternative,
                    "significant_at_0.05": None,
                }

            # Calculate statistics
            mean1, mean2 = values1.mean(), values2.mean()
            std1, std2 = values1.std(), values2.std()
            var1, var2 = std1**2, std2**2

            # Welch's standard error
            se = math.sqrt(var1 / n1 + var2 / n2)

            # Welch's t-statistic
            t_stat = (mean1 - mean2) / se if se > 0 else float("inf")

            # Welch-Satterthwaite degrees of freedom
            if var1 > 0 and var2 > 0:
                numerator = (var1 / n1 + var2 / n2) ** 2
                denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
                df_val = numerator / denominator if denominator > 0 else n1 + n2 - 2
            else:
                df_val = n1 + n2 - 2

            # Calculate p-value
            from scipy import stats

            if alternative == "two-sided":
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
            elif alternative == "greater":
                p_value = 1 - stats.t.cdf(t_stat, df_val)
            else:  # less
                p_value = stats.t.cdf(t_stat, df_val)

            return {
                "group1": str(group1_val),
                "group2": str(group2_val),
                "n1": n1,
                "n2": n2,
                "mean1": float(mean1),
                "mean2": float(mean2),
                "std1": float(std1),
                "std2": float(std2),
                "t_statistic": float(t_stat),
                "df": float(df_val),
                "p_value": float(p_value),
                "alternative": alternative,
                "significant_at_0.05": p_value < 0.05,
            }

    # Perform test(s)
    if group_by is None:
        # Single test on entire dataset
        result = perform_test(working_df)
        result_df = pl.DataFrame([result])
    else:
        # Group by and perform tests
        if isinstance(group_by, str):
            group_by = [group_by]

        # Validate group columns exist
        for col in group_by:
            if col not in working_df.columns:
                raise ValueError(f"Group column '{col}' not found in DataFrame")

        groups = working_df.group_by(group_by, maintain_order=True)
        results = []

        for group_values, group_df in groups:
            try:
                result = perform_test(group_df)
                # Add group values
                if len(group_by) == 1:
                    result[group_by[0]] = group_values
                else:
                    for i, col in enumerate(group_by):
                        result[col] = group_values[i]
                results.append(result)
            except ValueError as e:
                # Skip groups that don't have exactly 2 values in grouping mode
                if "must have exactly 2 unique values" in str(e):
                    continue
                raise

        result_df = pl.DataFrame(results)
        # Reorder columns to put group columns first
        other_cols = [c for c in result_df.columns if c not in group_by]
        result_df = result_df.select(group_by + other_cols)

    return result_df
