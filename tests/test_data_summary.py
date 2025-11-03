# Test for data_summary

import pytest
import polars as pl
from datetime import datetime, date
from quantpolars.data_summary import sm, to_gt


class TestSMFunction:
    """Test suite for the sm (summary) function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with various column types for testing."""
        return pl.DataFrame(
            {
                "numeric_int": [1, 2, 3, 4, 5, None],
                "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5, None],
                "date_col": [
                    date(2023, 1, 1),
                    date(2023, 6, 15),
                    date(2023, 12, 31),
                    None,
                    date(2023, 3, 10),
                    date(2023, 9, 20),
                ],
                "datetime_col": [
                    datetime(2023, 1, 1, 10, 0),
                    datetime(2023, 6, 15, 14, 30),
                    datetime(2023, 12, 31, 23, 59),
                    None,
                    datetime(2023, 3, 10, 8, 15),
                    datetime(2023, 9, 20, 16, 45),
                ],
                "categorical_str": ["A", "B", "A", "C", "B", "A"],
                "categorical_bool": [True, False, True, False, True, False],
            }
        )

    def test_sm_with_dataframe(self, sample_dataframe):
        """Test sm function with a regular DataFrame."""
        result = sm(sample_dataframe)

        # Check that result is a DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        expected_columns = [
            "variable",
            "type",
            "nobs",
            "pct_missing",
            "mean",
            "sd",
            "min",
            "max",
            "p1",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "n_unique",
        ]
        assert result.columns == expected_columns

        # Check that all variables are present
        variables = result["variable"].to_list()
        expected_vars = [
            "date_col",
            "datetime_col",
            "categorical_str",
            "categorical_bool",
            "numeric_int",
            "numeric_float",
        ]
        assert set(variables) == set(expected_vars)

        # Check types are correctly identified
        type_mapping = dict(zip(result["variable"], result["type"]))
        assert type_mapping["numeric_int"] == "numeric"
        assert type_mapping["numeric_float"] == "numeric"
        assert type_mapping["date_col"] == "date"
        assert type_mapping["datetime_col"] == "date"
        assert type_mapping["categorical_str"] == "categorical"
        assert type_mapping["categorical_bool"] == "categorical"

    def test_sm_with_lazyframe(self, sample_dataframe):
        """Test sm function with a LazyFrame."""
        lazy_df = sample_dataframe.lazy()
        result = sm(lazy_df)

        # Check that result is a DataFrame (not LazyFrame)
        assert isinstance(result, pl.DataFrame)

        # Results should be the same as with DataFrame (check key values)
        df_result = sm(sample_dataframe)
        assert result.shape == df_result.shape
        assert result["variable"].equals(df_result["variable"])
        assert result["type"].equals(df_result["type"])
        # Check a few key values
        assert result.filter(pl.col("variable") == "numeric_int")["nobs"].equals(
            df_result.filter(pl.col("variable") == "numeric_int")["nobs"]
        )

    def test_sm_numeric_columns(self, sample_dataframe):
        """Test statistics for numeric columns."""
        result = sm(sample_dataframe)

        # Filter for numeric columns
        numeric_results = result.filter(pl.col("type") == "numeric")

        # Should have 2 numeric columns
        assert len(numeric_results) == 2

        # Check numeric_int statistics
        int_stats = numeric_results.filter(pl.col("variable") == "numeric_int").row(
            0, named=True
        )
        assert int_stats["nobs"] == 5  # excluding None
        assert (
            int_stats["pct_missing"] == 0.1667
        )  # 1 None out of 6 total rows (as decimal)
        # Do not compute n_unique for numeric columns
        assert int_stats["n_unique"] is None
        assert int_stats["mean"] == 3.0
        assert int_stats["p50"] == 3.0  # median

        # Check numeric_float statistics
        float_stats = numeric_results.filter(pl.col("variable") == "numeric_float").row(
            0, named=True
        )
        assert float_stats["nobs"] == 5
        # Do not compute n_unique for numeric columns
        assert float_stats["n_unique"] is None
        assert abs(float_stats["mean"] - 3.3) < 1e-10
        assert abs(float_stats["p50"] - 3.3) < 1e-10

    def test_sm_numeric_nan_inf_handling(self):
        """Test that NaN and Inf values are excluded from mean/std calculations."""

        # Create DataFrame with NaN, Inf, and normal values
        df_with_special = pl.DataFrame(
            {
                "mixed_numeric": [
                    1.0,
                    2.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    3.0,
                    4.0,
                    None,
                ]
            }
        )

        result = sm(df_with_special)
        numeric_stats = result.filter(pl.col("variable") == "mixed_numeric").row(
            0, named=True
        )

        # Should have 7 total observations (excluding None)
        assert numeric_stats["nobs"] == 7

        # Mean should be calculated from finite values only: [1.0, 2.0, 3.0, 4.0] = 2.5
        assert numeric_stats["mean"] == 2.5

        # Std should be calculated from finite values only
        # For [1.0, 2.0, 3.0, 4.0], std is approximately 1.290994
        expected_std = 1.2909944487358056
        assert abs(numeric_stats["sd"] - expected_std) < 1e-6

        # Min and max should be calculated from finite values only
        assert numeric_stats["min"] == 1.0
        assert numeric_stats["max"] == 4.0

        # Median (p50) should be 3.0 (Polars uses higher middle value for even count)
        assert numeric_stats["p50"] == 3.0

    def test_sm_date_columns(self, sample_dataframe):
        """Test statistics for date/datetime columns."""
        result = sm(sample_dataframe)

        # Filter for date columns
        date_results = result.filter(pl.col("type") == "date")

        # Should have 2 date columns
        assert len(date_results) == 2

        # Check date_col statistics
        date_stats = date_results.filter(pl.col("variable") == "date_col").row(
            0, named=True
        )
        assert date_stats["nobs"] == 5
        assert (
            date_stats["pct_missing"] == 0.1667
        )  # 1 None out of 6 total rows (as decimal)
        assert date_stats["n_unique"] == 5  # 5 unique dates, null not counted
        # Numeric stats should be None for date columns
        assert date_stats["mean"] is None
        assert date_stats["sd"] is None
        # All quantiles should be None for date columns
        assert date_stats["p1"] is None
        assert date_stats["p5"] is None
        assert date_stats["p25"] is None
        assert date_stats["p50"] is None
        assert date_stats["p75"] is None
        assert date_stats["p95"] is None
        assert date_stats["p99"] is None
        # But min and max should not be None
        assert date_stats["min"] is not None
        assert date_stats["max"] is not None

        # Check datetime_col statistics
        datetime_stats = date_results.filter(pl.col("variable") == "datetime_col").row(
            0, named=True
        )
        assert datetime_stats["nobs"] == 5
        # Do not compute n_unique for datetime columns
        assert datetime_stats["n_unique"] is None
        assert datetime_stats["mean"] is None
        assert datetime_stats["sd"] is None
        # All quantiles should be None for datetime columns too
        assert datetime_stats["p1"] is None
        assert datetime_stats["p5"] is None
        assert datetime_stats["p25"] is None
        assert datetime_stats["p50"] is None
        assert datetime_stats["p75"] is None
        assert datetime_stats["p95"] is None
        assert datetime_stats["p99"] is None
        # But min and max should not be None
        assert datetime_stats["min"] is not None
        assert datetime_stats["max"] is not None

    def test_sm_categorical_columns(self, sample_dataframe):
        """Test statistics for categorical columns."""
        result = sm(sample_dataframe)

        # Filter for categorical columns
        cat_results = result.filter(pl.col("type") == "categorical")

        # Should have 2 categorical columns
        assert len(cat_results) == 2

        # Check categorical_str statistics
        str_stats = cat_results.filter(pl.col("variable") == "categorical_str").row(
            0, named=True
        )
        assert str_stats["nobs"] == 6
        assert str_stats["pct_missing"] == 0.0  # no None values
        assert str_stats["n_unique"] == 3  # A, B, C
        assert str_stats["mean"] is None
        assert str_stats["sd"] is None

        # Check categorical_bool statistics
        bool_stats = cat_results.filter(pl.col("variable") == "categorical_bool").row(
            0, named=True
        )
        assert bool_stats["nobs"] == 6
        assert bool_stats["pct_missing"] == 0.0  # no None values
        assert bool_stats["n_unique"] == 2  # True, False
        assert bool_stats["mean"] is None

    def test_sm_sorting(self, sample_dataframe):
        """Test that results are sorted by type: date, categorical, numeric."""
        result = sm(sample_dataframe)

        types_order = result["type"].to_list()
        # Should be sorted: date, categorical, numeric
        expected_order = [
            "date",
            "date",
            "categorical",
            "categorical",
            "numeric",
            "numeric",
        ]
        assert types_order == expected_order

    def test_sm_empty_dataframe(self):
        """Test sm function with an empty DataFrame."""
        empty_df = pl.DataFrame({"col1": [], "col2": []}).cast(
            {"col1": pl.Int64, "col2": pl.Utf8}
        )

        result = sm(empty_df)

        # Should return summary with 0 observations
        assert len(result) == 2
        assert all(row["nobs"] == 0 for row in result.iter_rows(named=True))

    def test_sm_single_row(self):
        """Test sm function with a single row DataFrame."""
        single_df = pl.DataFrame(
            {"num": [42], "cat": ["test"], "dt": [date(2023, 1, 1)]}
        )

        result = sm(single_df)

        assert len(result) == 3
        assert all(row["nobs"] == 1 for row in result.iter_rows(named=True))

        # Check numeric stats for single value
        num_stats = result.filter(pl.col("variable") == "num").row(0, named=True)
        assert num_stats["mean"] == 42.0
        assert num_stats["sd"] is None  # single value has undefined std
        # Do not compute n_unique for numeric columns
        assert num_stats["n_unique"] is None

    def test_sm_styled_option(self, sample_dataframe):
        """Test styled option for GT table output."""
        # Get summary DataFrame
        result = sm(sample_dataframe)
        assert isinstance(result, pl.DataFrame)

        # Test that to_gt() raises ImportError when GT not available
        # (This will be the case in test environment)
        try:
            result_gt = to_gt(result)
            # If we get here, GT is available, check it's a GT object
            assert hasattr(result_gt, "_repr_html_")  # GT objects have this method
        except ImportError as e:
            assert "great_tables" in str(e)
