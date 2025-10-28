# Test for data_summary

import pytest
import polars as pl
from datetime import datetime, date
from quantpolars.data_summary import sm, DataSummary


class TestSMFunction:
    """Test suite for the sm (summary) function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with various column types for testing."""
        return pl.DataFrame({
            "numeric_int": [1, 2, 3, 4, 5, None],
            "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5, None],
            "date_col": [date(2023, 1, 1), date(2023, 6, 15), date(2023, 12, 31), None, date(2023, 3, 10), date(2023, 9, 20)],
            "datetime_col": [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 6, 15, 14, 30),
                datetime(2023, 12, 31, 23, 59),
                None,
                datetime(2023, 3, 10, 8, 15),
                datetime(2023, 9, 20, 16, 45)
            ],
            "categorical_str": ["A", "B", "A", "C", "B", "A"],
            "categorical_bool": [True, False, True, False, True, False]
        })

    def test_sm_with_dataframe(self, sample_dataframe):
        """Test sm function with a regular DataFrame."""
        result = sm(sample_dataframe)

        # Check that result is a DataSummary object
        assert isinstance(result, DataSummary)

        # Check that it has a df attribute that is a DataFrame
        assert isinstance(result.df, pl.DataFrame)

        # Check expected columns
        expected_columns = ["variable", "type", "nobs", "pct_missing", "mean", "sd", "min", "max", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "n_unique"]
        assert result.df.columns == expected_columns

        # Check that all variables are present
        variables = result.df["variable"].to_list()
        expected_vars = ["date_col", "datetime_col", "categorical_str", "categorical_bool", "numeric_int", "numeric_float"]
        assert set(variables) == set(expected_vars)

        # Check types are correctly identified
        type_mapping = dict(zip(result.df["variable"], result.df["type"]))
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

        # Check that result is a DataSummary object
        assert isinstance(result, DataSummary)

        # Check that it has a df attribute that is a DataFrame
        assert isinstance(result.df, pl.DataFrame)

        # Results should be the same as with DataFrame (check key values)
        df_result = sm(sample_dataframe)
        assert result.df.shape == df_result.df.shape
        assert result.df["variable"].equals(df_result.df["variable"])
        assert result.df["type"].equals(df_result.df["type"])
        # Check a few key values
        assert result.df.filter(pl.col("variable") == "numeric_int")["nobs"].equals(
            df_result.df.filter(pl.col("variable") == "numeric_int")["nobs"]
        )

    def test_sm_numeric_columns(self, sample_dataframe):
        """Test statistics for numeric columns."""
        result = sm(sample_dataframe)

        # Filter for numeric columns
        numeric_results = result.df.filter(pl.col("type") == "numeric")

        # Should have 2 numeric columns
        assert len(numeric_results) == 2

        # Check numeric_int statistics
        int_stats = numeric_results.filter(pl.col("variable") == "numeric_int").row(0, named=True)
        assert int_stats["nobs"] == 5  # excluding None
        assert int_stats["pct_missing"] == 0.1667  # 1 None out of 6 total rows (as decimal)
        assert int_stats["n_unique"] == 5  # unique non-null values
        assert int_stats["mean"] == 3.0
        assert int_stats["p50"] == 3.0  # median

        # Check numeric_float statistics
        float_stats = numeric_results.filter(pl.col("variable") == "numeric_float").row(0, named=True)
        assert float_stats["nobs"] == 5
        assert float_stats["n_unique"] == 5
        assert abs(float_stats["mean"] - 3.3) < 1e-10
        assert abs(float_stats["p50"] - 3.3) < 1e-10

    def test_sm_date_columns(self, sample_dataframe):
        """Test statistics for date/datetime columns."""
        result = sm(sample_dataframe)

        # Filter for date columns
        date_results = result.df.filter(pl.col("type") == "date")

        # Should have 2 date columns
        assert len(date_results) == 2

        # Check date_col statistics
        date_stats = date_results.filter(pl.col("variable") == "date_col").row(0, named=True)
        assert date_stats["nobs"] == 5
        assert date_stats["pct_missing"] == 0.1667  # 1 None out of 6 total rows (as decimal)
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
        datetime_stats = date_results.filter(pl.col("variable") == "datetime_col").row(0, named=True)
        assert datetime_stats["nobs"] == 5
        assert datetime_stats["n_unique"] == 5
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
        cat_results = result.df.filter(pl.col("type") == "categorical")

        # Should have 2 categorical columns
        assert len(cat_results) == 2

        # Check categorical_str statistics
        str_stats = cat_results.filter(pl.col("variable") == "categorical_str").row(0, named=True)
        assert str_stats["nobs"] == 6
        assert str_stats["pct_missing"] == 0.0  # no None values
        assert str_stats["n_unique"] == 3  # A, B, C
        assert str_stats["mean"] is None
        assert str_stats["sd"] is None

        # Check categorical_bool statistics
        bool_stats = cat_results.filter(pl.col("variable") == "categorical_bool").row(0, named=True)
        assert bool_stats["nobs"] == 6
        assert bool_stats["pct_missing"] == 0.0  # no None values
        assert bool_stats["n_unique"] == 2  # True, False
        assert bool_stats["mean"] is None

    def test_sm_sorting(self, sample_dataframe):
        """Test that results are sorted by type: date, categorical, numeric."""
        result = sm(sample_dataframe)

        types_order = result.df["type"].to_list()
        # Should be sorted: date, categorical, numeric
        expected_order = ["date", "date", "categorical", "categorical", "numeric", "numeric"]
        assert types_order == expected_order

    def test_sm_empty_dataframe(self):
        """Test sm function with an empty DataFrame."""
        empty_df = pl.DataFrame({
            "col1": [],
            "col2": []
        }).cast({"col1": pl.Int64, "col2": pl.Utf8})

        result = sm(empty_df)

        # Should return summary with 0 observations
        assert len(result.df) == 2
        assert all(row["nobs"] == 0 for row in result.df.iter_rows(named=True))

    def test_sm_single_row(self):
        """Test sm function with a single row DataFrame."""
        single_df = pl.DataFrame({
            "num": [42],
            "cat": ["test"],
            "dt": [date(2023, 1, 1)]
        })

        result = sm(single_df)

        assert len(result.df) == 3
        assert all(row["nobs"] == 1 for row in result.df.iter_rows(named=True))

        # Check numeric stats for single value
        num_stats = result.df.filter(pl.col("variable") == "num").row(0, named=True)
        assert num_stats["mean"] == 42.0
        assert num_stats["sd"] is None  # single value has undefined std
        assert num_stats["n_unique"] == 1

    def test_sm_styled_option(self, sample_dataframe):
        """Test styled option for GT table output."""
        # Get DataSummary object
        result = sm(sample_dataframe)
        assert isinstance(result, DataSummary)
        
        # Test that .df returns DataFrame
        assert isinstance(result.df, pl.DataFrame)
        
        # Test that .to_gt() raises ImportError when GT not available
        # (This will be the case in test environment)
        try:
            result_gt = result.to_gt()
            # If we get here, GT is available, check it's a GT object
            assert hasattr(result_gt, '_repr_html_')  # GT objects have this method
        except ImportError as e:
            assert "great_tables" in str(e)