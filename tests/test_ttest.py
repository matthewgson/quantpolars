# Test for Welch's t-test functions

import pytest
import polars as pl
import numpy as np
from quantpolars.ttest import one_t, two_t


class TestOneSampleTTest:
    """Test suite for one-sample Welch's t-test."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pl.DataFrame(
            {
                "values": np.random.normal(loc=5, scale=2, size=100),
                "group": ["A"] * 50 + ["B"] * 50,
                "category": ["X", "Y"] * 50,
            }
        )

    def test_one_sample_basic(self, sample_data):
        """Test basic one-sample t-test."""
        result = one_t(sample_data, column="values", mu=0)

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

        # Check columns
        expected_cols = [
            "n",
            "mean",
            "std",
            "t_statistic",
            "df",
            "p_value",
            "alternative",
            "significant_at_0.05",
        ]
        assert result.columns == expected_cols

        # Check values
        row = result.row(0, named=True)
        assert row["n"] == 100
        assert row["mean"] is not None
        assert row["std"] is not None
        assert row["t_statistic"] is not None
        assert row["df"] == 99  # n - 1
        assert 0 <= row["p_value"] <= 1
        assert row["alternative"] == "two-sided"

    def test_one_sample_against_known_mean(self):
        """Test one-sample t-test with known values."""
        # Create data with known mean
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = one_t(df, column="x", mu=3.0)

        row = result.row(0, named=True)
        assert row["n"] == 5
        assert abs(row["mean"] - 3.0) < 1e-10  # Mean should be exactly 3
        assert abs(row["t_statistic"]) < 1e-10  # t-stat should be near 0
        assert row["p_value"] > 0.05  # Should not be significant

    def test_one_sample_alternative_greater(self, sample_data):
        """Test one-sample t-test with greater alternative."""
        result = one_t(
            sample_data, column="values", mu=0, alternative="greater"
        )

        row = result.row(0, named=True)
        assert row["alternative"] == "greater"
        # Since data is generated from N(5, 2), it should be significantly > 0
        assert row["p_value"] < 0.05
        assert row["significant_at_0.05"] == True

    def test_one_sample_alternative_less(self, sample_data):
        """Test one-sample t-test with less alternative."""
        result = one_t(
            sample_data, column="values", mu=10, alternative="less"
        )

        row = result.row(0, named=True)
        assert row["alternative"] == "less"
        # Since data is generated from N(5, 2), it should be significantly < 10
        assert row["p_value"] < 0.05
        assert row["significant_at_0.05"] == True

    def test_one_sample_with_nulls(self):
        """Test one-sample t-test with null values."""
        df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0, None]})

        result = one_t(df, column="x", mu=3.0)

        row = result.row(0, named=True)
        assert row["n"] == 4  # Only non-null values
        assert row["mean"] == 3.0  # (1+2+4+5)/4

    def test_one_sample_with_group_by(self, sample_data):
        """Test one-sample t-test with grouping."""
        result = one_t(
            sample_data, column="values", mu=5, group_by="group"
        )

        assert len(result) == 2  # Two groups: A and B
        assert "group" in result.columns

        # Check both groups are present
        groups = result["group"].to_list()
        # Groups might be tuples or strings depending on single/multiple group_by
        group_values = [g[0] if isinstance(g, (list, tuple)) else g for g in groups]
        assert "A" in group_values
        assert "B" in group_values

        # Check each group has correct sample size
        for row in result.iter_rows(named=True):
            assert row["n"] == 50

    def test_one_sample_with_multiple_group_by(self, sample_data):
        """Test one-sample t-test with multiple grouping variables."""
        result = one_t(
            sample_data, column="values", mu=5, group_by=["group", "category"]
        )

        assert len(result) == 4  # 2 groups × 2 categories
        assert "group" in result.columns
        assert "category" in result.columns

    def test_one_sample_insufficient_data(self):
        """Test one-sample t-test with insufficient data."""
        df = pl.DataFrame({"x": [1.0]})  # Only 1 value

        result = one_t(df, column="x", mu=0)

        row = result.row(0, named=True)
        assert row["n"] == 1
        assert row["mean"] is None
        assert row["t_statistic"] is None
        assert row["p_value"] is None

    def test_one_sample_lazy_frame(self, sample_data):
        """Test one-sample t-test with LazyFrame."""
        lazy_df = sample_data.lazy()
        result = one_t(lazy_df, column="values", mu=0)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_one_sample_invalid_column(self, sample_data):
        """Test error handling for invalid column name."""
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            one_t(sample_data, column="invalid", mu=0)

    def test_one_sample_invalid_group_column(self, sample_data):
        """Test error handling for invalid group column."""
        with pytest.raises(ValueError, match="Group column 'invalid' not found"):
            one_t(sample_data, column="values", mu=0, group_by="invalid")


class TestTwoSampleTTest:
    """Test suite for two-sample Welch's t-test."""

    @pytest.fixture
    def sample_data_two_cols(self):
        """Create sample data with two columns."""
        np.random.seed(42)
        return pl.DataFrame(
            {
                "group1": np.random.normal(loc=5, scale=2, size=100),
                "group2": np.random.normal(loc=6, scale=2, size=100),
                "category": ["A", "B"] * 50,
            }
        )

    @pytest.fixture
    def sample_data_grouping(self):
        """Create sample data for grouping mode."""
        np.random.seed(42)
        values_a = np.random.normal(loc=5, scale=2, size=50)
        values_b = np.random.normal(loc=6, scale=2, size=50)
        return pl.DataFrame(
            {
                "value": np.concatenate([values_a, values_b]),
                "treatment": ["Control"] * 50 + ["Treatment"] * 50,
                "region": ["North"] * 25 + ["South"] * 25 + ["North"] * 25 + ["South"] * 25,
            }
        )

    def test_two_sample_two_columns_basic(self, sample_data_two_cols):
        """Test basic two-sample t-test with two columns."""
        result = two_t(
            sample_data_two_cols, column1="group1", column2="group2"
        )

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

        # Check columns
        expected_cols = [
            "n1",
            "n2",
            "mean1",
            "mean2",
            "std1",
            "std2",
            "t_statistic",
            "df",
            "p_value",
            "alternative",
            "significant_at_0.05",
        ]
        assert result.columns == expected_cols

        # Check values
        row = result.row(0, named=True)
        assert row["n1"] == 100
        assert row["n2"] == 100
        assert row["alternative"] == "two-sided"
        assert 0 <= row["p_value"] <= 1

    def test_two_sample_grouping_mode_basic(self, sample_data_grouping):
        """Test basic two-sample t-test with grouping mode."""
        result = two_t(
            sample_data_grouping, column1="value", group_column="treatment"
        )

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

        # Check columns
        expected_cols = [
            "group1",
            "group2",
            "n1",
            "n2",
            "mean1",
            "mean2",
            "std1",
            "std2",
            "t_statistic",
            "df",
            "p_value",
            "alternative",
            "significant_at_0.05",
        ]
        assert result.columns == expected_cols

        # Check values
        row = result.row(0, named=True)
        assert row["group1"] == "Control"
        assert row["group2"] == "Treatment"
        assert row["n1"] == 50
        assert row["n2"] == 50

    def test_two_sample_known_difference(self):
        """Test two-sample t-test with known difference."""
        # Create two groups with same mean
        df = pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = two_t(df, column1="col1", column2="col2")

        row = result.row(0, named=True)
        assert abs(row["t_statistic"]) < 1e-10  # Should be near 0
        assert row["p_value"] > 0.05  # Should not be significant

    def test_two_sample_alternative_greater(self, sample_data_two_cols):
        """Test two-sample t-test with greater alternative."""
        # group1 mean ~ 5, group2 mean ~ 6, so group1 < group2
        result = two_t(
            sample_data_two_cols,
            column1="group2",
            column2="group1",
            alternative="greater",
        )

        row = result.row(0, named=True)
        assert row["alternative"] == "greater"
        # group2 > group1, so this should be significant
        assert row["p_value"] < 0.5

    def test_two_sample_alternative_less(self, sample_data_two_cols):
        """Test two-sample t-test with less alternative."""
        result = two_t(
            sample_data_two_cols,
            column1="group1",
            column2="group2",
            alternative="less",
        )

        row = result.row(0, named=True)
        assert row["alternative"] == "less"

    def test_two_sample_with_nulls(self):
        """Test two-sample t-test with null values."""
        df = pl.DataFrame(
            {
                "col1": [1.0, 2.0, None, 4.0, 5.0],
                "col2": [2.0, None, 4.0, 5.0, 6.0],
            }
        )

        result = two_t(df, column1="col1", column2="col2")

        row = result.row(0, named=True)
        assert row["n1"] == 4  # Excluding nulls
        assert row["n2"] == 4

    def test_two_sample_with_group_by(self, sample_data_two_cols):
        """Test two-sample t-test with grouping."""
        result = two_t(
            sample_data_two_cols,
            column1="group1",
            column2="group2",
            group_by="category",
        )

        assert len(result) == 2  # Two categories: A and B
        assert "category" in result.columns

        # Check both categories are present
        categories = result["category"].to_list()
        # Categories might be tuples or strings depending on single/multiple group_by
        category_values = [c[0] if isinstance(c, (list, tuple)) else c for c in categories]
        assert "A" in category_values
        assert "B" in category_values

    def test_two_sample_grouping_mode_with_group_by(self, sample_data_grouping):
        """Test two-sample t-test in grouping mode with group_by."""
        result = two_t(
            sample_data_grouping,
            column1="value",
            group_column="treatment",
            group_by="region",
        )

        assert len(result) == 2  # Two regions
        assert "region" in result.columns

    def test_two_sample_insufficient_data(self):
        """Test two-sample t-test with insufficient data."""
        # Create DataFrame with different length columns using None padding
        df = pl.DataFrame({"col1": [1.0, None], "col2": [2.0, 3.0]})

        result = two_t(df, column1="col1", column2="col2")

        row = result.row(0, named=True)
        assert row["n1"] == 1  # Too small
        assert row["t_statistic"] is None

    def test_two_sample_grouping_wrong_number_of_groups(self):
        """Test error when grouping column has wrong number of groups."""
        df = pl.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6],
                "group": ["A", "B", "C", "A", "B", "C"],  # 3 groups, not 2
            }
        )

        with pytest.raises(ValueError, match="must have exactly 2 unique values"):
            two_t(df, column1="value", group_column="group")

    def test_two_sample_lazy_frame(self, sample_data_two_cols):
        """Test two-sample t-test with LazyFrame."""
        lazy_df = sample_data_two_cols.lazy()
        result = two_t(lazy_df, column1="group1", column2="group2")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_two_sample_invalid_inputs(self, sample_data_two_cols):
        """Test error handling for invalid inputs."""
        # No column2 or group_column
        with pytest.raises(ValueError, match="Must specify either column2 or group_column"):
            two_t(sample_data_two_cols, column1="group1")

        # Both column2 and group_column
        with pytest.raises(
            ValueError, match="Cannot specify both column2 and group_column"
        ):
            two_t(
                sample_data_two_cols,
                column1="group1",
                column2="group2",
                group_column="category",
            )

        # Invalid column name
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            two_t(
                sample_data_two_cols, column1="invalid", column2="group2"
            )


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_ab_test_scenario(self):
        """Test A/B testing scenario."""
        np.random.seed(123)

        # Simulate A/B test data
        control = np.random.normal(loc=100, scale=15, size=500)
        treatment = np.random.normal(loc=105, scale=15, size=500)  # 5% lift

        df = pl.DataFrame(
            {
                "conversion_rate": np.concatenate([control, treatment]),
                "variant": ["Control"] * 500 + ["Treatment"] * 500,
            }
        )

        result = two_t(
            df, column1="conversion_rate", group_column="variant", alternative="greater"
        )

        row = result.row(0, named=True)
        assert row["group1"] == "Control"
        assert row["group2"] == "Treatment"
        # With 500 samples and 5% effect, should detect difference
        assert row["mean2"] > row["mean1"]

    def test_multiple_experiments(self):
        """Test analyzing multiple experiments at once."""
        np.random.seed(456)

        # Multiple experiments with different effect sizes
        data = []
        for exp in ["Exp1", "Exp2", "Exp3"]:
            control = np.random.normal(loc=50, scale=10, size=100)
            treatment = np.random.normal(loc=55, scale=10, size=100)
            for i, val in enumerate(control):
                data.append({"experiment": exp, "value": val, "group": "Control"})
            for i, val in enumerate(treatment):
                data.append({"experiment": exp, "value": val, "group": "Treatment"})

        df = pl.DataFrame(data)

        result = two_t(
            df, column1="value", group_column="group", group_by="experiment"
        )

        assert len(result) == 3
        assert "experiment" in result.columns
        # Each experiment should have results
        for row in result.iter_rows(named=True):
            assert row["n1"] == 100
            assert row["n2"] == 100
            assert row["p_value"] is not None

    def test_before_after_comparison(self):
        """Test before/after comparison scenario."""
        np.random.seed(789)

        # Same subjects before and after treatment
        before = np.random.normal(loc=70, scale=12, size=50)
        after = before + np.random.normal(loc=5, scale=8, size=50)  # Improvement

        df = pl.DataFrame({"before": before, "after": after})

        result = two_t(df, column1="after", column2="before", alternative="greater")

        row = result.row(0, named=True)
        assert row["mean1"] > row["mean2"]  # After should be higher than before

    def test_quality_control_scenario(self):
        """Test quality control scenario."""
        np.random.seed(321)

        # Test if production quality meets target
        measurements = np.random.normal(loc=100.5, scale=2, size=200)
        df = pl.DataFrame({"measurement": measurements})

        # Test if measurements significantly differ from target of 100
        result = one_t(
            df, column="measurement", mu=100.0, alternative="two-sided"
        )

        row = result.row(0, named=True)
        assert row["n"] == 200
        assert row["mean"] > 100  # Should be slightly above target
