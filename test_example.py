#!/usr/bin/env python3
"""
Test example for the quantpolars sm function
"""

import polars as pl
from datetime import date, datetime
from quantpolars.data_summary import sm


def main():
    """Demonstrate the sm function with sample data."""

    # Create sample data with different column types
    print("Creating sample DataFrame...")
    df = pl.DataFrame({
        "sales": [100, 200, 150, 300, 250, 180, None, 220],  # numeric
        "prices": [10.5, 15.2, 12.8, 18.9, 14.3, 16.7, 11.2, 13.5],  # numeric
        "dates": [date(2023, 1, 1), date(2023, 2, 15), date(2023, 3, 10),
                 date(2023, 4, 5), date(2023, 5, 20), date(2023, 6, 12),
                 date(2023, 7, 8), date(2023, 8, 14)],  # date
        "categories": ["A", "B", "A", "C", "B", "A", "C", "B"],  # categorical
        "active": [True, False, True, True, False, True, False, True]  # categorical
    })

    print("Original DataFrame:")
    print(df)
    print("\n" + "="*80 + "\n")

    # Use the sm function to get summary statistics
    print("Generating summary statistics with sm()...")
    summary = sm(df)

    print("Summary Statistics:")
    print(summary)
    print("\n" + "="*80 + "\n")

    # Show specific examples
    print("Key insights from the summary:")
    print("- Total observations per column")
    print("- Mean and standard deviation for numeric columns")
    print("- Min/Max values for numeric and date columns")
    print("- Percentiles (p1, p5, p25, p50, p75, p95, p99) for numeric columns")
    print("- Number of unique values for all columns")
    print("- Date columns show min/max dates but no percentiles")
    print("- Categorical columns show only unique counts")

    # Example with LazyFrame
    print("\n" + "="*80)
    print("Testing with LazyFrame...")
    lazy_df = df.lazy()
    lazy_summary = sm(lazy_df)
    print("LazyFrame summary (same results):")
    print(lazy_summary)

    # Example filtering and selecting specific columns
    print("\n" + "="*80)
    print("Example: Filtering summary to show only numeric columns...")
    numeric_summary = summary.filter(pl.col("type") == "numeric")
    print(numeric_summary)


if __name__ == "__main__":
    main()