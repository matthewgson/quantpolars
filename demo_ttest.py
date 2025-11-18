"""
Demo script for Welch's t-test functions in quantpolars

This script demonstrates various use cases for:
1. One-sample Welch's t-test (one_t)
2. Two-sample Welch's t-test (two_t)
"""

import polars as pl
import numpy as np
from quantpolars import one_t, two_t

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("WELCH'S T-TEST DEMO - QUANTPOLARS")
print("=" * 80)

# =============================================================================
# Example 1: One-Sample T-Test - Basic Usage
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: One-Sample T-Test - Testing if mean differs from 0")
print("=" * 80)

# Create sample data
df1 = pl.DataFrame(
    {"measurements": np.random.normal(loc=5.0, scale=2.0, size=100)}
)

print("\nData sample:")
print(df1.head())

# Test if mean is significantly different from 0
result1 = one_t(
    df1, column="measurements", mu=0.0, alternative="two-sided"
)

print("\nTest Result:")
print(result1)

print(f"\nInterpretation:")
print(f"- Sample mean: {result1['mean'][0]:.3f}")
print(f"- T-statistic: {result1['t_statistic'][0]:.3f}")
print(f"- P-value: {result1['p_value'][0]:.6f}")
print(f"- Significant at α=0.05: {result1['significant_at_0.05'][0]}")

# =============================================================================
# Example 2: One-Sample T-Test with Directional Hypothesis
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: One-Sample T-Test - Directional test (greater than)")
print("=" * 80)

# Test if measurements are significantly greater than 4.5
result2 = one_t(
    df1, column="measurements", mu=4.5, alternative="greater"
)

print("\nTest Result (H₁: μ > 4.5):")
print(result2)

# =============================================================================
# Example 3: One-Sample T-Test with Grouping
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: One-Sample T-Test with Grouping")
print("=" * 80)

# Create data with multiple groups
df3 = pl.DataFrame(
    {
        "score": np.concatenate(
            [
                np.random.normal(loc=75, scale=10, size=50),
                np.random.normal(loc=82, scale=10, size=50),
                np.random.normal(loc=68, scale=10, size=50),
            ]
        ),
        "class": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
    }
)

print("\nData by class:")
print(df3.group_by("class").agg(pl.col("score").mean().alias("mean_score")))

# Test if each class's mean score differs from passing grade of 70
result3 = one_t(
    df3, column="score", mu=70.0, alternative="two-sided", group_by="class"
)

print("\nTest Results (H₀: μ = 70 for each class):")
print(result3)

# =============================================================================
# Example 4: Two-Sample T-Test - Two Columns Mode
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Two-Sample T-Test - Comparing two columns")
print("=" * 80)

# Create paired data (e.g., before and after treatment)
df4 = pl.DataFrame(
    {
        "before_treatment": np.random.normal(loc=100, scale=15, size=80),
        "after_treatment": np.random.normal(loc=105, scale=15, size=80),
    }
)

print("\nData sample:")
print(df4.head())

# Compare before vs after
result4 = two_t(
    df4, column1="after_treatment", column2="before_treatment", alternative="greater"
)

print("\nTest Result (H₁: after > before):")
print(result4)

print(f"\nInterpretation:")
print(f"- Mean before: {result4['mean2'][0]:.2f}")
print(f"- Mean after: {result4['mean1'][0]:.2f}")
print(f"- Difference: {result4['mean1'][0] - result4['mean2'][0]:.2f}")
print(f"- P-value: {result4['p_value'][0]:.6f}")

# =============================================================================
# Example 5: Two-Sample T-Test - Grouping Mode
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Two-Sample T-Test - Grouping mode (A/B test)")
print("=" * 80)

# Create A/B test data
df5 = pl.DataFrame(
    {
        "conversion_rate": np.concatenate(
            [
                np.random.normal(loc=0.12, scale=0.03, size=500),  # Control
                np.random.normal(loc=0.15, scale=0.03, size=500),  # Treatment
            ]
        ),
        "variant": ["Control"] * 500 + ["Treatment"] * 500,
    }
)

print("\nData by variant:")
print(
    df5.group_by("variant").agg(
        pl.col("conversion_rate").mean().alias("mean_conversion")
    )
)

# Compare Control vs Treatment
result5 = two_t(
    df5, column1="conversion_rate", group_column="variant", alternative="two-sided"
)

print("\nTest Result (Control vs Treatment):")
print(result5)

print(f"\nInterpretation:")
print(f"- {result5['group1'][0]} mean: {result5['mean1'][0]:.4f}")
print(f"- {result5['group2'][0]} mean: {result5['mean2'][0]:.4f}")
print(f"- Lift: {(result5['mean2'][0] / result5['mean1'][0] - 1) * 100:.2f}%")
print(f"- P-value: {result5['p_value'][0]:.6f}")
print(f"- Significant: {result5['significant_at_0.05'][0]}")

# =============================================================================
# Example 6: Two-Sample T-Test with Multiple Groups
# =============================================================================
print("\n" + "=" * 80)
print("Example 6: Multiple A/B tests across different segments")
print("=" * 80)

# Create data with multiple segments
segments_data = []
for segment in ["Mobile", "Desktop", "Tablet"]:
    control = np.random.normal(loc=50, scale=10, size=200)
    treatment = np.random.normal(loc=55, scale=10, size=200)

    for val in control:
        segments_data.append(
            {"segment": segment, "revenue": val, "variant": "Control"}
        )
    for val in treatment:
        segments_data.append(
            {"segment": segment, "revenue": val, "variant": "Treatment"}
        )

df6 = pl.DataFrame(segments_data)

print("\nData summary:")
print(
    df6.group_by(["segment", "variant"]).agg(
        pl.col("revenue").mean().alias("mean_revenue")
    )
)

# Run tests for each segment
result6 = two_t(
    df6,
    column1="revenue",
    group_column="variant",
    alternative="two-sided",
    group_by="segment",
)

print("\nTest Results by segment:")
print(result6)

# =============================================================================
# Example 7: Quality Control - Testing against specification
# =============================================================================
print("\n" + "=" * 80)
print("Example 7: Quality Control - Testing against specification")
print("=" * 80)

# Production measurements should be 100mm ± tolerance
df7 = pl.DataFrame(
    {
        "measurement_mm": np.random.normal(loc=100.5, scale=2.0, size=150),
        "machine": (["Machine_A"] * 50 + ["Machine_B"] * 50 + ["Machine_C"] * 50),
    }
)

print("\nMeasurements by machine:")
print(
    df7.group_by("machine").agg(
        pl.col("measurement_mm").mean().alias("mean_measurement")
    )
)

# Test if each machine produces parts at the target of 100mm
result7 = one_t(
    df7,
    column="measurement_mm",
    mu=100.0,
    alternative="two-sided",
    group_by="machine",
)

print("\nTest Results (H₀: μ = 100mm):")
print(result7)

# =============================================================================
# Example 8: Working with LazyFrames
# =============================================================================
print("\n" + "=" * 80)
print("Example 8: Working with LazyFrames for large datasets")
print("=" * 80)

# Create a larger dataset as LazyFrame
large_df = pl.DataFrame(
    {
        "value": np.random.normal(loc=50, scale=10, size=10000),
        "group": np.random.choice(["A", "B"], size=10000),
    }
).lazy()

print("\nRunning test on LazyFrame...")

# The function automatically handles LazyFrames
result8 = two_t(
    large_df, column1="value", group_column="group", alternative="two-sided"
)

print("\nTest Result:")
print(result8)

print(f"\nSample sizes: n1={result8['n1'][0]}, n2={result8['n2'][0]}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key features demonstrated:

1. One-Sample Tests:
   - Test if population mean differs from a hypothesized value
   - Support for directional tests (greater, less, two-sided)
   - Group-by functionality for testing multiple groups at once

2. Two-Sample Tests:
   - Two modes: compare two columns OR compare two groups
   - Welch's correction for unequal variances
   - Support for A/B testing and other comparative analyses
   - Group-by functionality for stratified analyses

3. Output:
   - Returns Polars DataFrame with all relevant statistics
   - Includes t-statistic, p-value, degrees of freedom
   - Significance indicator at α=0.05
   - Sample sizes, means, and standard deviations

4. Data handling:
   - Works with both DataFrame and LazyFrame
   - Automatically handles null values
   - Validates input columns and groups
""")

print("\nDemo completed successfully!")
