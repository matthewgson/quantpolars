# Welch's T-Test Functions

This document describes the Welch's t-test functions available in the `quantpolars` package.

## Overview

The package provides two main functions for performing Welch's t-tests on Polars DataFrames:

1. **`one_t`** - Test if a sample mean differs from a hypothesized value
2. **`two_t`** - Compare means between two samples (with two different modes)

Both functions support:
- Directional and two-sided tests
- Group-by functionality for stratified analysis
- LazyFrame input
- Automatic handling of null values

## One-Sample T-Test

### Function Signature

```python
one_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column: str,
    mu: float = 0.0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    group_by: Optional[Union[str, list[str]]] = None,
) -> pl.DataFrame
```

### Parameters

- **df**: Polars DataFrame or LazyFrame containing the data
- **column**: Name of the column to test
- **mu**: Hypothesized population mean (default: 0.0)
- **alternative**: Direction of the test
  - `"two-sided"`: Test if mean ≠ mu
  - `"greater"`: Test if mean > mu
  - `"less"`: Test if mean < mu
- **group_by**: Optional column name(s) to group by before testing

### Returns

A Polars DataFrame with columns:
- `n`: Sample size
- `mean`: Sample mean
- `std`: Sample standard deviation
- `t_statistic`: Welch's t-statistic
- `df`: Degrees of freedom
- `p_value`: P-value for the test
- `alternative`: Direction of test
- `significant_at_0.05`: Boolean indicator (1.0 or 0.0)

### Examples

#### Basic usage

```python
import polars as pl
from quantpolars import one_t

df = pl.DataFrame({"values": [1.2, 2.3, 1.8, 2.1, 1.9]})

# Test if mean differs from 2.0
result = one_t(df, column="values", mu=2.0)
print(result)
```

#### Directional test

```python
# Test if mean is greater than 1.5
result = one_t(
    df, 
    column="values", 
    mu=1.5, 
    alternative="greater"
)
```

#### With grouping

```python
df = pl.DataFrame({
    "score": [75, 82, 78, 90, 88, 92],
    "class": ["A", "A", "A", "B", "B", "B"]
})

# Test each class separately
result = one_t(
    df, 
    column="score", 
    mu=80.0, 
    group_by="class"
)
```

## Two-Sample T-Test

### Function Signature

```python
two_t(
    df: Union[pl.DataFrame, pl.LazyFrame],
    column1: str,
    column2: Optional[str] = None,
    group_column: Optional[str] = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    group_by: Optional[Union[str, list[str]]] = None,
) -> pl.DataFrame
```

### Parameters

- **df**: Polars DataFrame or LazyFrame containing the data
- **column1**: Name of first column (or value column in grouping mode)
- **column2**: Name of second column (required in two-columns mode)
- **group_column**: Column defining groups (required in grouping mode, must have exactly 2 unique values)
- **alternative**: Direction of the test
  - `"two-sided"`: Test if mean1 ≠ mean2
  - `"greater"`: Test if mean1 > mean2
  - `"less"`: Test if mean1 < mean2
- **group_by**: Optional column name(s) to group by before testing

### Two Modes of Operation

#### Mode 1: Two Columns
Compare values in two different columns (e.g., before/after measurements).

```python
df = pl.DataFrame({
    "before": [100, 105, 98, 102],
    "after": [105, 110, 102, 108]
})

result = two_t(
    df, 
    column1="after", 
    column2="before",
    alternative="greater"
)
```

#### Mode 2: Grouping
Compare two groups defined by a grouping column (e.g., A/B testing).

```python
df = pl.DataFrame({
    "conversion": [0.12, 0.15, 0.11, 0.16, 0.14, 0.17],
    "variant": ["Control", "Treatment", "Control", "Treatment", "Control", "Treatment"]
})

result = two_t(
    df, 
    column1="conversion", 
    group_column="variant"
)
```

### Returns

A Polars DataFrame with columns:
- `group1`, `group2`: Group labels (only in grouping mode)
- `n1`, `n2`: Sample sizes
- `mean1`, `mean2`: Sample means
- `std1`, `std2`: Sample standard deviations
- `t_statistic`: Welch's t-statistic
- `df`: Welch-Satterthwaite degrees of freedom
- `p_value`: P-value for the test
- `alternative`: Direction of test
- `significant_at_0.05`: Boolean indicator (1.0 or 0.0)

### Examples

#### A/B Test

```python
import polars as pl
import numpy as np
from quantpolars import two_t

# Simulate A/B test data
np.random.seed(42)
df = pl.DataFrame({
    "revenue": np.concatenate([
        np.random.normal(50, 10, 500),  # Control
        np.random.normal(55, 10, 500)   # Treatment
    ]),
    "variant": ["Control"] * 500 + ["Treatment"] * 500
})

result = two_t(
    df, 
    column1="revenue", 
    group_column="variant",
    alternative="two-sided"
)

print(result)
```

#### Multiple Segments

```python
# Test across multiple segments
df = pl.DataFrame({
    "revenue": [...],
    "variant": [...],
    "segment": ["Mobile", "Desktop", ...]
})

result = two_t(
    df, 
    column1="revenue", 
    group_column="variant",
    group_by="segment"
)
```

## Use Cases

### Quality Control
Test if manufacturing measurements meet specifications:

```python
df = pl.DataFrame({
    "measurement_mm": [100.5, 100.2, 100.8, 99.9, 100.3],
    "machine": ["A", "A", "A", "A", "A"]
})

result = one_t(
    df, 
    column="measurement_mm", 
    mu=100.0,
    group_by="machine"
)
```

### Clinical Trials
Compare treatment effectiveness:

```python
result = two_t(
    clinical_data, 
    column1="recovery_time", 
    group_column="treatment_group",
    alternative="less"  # New treatment should reduce recovery time
)
```

### Marketing Analytics
Evaluate campaign performance:

```python
result = two_t(
    campaign_data, 
    column1="click_rate", 
    group_column="campaign_version",
    group_by=["region", "device_type"]
)
```

## Statistical Notes

1. **Welch's T-Test**: Does not assume equal variances between groups (unlike Student's t-test)
2. **Degrees of Freedom**: Uses Welch-Satterthwaite equation for more accurate results
3. **Null Handling**: Automatically excludes null values from calculations
4. **Minimum Sample Size**: Requires at least 2 non-null observations per group

## Performance Tips

1. Use LazyFrame for large datasets
2. Use `group_by` to perform multiple tests in a single pass
3. The functions are optimized for Polars' columnar operations

## Running the Demo

A comprehensive demo script is included:

```bash
python demo_ttest.py
```

This demonstrates all major features with realistic examples.
