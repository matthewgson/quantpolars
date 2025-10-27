# QuantPolars Data Summary Test Example

This example demonstrates how to use the `sm()` function from the quantpolars package to generate comprehensive summary statistics for your data.

## Quick Start

```python
import polars as pl
from datetime import date
from quantpolars.data_summary import sm

# Create your DataFrame
df = pl.DataFrame({
    "sales": [100, 200, 150, 300, 250],
    "prices": [10.5, 15.2, 12.8, 18.9, 14.3],
    "dates": [date(2023, 1, 1), date(2023, 2, 15), date(2023, 3, 10), date(2023, 4, 5), date(2023, 5, 20)],
    "categories": ["A", "B", "A", "C", "B"]
})

# Generate summary statistics
summary = sm(df)
print(summary)
```

## Output Columns

The `sm()` function returns a DataFrame with these columns:

- `variable`: Column name from your original DataFrame
- `type`: Data type category (`numeric`, `date`, or `categorical`)
- `nobs`: Number of non-null observations
- `mean`: Mean value (numeric columns only)
- `sd`: Standard deviation (numeric columns only)
- `min`: Minimum value (numeric and date columns only)
- `max`: Maximum value (numeric and date columns only)
- `p1`-`p99`: Percentiles from 1st to 99th (numeric columns only)
- `n_unique`: Number of unique values in the column

## Supported Data Types

- **Numeric**: Integers and floats - get full statistics including percentiles
- **Date**: Date columns - get min/max dates and unique counts
- **Categorical**: Strings and booleans - get unique counts only

## Usage with LazyFrames

The function also works with Polars LazyFrames:

```python
lazy_df = df.lazy()
lazy_summary = sm(lazy_df)
```

## Installation

```bash
pip install git+https://github.com/matthewgson/quantpolars.git
```

Run the full example with:
```bash
python test_example.py
```