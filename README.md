# QuantPolars

A Python package for quantitative finance analysis using Polars, providing blazingly fast tools for data summarization and option pricing.

## Installation

```bash
pip3 install git+https://github.com/matthewgson/quantpolars.git
```

**Requirements**: Python 3.8+, Polars

## Data Summary Function (`sm`)

Generate comprehensive summary statistics for all columns in your DataFrame with a single function call. Returns a `DataSummary` object that provides both raw DataFrame access and styled output capabilities.

### Features

- **Blazingly Fast**: Single-pass computation using Polars expressions
- **Type-Aware**: Different statistics based on data type (numeric, date, categorical)
- **Missing Data**: Includes percentage of missing values for each column
- **Object-Oriented**: Returns `DataSummary` object with `.df` for raw data and `.to_gt()` for styled output
- **Styled Output**: Optional Great Tables formatting for beautiful HTML tables
- **LazyFrame Support**: Works with both eager and lazy evaluation

### Basic Usage

```python
import polars as pl
from datetime import date
from quantpolars import sm

# Create sample data
df = pl.DataFrame({
    'revenue': [1000, 2500, 1800, 3200, 2900, None, 2100, 1750],
    'profit_margin': [0.15, 0.22, 0.18, 0.25, 0.20, 0.17, 0.19, 0.16],
    'transaction_date': [
        date(2024, 1, 15), date(2024, 2, 20), date(2024, 3, 10),
        date(2024, 4, 5), date(2024, 5, 12), date(2024, 6, 8),
        date(2024, 7, 22), None
    ],
    'customer_segment': ['Enterprise', 'SMB', 'Enterprise', 'SMB', 'Enterprise', 'SMB', 'Enterprise', 'SMB'],
    'active': [True, True, False, True, False, True, True, False]
})

print("Sample Data:")
df
```

```python
# Generate summary statistics
summary = sm(df)
print("Summary Statistics with % Missing:")
summary.df  # Access the raw DataFrame
```

**Output:**
```
shape: (5, 16)
┌──────────────────┬─────────────┬──────┬─────────────┬───┬────────┬────────┬────────┬──────────┐
│ variable         ┆ type        ┆ nobs ┆ pct_missing ┆ … ┆ p75    ┆ p95    ┆ p99    ┆ n_unique │
│ ---              ┆ ---         ┆ ---  ┆ ---         ┆   ┆ ---    ┆ ---    ┆ ---    ┆ ---      │
│ str              ┆ str         ┆ i64  ┆ f64         ┆   ┆ f64    ┆ f64    ┆ f64    ┆ i64      │
╞══════════════════╪═════════════╪══════╪═════════════╪═══╪════════╪════════╪════════╪══════════╡
│ transaction_date ┆ date        ┆ 7    ┆ 12.5        ┆ … ┆ null   ┆ null   ┆ null   ┆ 7        │
│ customer_segment ┆ categorical ┆ 8    ┆ 0.0         ┆ … ┆ null   ┆ null   ┆ null   ┆ 2        │
│ active           ┆ categorical ┆ 8    ┆ 0.0         ┆ … ┆ null   ┆ null   ┆ null   ┆ 2        │
│ revenue          ┆ numeric     ┆ 7    ┆ 12.5        ┆ … ┆ 2900.0 ┆ 3200.0 ┆ 3200.0 ┆ 7        │
│ profit_margin    ┆ numeric     ┆ 8    ┆ 0.0         ┆ … ┆ 0.2    ┆ 0.25   ┆ 0.25   ┆ 8        │
└──────────────────┴─────────────┴──────┴─────────────┴───┴────────┴────────┴────────┴──────────┘
```


### Column Reference

| Column | Description |
|--------|-------------|
| `variable` | Column name |
| `type` | Data type category (`numeric`, `date`, `categorical`) |
| `nobs` | Number of non-null observations |
| `pct_missing` | Percentage of missing values |
| `mean` | Mean value (numeric columns only) |
| `sd` | Standard deviation (numeric columns only) |
| `min` | Minimum value (numeric and date columns only) |
| `max` | Maximum value (numeric and date columns only) |
| `p1-p99` | Percentiles (numeric columns only) |
| `n_unique` | Number of unique values |

### Styled Output

For beautiful formatted tables with proper date formatting:

```python
# Requires: pip3 install great-tables
styled_summary = summary.to_gt()  # Convert to styled GT table
styled_summary  # In Jupyter, displays as formatted HTML table
```

**Rendered Output Example:**
The `.to_gt()` method returns a Great Tables (GT) object that renders as a beautifully formatted HTML table in Jupyter notebooks with:

- **Table Header**: "Data Summary Statistics" with subtitle showing variable count
- **Formatted Numbers**: Statistics rounded to 2 decimal places
- **Percentage Formatting**: Missing values shown as percentages (e.g., "12.5%")
- **Date Formatting**: Min/max dates in readable format (e.g., "Jan 15, 2024")
- **Professional Styling**: Clean borders, alternating row colors, proper alignment
- **Column Labels**: User-friendly names ("Std Dev" instead of "sd", "N Obs" instead of "nobs")

**Example of what the styled table displays:**

| Variable | Type | N Obs | % Missing | Mean | Std Dev | Min | Max | 1% | 5% | 25% | 50% | 75% | 95% | 99% | N Unique |
|----------|------|-------|-----------|------|---------|-----|-----|----|----|-----|-----|-----|-----|-----|----------|
| transaction_date | date | 7 | 12.5% | — | — | Jan 15, 2024 | Jul 22, 2024 | — | — | — | — | — | — | — | 7 |
| customer_segment | categorical | 8 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | 2 |
| active | categorical | 8 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | 2 |
| revenue | numeric | 7 | 12.5% | 2,225.00 | 716.02 | 1,000.00 | 3,200.00 | 1,000.00 | 1,000.00 | 1,800.00 | 2,100.00 | 2,900.00 | 3,200.00 | 3,200.00 | 7 |
| profit_margin | numeric | 8 | 0.0% | 0.19 | 0.03 | 0.15 | 0.25 | 0.15 | 0.15 | 0.17 | 0.19 | 0.22 | 0.25 | 0.25 | 8 |



### Data Type Handling

- **Numeric**: Full statistics including percentiles
- **Date**: Min/max dates only (percentiles not supported by Polars)
- **Categorical**: Unique counts only

### Out-of-Core Example

```python
import polars as pl
import quantpolars as qp

# Batch price 1M options
df = pl.scan_csv("options_data.csv")  # Out-of-core
df = df.with_columns(
    price = qp.black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'call')['price']
)
```

## Features

- **Data Summary Tools**: Out-of-core data summarization for big data
- **Option Pricing**: Black-Scholes, Cox-Ross-Rubinstein (CRR), Barone-Adesi-Whaley (BAW) models
- **Implied Volatility**: Calculation of implied volatility
- **Greeks**: Delta, Gamma, Theta, Vega, Rho calculators

## Key Optimizations

- **Vectorized DataFrame API**: Functions operate on Polars DataFrames for batch processing of multiple options
- **Fast Norm CDF Approximation**: Implemented Abramowitz & Stegun approximation using Polars expressions
- **Lazy Evaluation**: All operations are lazy, enabling out-of-core processing for big data

## Updated API

The functions now work on Polars DataFrames, allowing for:

- **Batch Processing**: Price thousands of options in a single operation
- **Big Data Ready**: Handles datasets larger than memory with Polars' streaming
- **Extreme Speed**: Vectorized operations on columnar data

## Performance Benefits

- **No Loops**: All vectorized in Polars/Rust
- **Memory Efficient**: Columnar storage and lazy evaluation
- **Scalable**: Handles billions of rows with minimal memory
- **Parallel**: Automatic parallelization where possible
