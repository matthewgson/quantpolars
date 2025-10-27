# QuantPolars

A Python package for quantitative finance analysis using Polars, providing blazingly fast tools for data summarization and option pricing.

## Installation

```bash
pip3 install git+https://github.com/matthewgson/quantpolars.git
```

**Requirements**: Python 3.8+, Polars


## Data Summary Function (`sm`)

Generate comprehensive summary statistics for all columns in your DataFrame with a single function call.

### Features

- **Blazingly Fast**: Single-pass computation using Polars expressions
- **Type-Aware**: Different statistics based on data type (numeric, date, categorical)
- **Missing Data**: Includes percentage of missing values for each column
- **Styled Output**: Optional Great Tables formatting for beautiful HTML tables
- **LazyFrame Support**: Works with both eager and lazy evaluation

### Basic Usage

```python
import polars as pl
from datetime import date
from quantpolars.data_summary import sm

# Create sample data
df = pl.DataFrame({
    'sales': [100, 200, 150, 300, 250, None],
    'prices': [10.5, 15.2, 12.8, 18.9, 14.3],
    'dates': [date(2023, 1, 1), date(2023, 2, 15), date(2023, 3, 10), date(2023, 4, 5), date(2023, 5, 20)],
    'categories': ['A', 'B', 'A', 'C', 'B']
})

# Generate summary statistics
summary = sm(df)
print(summary)
```

### Output Columns

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
# Requires: pip install great-tables
styled_summary = sm(df, styled=True)
styled_summary  # In Jupyter, displays as formatted HTML table
```

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

- **No Loops**: All math is vectorized in Polars/Rust
- **Memory Efficient**: Columnar storage and lazy evaluation
- **Scalable**: Handles billions of rows with minimal memory
- **Parallel**: Automatic parallelization where possible
