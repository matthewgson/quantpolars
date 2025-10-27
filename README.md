# QuantPolars

A Python package for quantitative finance analysis using Polars, providing blazingly fast tools for data summarization and option pricing.

## Features

- **Data Summary Tools**: Out-of-core data summarization for big data
- **Option Pricing**: Black-Scholes, Cox-Ross-Rubinstein (CRR), Barone-Adesi-Whaley (BAW) models
- **Implied Volatility**: Calculation of implied volatility
- **Greeks**: Delta, Gamma, Theta, Vega, Rho calculators

## Key Optimizations

- **Removed NumPy/SciPy**: All computations now use Polars expressions and built-in math functions
- **Vectorized DataFrame API**: Functions operate on Polars DataFrames for batch processing of multiple options
- **Fast Norm CDF Approximation**: Implemented Abramowitz & Stegun approximation using Polars expressions
- **Lazy Evaluation**: All operations are lazy, enabling out-of-core processing for big data

## Updated API

The functions now work on Polars DataFrames, allowing for:
- **Batch Processing**: Price thousands of options in a single operation
- **Big Data Ready**: Handles datasets larger than memory with Polars' streaming
- **Extreme Speed**: Vectorized operations on columnar data

## Performance Benefits

- **No Python Loops**: All math is vectorized in Polars/Rust
- **Memory Efficient**: Columnar storage and lazy evaluation
- **Scalable**: Handles billions of rows with minimal memory
- **Parallel**: Automatic parallelization where possible

## Installation

```bash
pip install .
```

## Usage

```python
import polars as pl
import quantpolars as qp

# Create a DataFrame with option data
df = pl.DataFrame({
    'S': [100.0, 105.0],
    'K': [100.0, 100.0],
    'T': [1.0, 0.5],
    'r': [0.05, 0.05],
    'sigma': [0.2, 0.25]
})

# Price options
df = qp.black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'call')

# Calculate Greeks
df = qp.calculate_greeks(df, 'S', 'K', 'T', 'r', 'sigma', 'call')

# Implied volatility
df = df.with_columns(market_price = [10.45, 8.0])  # example
df = qp.implied_volatility(df, 'S', 'K', 'T', 'r', 'market_price', 'call')

print(df)
```

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

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```