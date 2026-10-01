# QuantPolars

A Python package for quantitative finance analysis using Polars, providing blazingly fast tools for data summarization and option pricing.

## Installation

```bash
pip3 install git+https://github.com/matthewgson/quantpolars.git
```

**Requirements**: Python 3.8+, Polars

## Data Summary Function (`sm`)

Generate comprehensive summary statistics for all columns in your DataFrame with a single function call. Returns a Polars DataFrame with summary statistics that can be optionally converted to styled GT tables.

### Features

- **Blazingly Fast**: Single-pass computation using Polars expressions
- **Type-Aware**: Different statistics based on data type (numeric, date, categorical)
- **Missing Data**: Includes percentage of missing values for each column
- **Simple API**: Returns DataFrame directly, convert to GT styling when needed
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
summary  # This is now a Polars DataFrame directly
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
from quantpolars import to_gt

# Requires: pip3 install great-tables
styled_summary = to_gt(summary)  # Convert DataFrame to styled GT table
styled_summary  # In Jupyter, displays as formatted HTML table
```

**Rendered Output Example:**
The `.to_gt()` method returns a Great Tables (GT) object that renders as a beautifully formatted HTML table in Jupyter notebooks with:

- **Table Header**: "Data Summary Statistics" with subtitle showing variable count
- **Formatted Numbers**: Statistics rounded to 2 decimal places
- **Percentage Formatting**: Missing values shown as percentages (e.g., "12.5%")
- **Date Formatting**: Min/max dates formatted as MM/DD/YYYY (e.g., "1/1/2023")
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

## Option Pricing, Greeks and Implied Volatility

Everything in this section compiles to Polars expressions: no Python UDFs, no
NumPy, no SciPy at run time. The functions therefore run inside lazy and
streaming queries, parallelise across cores, and work on frames larger than
memory.

```python
import polars as pl
import quantpolars as qp

df = pl.DataFrame({
    "S": [5900.0, 5900.0], "K": [5950.0, 5850.0], "T": [1 / 252, 30 / 365],
    "r": [0.0425, 0.0425], "q": [0.013, 0.013], "cp": ["C", "P"], "mkt": [12.40, 41.10],
})

# implied volatility from observed prices (null where no vol reprices the quote)
df = qp.implied_volatility(df, "S", "K", "T", "r", "mkt", option_type="cp", q_col="q")

# Greeks at that volatility
df = qp.calculate_greeks(df, "S", "K", "T", "r", "implied_vol", option_type="cp", q_col="q",
                         greeks=("delta", "gamma", "vega", "theta", "vanna", "charm"),
                         theta_per_day=True)

# model price (returns the frame with a `price` column)
df = qp.black_scholes(df, "S", "K", "T", "r", "implied_vol", option_type="cp", q_col="q")
```

Every argument may be a column name, a Polars expression, or a scalar. The
option type is a literal `'call'`/`'put'` or a column of per-row flags
(`'C'`/`'P'`, `'call'`/`'put'`, booleans, or `+1`/`-1`). LazyFrame in, LazyFrame
out; a DataFrame is evaluated with the streaming engine, which is the fast path
(see *Performance*). Expression-level versions (`bs_price`, `black76_price`,
`bs_greeks`) return `pl.Expr` for use inside your own `select`/`with_columns`.

### Functions

| Function | What it adds |
|---|---|
| `black_scholes(df, S, K, T, r, sigma, option_type, q_col=0, out_col="price")` | Black-Scholes-Merton price with a continuous dividend yield |
| `black76(df, F, K, T, r, sigma, option_type, out_col="price")` | Black-76 price on a forward |
| `implied_volatility(df, S, K, T, r, market, option_type, q_col=0, out_col="implied_vol", max_iter=8, diagnostics=False)` | implied volatility; null outside the no-arbitrage band or at expiry |
| `implied_volatility_black76(df, F, K, T, r, market, option_type, ...)` | the same from a forward |
| `calculate_greeks(df, S, K, T, r, sigma, option_type, q_col=0, greeks=(...), theta_per_day=False, prefix="")` | any of `delta gamma vega theta rho vanna vomma charm dual_delta` |
| `bs_price`, `black76_price`, `bs_greeks` | expression-level equivalents |
| `norm_cdf`, `norm_sf`, `norm_pdf`, `norm_ppf`, `erf`, `erfc`, `erfcx` | standard-normal expressions, double precision |

Conventions: `T` in years, `r` and `q` continuously compounded, `sigma`
annualised. Vega is per unit of volatility (divide by 100 for "per vol point"),
theta per year unless `theta_per_day=True`, rho per unit of rate. At or past
expiry the price is intrinsic value, delta the exercise indicator and the other
Greeks zero; null inputs give null outputs.

### Accuracy

* `erf`, `erfc`, `erfcx` implement Cody's (1969) rational Chebyshev
  approximations; `norm_ppf` implements Wichura's AS241. All agree with SciPy
  to about 1e-15 relative error over the whole real line, including the tails
  (`norm_cdf(-37)` is 5.7e-300, not zero). The previous Abramowitz-Stegun
  approximation had a 7e-8 absolute error, which is a 5% *relative* error
  three standard deviations out, exactly where deep out-of-the-money option
  prices live.
* Prices and Greeks match the closed forms to 1e-12 absolute (prices on an
  index at 5900) and 1e-14 (delta).
* Implied volatility follows the structure of Jaeckel's *Let's Be Rational*:
  the option is mapped to its out-of-the-money twin by put-call parity and
  expressed in the normalised Black form `b(v)`, `v = sigma sqrt(T)`; the
  closed-form Stefanica-Radoicic (2017) approximation gives the start; eight
  safeguarded Halley steps on `ln b(v)` finish, with a bracket that falls back
  to bisection so the iteration cannot diverge. `b` is evaluated through
  `erfcx`, so prices as small as 1e-300 of the geometric mean of forward and
  strike are inverted at full precision. On a 1,386-point grid with 50-digit
  reference prices (strikes 4000-9000 on a 5900 index, maturities from 30
  seconds to 3 years, volatilities 3%-300%) the solver reproduces the true
  volatility to 1e-10 or better on every row where `py_vollib`'s
  Let's-Be-Rational does, and fails on the same rows, which are the ones whose
  extrinsic value is below the double-precision resolution of the price.

### Performance

Measured on a 128-core Linux box, Polars 1.42, 9.2 million synthetic options
(the `benchmarks/bench_synthetic.py` script) and on one day of the Cboe SPX
tape (871k executions, `benchmarks/bench_tbt.py`):

| Task | NumPy + SciPy (1 thread) | quantpolars | Speed-up |
|---|---|---|---|
| Black-Scholes price | 2.6 M rows/s | 58 M rows/s | 22x |
| Five Greeks | 2.1 M rows/s | 58 M rows/s | 28x |
| Implied volatility | 0.055 M rows/s (60-step bisection) | 3.8 M rows/s | 70x |
| Implied volatility, `py_vollib` scalar Let's Be Rational loop | 0.025 M rows/s | 3.8 M rows/s | 150x |

The implied-volatility rate depends on the frame size, because the streaming
engine parallelises over morsels of rows: 0.9 M rows/s on the 871k-row tape
day, 3.8 M rows/s on 9.2 M rows. Four Halley steps instead of eight double the
rate and are enough for ordinary quotes; the default of eight covers the deep
tails. On the 100k real executions checked, the volatilities agree with
`py_vollib`'s Let's Be Rational to 1.3e-12 relative. (`py_vollib_vectorized`
did not compile under the numba available here and is not in the table.)

Two things matter for speed, and both are built into the frame-level functions:

1. **Materialise shared intermediates.** `d1`, `d2`, `N(d1)`, `N(d2)` and the
   density are computed once as temporary columns and shared by every output.
   A single self-contained expression per Greek recomputes them, and under the
   streaming engine that costs far more than the arithmetic itself.
2. **Use the streaming engine.** Polars' in-memory engine parallelises across
   the expressions in a `with_columns`, not across rows, so an eight-step
   solver with three expressions per step uses three cores. The streaming
   engine splits the rows into morsels and uses every core. A DataFrame
   passed to any frame-level function is evaluated this way automatically; if
   you pass a LazyFrame, collect it with `.collect(engine="streaming")`.

### Changes in 0.4.0

* `implied_volatility` and `calculate_greeks` in 0.3.x did not run on current
  Polars (`pl.pi` does not exist, and the Newton loop referenced columns it had
  dropped); both are rewritten.
* Dividend yield (`q_col`), per-row option-type flags, expiry and zero-volatility
  handling, null propagation, Black-76 variants, four more Greeks, and
  `norm_ppf`/`erfcx` are new.
* `crr_binomial` and `baw_american_call` were placeholders that returned the
  Black-Scholes price; they still do, and now warn.
* Requires Polars 1.0 or later.
