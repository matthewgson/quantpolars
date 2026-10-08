# QuantPolars

Fast quantitative-finance tools for [Polars](https://pola.rs): option pricing, Greeks, implied
volatility, American options, Welch t-tests and one-call data summaries. The numerics are Rust
expression plugins, so every function is an ordinary Polars expression: it works in eager, lazy
and streaming queries and inside `group_by`, uses every core, and never holds the GIL.

## Installation

```bash
pip install git+https://github.com/matthewgson/quantpolars.git
```

Requires Python 3.9+ and **Polars 2.0+**. Installing from source compiles the Rust extension, so a
[Rust toolchain](https://rustup.rs) is needed (first build takes a few minutes).

## Options in one query

```python
import polars as pl
import quantpolars as qp

df = pl.DataFrame({
    "S": [5900.0, 5900.0], "K": [5950.0, 5850.0], "T": [1 / 252, 30 / 365],
    "r": [0.0425, 0.0425], "q": [0.013, 0.013], "cp": ["C", "P"], "mkt": [12.40, 41.10],
})

df = qp.implied_volatility(df, "S", "K", "T", "r", "mkt", option_type="cp", q_col="q")
df = qp.calculate_greeks(df, "S", "K", "T", "r", "implied_vol", option_type="cp", q_col="q",
                         greeks=("delta", "gamma", "vega", "theta"), theta_per_day=True)
df = df.with_columns(american=qp.baw_price("S", "K", "T", "r", "implied_vol", "q", "cp"))
```

Arguments can be column names, expressions or scalars. `option_type` is `"call"`/`"put"` or a
column of per-row flags (`"C"`/`"P"`, `"call"`/`"put"`, booleans, or `+1`/`-1`). LazyFrame in,
LazyFrame out.

| Frame functions | Expressions | |
|---|---|---|
| `black_scholes`, `black76` | `bs_price`, `black76_price` | European price (dividend yield `q`) |
| `calculate_greeks` | `bs_greeks`, `greeks_struct` | delta gamma vega theta rho vanna vomma charm dual_delta |
| `implied_volatility`, `implied_volatility_black76` | `bs_iv` | Jäckel's *Let's Be Rational*; null where no vol exists |
| `crr_binomial`, `baw_american_call` | `crr_price`, `baw_price` | American: CRR tree, Barone-Adesi-Whaley |
| | `norm_cdf` `norm_sf` `norm_pdf` `norm_ppf` `erf` `erfc` `erfcx` | double precision, full tail accuracy |

Units: `T` in years, rates continuously compounded, vega per 1.00 vol, theta per year (or per day),
rho per 1.00 rate. At expiry or zero vol: intrinsic value, delta = exercise indicator.

## Performance

1.67 M options, Apple M2 Max, Polars 2.0 (million rows per second; full table, accuracy and method
in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)):

| | price | delta | implied vol |
|---|---:|---:|---:|
| **quantpolars, 1 thread** | **58.7** | **88.8** | **4.47** |
| **quantpolars, 12 threads** | **282** | **336** | **38.3** |
| pyfeng (NumPy), 1 thread / 12 threads | 23.9 / 133 | 38.8 / 176 | 3.24 / 7.25 |
| NumPy + SciPy, 1 thread / 12 threads | 11.0 / 66.4 | 23.0 / 117 | 0.19 / 1.14 (bisection) |
| QuantLib, 1 process / 12 processes | 2.25 / 14.1 | 0.34 / 2.79 | 0.52 / 4.19 |

pyfeng has no parallel mode, so it was run in chunks on a thread pool; QuantLib's Python bindings
hold the GIL (threads do not help), so it was run on a process pool. Prices agree with the SciPy
closed form to 3e-12, deltas to 3e-16, implied vols to 1.5e-15 median relative error. Thread
count follows `POLARS_MAX_THREADS`.

## Data summary and t-tests

```python
qp.sm(df)                                     # per-column stats in one pass (eager or lazy)
qp.to_gt(qp.sm(df))                           # styled table (pip install great-tables)
qp.one_t(df, "ret", mu=0, group_by="firm")    # one-sample t-test per group
qp.two_t(df, "ret", group_column="treated", group_by="year")  # Welch two-sample
```

The t-tests are one vectorised aggregation plus a Rust Student-t kernel: 5,000 groups over 9 M rows
in 29 ms. See [`TTEST_DOCUMENTATION.md`](TTEST_DOCUMENTATION.md).

## Development

```bash
pip install maturin pytest scipy
maturin develop --release   # build and install into the active environment
pytest                      # Python tests
cargo test --release        # Rust tests
python benchmarks/bench_packages.py
```

## Changes in 0.5.0

* Pricing, Greeks, implied volatility and the normal-distribution functions run on a Rust engine:
  18-24x faster than 0.4.0 single-threaded and 11-32x on 12 threads; implied vols reprice every
  quote (worst relative error 6e-12) and fewer rows come back null.
* `crr_binomial` and `baw_american_call` are real CRR / Barone-Adesi-Whaley models (they were
  Black-Scholes placeholders); new `crr_price`, `baw_price`, `bs_iv`, `greeks_struct` expressions.
* `implied_volatility(max_iter=...)` is accepted but ignored; `iv_bracket` is always null.
* Welch t-tests are vectorised; SciPy is no longer a dependency. Requires Polars 2.0+.
