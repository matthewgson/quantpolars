# quantpolars 0.5.0 (Rust engine) vs other packages, 2026-10-08

`bench_packages.py` with its defaults: 1,669,576 synthetic SPX-style options (strikes 5000-6800 on
a 5900 index, one hour to one year, vols 5%-100%, calls and puts, zero dividend yield, rows whose
extrinsic value is resolvable in double precision). quantpolars 0.4.0 (pure Polars) was run from a
second environment with the same script and appended. Best of 3 timings for vectorised packages.

Machine: Apple M2 Max (8 performance + 4 efficiency cores), macOS, Python 3.13.15, Polars 2.0.0,
NumPy 2.5.3, SciPy 1.18.1, numba 0.68.0, pyfeng 0.5.0, financepy 1.0.1, QuantLib 1.43,
vollib 1.0.11, blackscholes 0.2.2, mibian 0.1.3.

Million rows per second (higher is better):

| Package | price | delta | implied vol | IV median rel err |
|---|---:|---:|---:|---:|
| **quantpolars 0.5.0, 1 thread** | **58.7** | **88.8** | **4.47** | 1.5e-15 |
| **quantpolars 0.5.0, 12 threads** | **282.1** | **336.4** | **38.3** | 1.5e-15 |
| quantpolars 0.4.0, 1 thread | 3.22 | 4.67 | 0.189 | 1.8e-15 |
| quantpolars 0.4.0, 12 threads | 22.3 | 31.7 | 1.21 | 1.8e-15 |
| pyfeng (NumPy), 1 thread | 23.9 | 38.8 | 3.24 | 1.6e-15 |
| pyfeng (NumPy), 12 threads (chunked) | 133.0 | 175.8 | 7.25 | 1.6e-15 |
| NumPy + SciPy, 1 thread (IV: 60-step bisection) | 11.0 | 23.0 | 0.186 | 1.7e-15 |
| NumPy + SciPy, 12 threads (chunked) | 66.4 | 116.6 | 1.14 | 1.7e-15 |
| QuantLib, 1 process | 2.25 | 0.34 | 0.52 | 3.6e-13 |
| QuantLib, 12 threads | 2.25 | | | |
| QuantLib, 12 processes | 14.1 | 2.79 | 4.19 | 3.6e-13 |
| financepy (numba) | 9.91 | 11.0 | 0.028 | 1.2e-6 |
| vollib (Let's Be Rational, pure Python, 50k rows) | 0.37 | 0.52 | 0.065 | 1.6e-15 |
| blackscholes (pure Python, 5k rows) | 0.41 | 0.69 | | |
| mibian (pure Python, 5k rows) | 0.003 | | 0.0004 | 2.1e-9 |

Accuracy against the SciPy closed form: quantpolars 0.5.0 price max abs err 2.7e-12, delta 2.8e-16
(pyfeng 4.1e-12 / 5.3e-14, QuantLib 3.7e-12 / 7.2e-14, financepy 8.4e-4 / 7.5e-8).

Notes:

* **pyfeng** has no parallel option of its own. It is NumPy/SciPy underneath, which release the GIL,
  so it also ran split into 12 chunks on a thread pool: 5.6x on price, 4.5x on delta, 2.2x on implied
  vol (its solver loop holds the GIL between NumPy calls). quantpolars stays 2.1x ahead on price,
  1.9x on delta and 5.3x on implied vol.
* **QuantLib** is C++, but from Python every option is one call through SWIG, so it is
  per-call-overhead bound, and its bindings keep the GIL: 12 threads are no faster than one.
  A 12-process pool is the way to use every core (6.3x on price, 8.2x on delta, 8.0x on IV).
  quantpolars on 12 threads is still 20x faster on price, 121x on delta and 9.2x on implied vol;
  even on one thread it beats QuantLib's 12 processes (4.2x, 32x and 1.07x).
* py_vollib_vectorized 0.1.1 fails to compile under numba 0.68 (and 0.57) and is not listed.

`bench_synthetic.py` (9,160,022 rows, 12 threads, with a 1.3% dividend yield): price 407 M rows/s,
five Greeks 260 M, all nine 198 M, implied vol 39.6 M in memory and 34.2 M through
`scan_parquet` + streaming (NumPy bisection: 0.19 M), Barone-Adesi-Whaley American 15.8 M,
200-step CRR tree 0.37 M; Welch `two_t` over 5,000 groups 29 ms.

`bench_tbt.py` (real Cboe tape) was not re-run for 0.5.0: the tape is not on this machine.

How 0.5.0 gets there: price and Greeks run in Rust in blocks of 4,096 rows, in three passes
(d1/d2, a batched normal CDF, combine). The batched CDF evaluates SciPy's central polynomial for
every value in straight-line code and gathers tail values branch-free for a single-exp erfc, with
Jäckel's full-precision CDF below x = -5.66, so the branch on the CDF region does not stall the
pipeline. Implied volatility is Jäckel's *Let's Be Rational* per row. Inputs are read as
contiguous slices, nulls are one bitmap AND, call/put flags are parsed in parallel, and the work
runs on a thread pool sized by `POLARS_MAX_THREADS`.

# Benchmark results, 2026-10-01

Machine: 128-core Linux, Python 3.13, Polars 1.42.1, NumPy 2.5, SciPy 1.18.
Scripts: `bench_synthetic.py 10000000` (9,160,022 rows after dropping prices below a tenth of a cent)
and `bench_tbt.py` on one day of the Cboe SPX options tape (835,526 executions with a spot
and positive time to expiry).

| Task | Rows | Seconds | M rows/s | Note |
|---|---|---|---|---|
| price, NumPy + SciPy | 9.16 M | 3.49 | 2.6 | one thread |
| price, quantpolars | 9.16 M | 0.16 | 57.7 | max abs diff 4e-12 |
| five Greeks, NumPy + SciPy | 9.16 M | 4.47 | 2.1 | one thread |
| five Greeks, quantpolars | 9.16 M | 0.16 | 58.0 | max delta diff 6e-14 |
| nine Greeks, quantpolars | 9.16 M | 0.18 | 52.3 | |
| implied vol, NumPy bisection, 60 steps | 9.16 M | 167.4 | 0.055 | one thread |
| implied vol, quantpolars, 8 Halley steps | 9.16 M | 2.41 | 3.8 | median rel err 1.7e-15 |
| implied vol, quantpolars, 4 Halley steps | 9.16 M | 1.29 | 7.1 | |
| implied vol, quantpolars, scan_parquet streaming | 9.16 M | 3.88 | 2.4 | includes the parquet read |
| implied vol, quantpolars, 8 steps, tape day | 0.84 M | 0.92 | 0.91 | vol exists on 98.0% of rows |
| implied vol, quantpolars, 4 steps, tape day | 0.84 M | 0.50 | 1.66 | |
| price at the implied vol, tape day | 0.82 M | 0.06 | 14.3 | |
| five Greeks, tape day | 0.82 M | 0.06 | 13.2 | |
| nine Greeks, tape day | 0.82 M | 0.07 | 12.0 | |
| implied vol, py_vollib scalar loop (Let's Be Rational) | 0.10 M | 3.95 | 0.025 | agrees with quantpolars to 1.3e-12 |

`py_vollib_vectorized` raised a numba typing error at import in this environment and is not listed.

# Cross-package benchmark, 2026-10-03

`bench_packages.py` (defaults: 2,000,000 synthetic rows, 1,669,576 after dropping prices below a
tenth of a cent and rows whose extrinsic value is not resolvable in double precision; zero dividend
yield). Versions: quantpolars 0.4.0, Polars 1.42.1, NumPy 2.2.6, SciPy 1.18.1, numba 0.61.2,
pyfeng 0.5.0, financepy 1.0.1, QuantLib 1.43, vollib 1.0.11 / py_vollib 1.0.12 /
py_lets_be_rational 1.1.2, blackscholes 0.2.2, mibian 0.1.3. py_vollib_vectorized 0.1.1 fails to
compile under numba 0.61 on Python 3.13 and is not listed.

| Package | Task | Rows | Seconds | M rows/s | Note |
|---|---|---|---|---|---|
| quantpolars (128 threads) | price | 1,669,576 | 0.140 | 11.9 | max abs err 3.9e-12 |
| quantpolars (128 threads) | delta | 1,669,576 | 0.070 | 23.9 | max abs err 4.0e-14 |
| quantpolars (128 threads) | implied vol | 1,669,576 | 1.083 | 1.54 | median rel err 1.8e-15, max 1.6e-9 |
| quantpolars (1 thread) | price | 1,669,576 | 1.040 | 1.61 | |
| quantpolars (1 thread) | delta | 1,669,576 | 0.782 | 2.14 | |
| quantpolars (1 thread) | implied vol | 1,669,576 | 16.202 | 0.103 | |
| NumPy + SciPy | price | 1,669,576 | 0.402 | 4.15 | reference |
| NumPy + SciPy | delta | 1,669,576 | 0.163 | 10.2 | reference |
| NumPy + SciPy, 60-step bisection | implied vol | 1,669,576 | 24.105 | 0.069 | median rel err 1.7e-15 |
| financepy (numba) | price | 1,669,576 | 0.251 | 6.64 | max abs err 8.4e-4 |
| financepy (numba) | delta | 1,669,576 | 0.210 | 7.95 | max abs err 7.5e-8 |
| financepy (numba) | implied vol | 1,669,576 | 112.069 | 0.015 | median rel err 1.2e-6, max 1.6e-4 |
| pyfeng (NumPy) | price | 1,669,576 | 0.152 | 11.0 | max abs err 4.0e-12 |
| pyfeng (NumPy) | delta | 1,669,576 | 0.082 | 20.3 | max abs err 5.3e-14 |
| pyfeng (NumPy) | implied vol | 1,669,576 | 1.248 | 1.34 | median rel err 1.6e-15, max 7.1e-10 |
| vollib (Let's Be Rational, scalar) | price | 50,000 | 0.366 | 0.137 | max abs err 2.6e-12 |
| vollib (Let's Be Rational, scalar) | delta | 50,000 | 0.355 | 0.141 | |
| vollib (Let's Be Rational, scalar) | implied vol | 50,000 | 1.903 | 0.026 | median rel err 1.6e-15, max 3.7e-10 |
| QuantLib (scalar) | price | 50,000 | 0.130 | 0.385 | max abs err 3.4e-12 |
| QuantLib (scalar) | delta | 50,000 | 0.397 | 0.126 | BlackCalculator |
| QuantLib (scalar) | implied vol | 50,000 | 0.208 | 0.240 | accuracy 1e-12; median rel err 3.6e-13 |
| blackscholes (pure Python) | price | 5,000 | 0.023 | 0.215 | |
| blackscholes (pure Python) | delta | 5,000 | 0.014 | 0.349 | |
| mibian (pure Python) | price | 5,000 | 5.086 | 0.001 | |
| mibian (pure Python) | implied vol | 5,000 | 38.376 | 0.00013 | median rel err 2.1e-9, max 1.1e-2 |
