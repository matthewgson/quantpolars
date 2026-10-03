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
