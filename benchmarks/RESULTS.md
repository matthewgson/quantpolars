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
