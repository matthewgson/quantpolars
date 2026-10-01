"""Throughput of quantpolars on a real options tape stored as Parquet.

Usage:
    python benchmarks/bench_tbt.py PATH.parquet [--spot S] [--strike K] [--expiry T]
                                   [--rate r] [--div q] [--price P] [--flag F] [--rows N]

Column names default to the ones written by the OptionsTBT project script
(S, strike_price, tau, r, q, price, call_put_flag). Scalars are accepted for the
rate and dividend yield (e.g. ``--rate 0.0425``). Reports seconds and million
rows per second for pricing, nine Greeks and implied volatility, in memory and
through ``scan_parquet`` with the streaming engine, and the share of rows whose
implied volatility exists (price inside the no-arbitrage band).
"""
import argparse, time
import polars as pl
import quantpolars as qp

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--spot", default="S"); ap.add_argument("--strike", default="strike_price")
ap.add_argument("--expiry", default="tau"); ap.add_argument("--rate", default="0.0425")
ap.add_argument("--div", default="0.013"); ap.add_argument("--price", default="price")
ap.add_argument("--flag", default="call_put_flag"); ap.add_argument("--rows", type=int, default=0)
a = ap.parse_args()

def col_or_scalar(s):
    try:
        return float(s)
    except ValueError:
        return s

r, q = col_or_scalar(a.rate), col_or_scalar(a.div)
cols = [a.spot, a.strike, a.expiry, a.price, a.flag] + [c for c in (a.rate, a.div) if isinstance(col_or_scalar(c), str)]
lf = pl.scan_parquet(a.path).select(cols).filter(pl.col(a.spot).is_not_null() & (pl.col(a.expiry) > 0))
if a.rows:
    lf = lf.head(a.rows)
df = lf.collect()
n = df.height
print(f"{a.path}: {n:,} rows with a spot and positive time to expiry; polars {pl.__version__}, {pl.thread_pool_size()} threads")

def timed(label, fn, reps=3):
    best = float("inf"); out = None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    print(f"  {label:52s} {best:7.3f} s  {n / best / 1e6:7.2f} M rows/s")
    return out

iv = timed("implied vol, in memory (8 Halley steps)",
           lambda: qp.implied_volatility(df, a.spot, a.strike, a.expiry, r, a.price, a.flag, q_col=q))
share = iv["implied_vol"].is_not_null().mean()
print(f"  {'':52s} implied vol exists on {share * 100:.1f}% of rows")
timed("implied vol, in memory (4 Halley steps)",
      lambda: qp.implied_volatility(df, a.spot, a.strike, a.expiry, r, a.price, a.flag, q_col=q, max_iter=4))
timed("implied vol, scan_parquet -> streaming -> mean",
      lambda: qp.implied_volatility(lf, a.spot, a.strike, a.expiry, r, a.price, a.flag, q_col=q)
      .select(pl.col("implied_vol").mean()).collect(engine="streaming"))
priced = iv.filter(pl.col("implied_vol").is_not_null())
m = priced.height
t0 = time.perf_counter(); qp.black_scholes(priced, a.spot, a.strike, a.expiry, r, "implied_vol", a.flag, q_col=q); el = time.perf_counter() - t0
print(f"  {'price at the implied vol':52s} {el:7.3f} s  {m / el / 1e6:7.2f} M rows/s")
t0 = time.perf_counter(); qp.calculate_greeks(priced, a.spot, a.strike, a.expiry, r, "implied_vol", a.flag, q_col=q); el = time.perf_counter() - t0
print(f"  {'five Greeks at the implied vol':52s} {el:7.3f} s  {m / el / 1e6:7.2f} M rows/s")
t0 = time.perf_counter(); qp.calculate_greeks(priced, a.spot, a.strike, a.expiry, r, "implied_vol", a.flag, q_col=q, greeks=qp.ALL_GREEKS); el = time.perf_counter() - t0
print(f"  {'all nine Greeks at the implied vol':52s} {el:7.3f} s  {m / el / 1e6:7.2f} M rows/s")
