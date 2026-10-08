"""Throughput of quantpolars pricing / Greeks / implied vol on synthetic options.

Usage:  python benchmarks/bench_synthetic.py [n_rows]
Compares, on the same rows:
  * quantpolars (Rust expression plugins)          -- eager DataFrame and lazy streaming
  * NumPy + SciPy closed forms                     -- vectorised, single thread
  * NumPy bisection (60 steps) for implied vol     -- the usual "vectorised" solver
  * py_vollib_vectorized (numba, Let's Be Rational) if installed
and times the quantpolars-only American models (BAW, CRR) and grouped Welch t-tests.
Prints a CSV table of seconds and million rows per second.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, polars as pl
from scipy.stats import norm
import quantpolars as qp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
rng = np.random.default_rng(0)
S = np.full(N, 5900.0)
K = np.round(rng.uniform(5000, 6800, N) / 5) * 5
T = np.exp(rng.uniform(np.log(1 / (252 * 6.5)), np.log(1.0), N))       # 1 hour .. 1 year
sigma = rng.uniform(0.05, 1.0, N)
cp = np.where(rng.random(N) < 0.5, "C", "P")
r = np.full(N, 0.0425); q = np.full(N, 0.013)
df = pl.DataFrame({"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "cp": cp})
df = qp.black_scholes(df, "S", "K", "T", "r", "sigma", "cp", q_col="q", out_col="mkt")
# drop prices below a tenth of a cent (no IV in double precision for either library)
df = df.filter(pl.col("mkt") > 1e-3)
N = df.height
isc = df["cp"].to_numpy() == "C"
S, K, T, r, q, sigma, mkt = (df[c].to_numpy() for c in ["S", "K", "T", "r", "q", "sigma", "mkt"])
rows = []
def rec(name, seconds, note="", n=None):
    n = n or N
    rows.append((name, seconds, n / seconds / 1e6, note)); print(f"{name:48s} {seconds:8.3f} s   {n/seconds/1e6:7.2f} M rows/s  {note}")
def timed(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return best, out

threads = pl.thread_pool_size()
print(f"rows: {N:,}   polars threads: {threads}   polars {pl.__version__}")

# ---------------- price ----------------
def np_price():
    v = sigma * np.sqrt(T); d1 = (np.log(S / K) + (r - q) * T) / v + 0.5 * v; d2 = d1 - v
    c = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    p = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(isc, c, p)
t, ref = timed(np_price); rec("price  numpy+scipy", t)
t, out = timed(lambda: qp.black_scholes(df, "S", "K", "T", "r", "sigma", "cp", q_col="q")); rec("price  quantpolars (eager)", t, f"max|diff| {np.max(np.abs(out['price'].to_numpy()-ref)):.1e}")

# ---------------- greeks (delta gamma vega theta rho) ----------------
def np_greeks():
    v = sigma * np.sqrt(T); d1 = (np.log(S / K) + (r - q) * T) / v + 0.5 * v; d2 = d1 - v
    th = np.where(isc, 1.0, -1.0); pdf = norm.pdf(d1); dq = np.exp(-q * T); dr = np.exp(-r * T)
    delta = th * dq * norm.cdf(th * d1); gamma = dq * pdf / (S * v); vega = S * dq * pdf * np.sqrt(T)
    theta = -S * dq * pdf * sigma / (2 * np.sqrt(T)) - th * r * K * dr * norm.cdf(th * d2) + th * q * S * dq * norm.cdf(th * d1)
    rho = th * K * T * dr * norm.cdf(th * d2)
    return delta, gamma, vega, theta, rho
t, refg = timed(np_greeks); rec("greeks numpy+scipy (5)", t)
t, outg = timed(lambda: qp.calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", q_col="q")); rec("greeks quantpolars (5, eager)", t, f"max|delta diff| {np.max(np.abs(outg['delta'].to_numpy()-refg[0])):.1e}")
t, outg = timed(lambda: qp.calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", q_col="q", greeks=qp.ALL_GREEKS)); rec("greeks quantpolars (all 9, eager)", t)

# ---------------- implied vol ----------------
def np_bisect():
    lo = np.full(N, 0.005); hi = np.full(N, 6.0)
    F = S * np.exp((r - q) * T); P = mkt * np.exp(r * T)
    for _ in range(60):
        mid = 0.5 * (lo + hi); v = mid * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * v ** 2) / v; d2 = d1 - v
        model = np.where(isc, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))
        too_low = model < P; lo = np.where(too_low, mid, lo); hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)
t, ivb = timed(np_bisect, reps=1); rec("iv     numpy bisection (60 steps)", t, f"median rel err {np.median(np.abs(ivb-sigma)/sigma):.1e}")
t, ivq = timed(lambda: qp.implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q")); 
e = np.abs(ivq["implied_vol"].to_numpy() - sigma) / sigma
rec("iv     quantpolars (Let's Be Rational, eager)", t, f"median rel err {np.nanmedian(e):.1e}, p99.9 {np.nanpercentile(e, 99.9):.1e}, nulls {ivq['implied_vol'].null_count()}")
# lazy + streaming from parquet (out-of-core path)
path = "/tmp/_qp_bench.parquet"; df.write_parquet(path)
t, _ = timed(lambda: qp.implied_volatility(pl.scan_parquet(path), "S", "K", "T", "r", "mkt", "cp", q_col="q").select(pl.col("implied_vol").mean()).collect(engine="streaming"), reps=2)
rec("iv     quantpolars (scan_parquet streaming)", t)
try:
    from py_vollib_vectorized import vectorized_implied_volatility as pv_iv
    flag = np.where(isc, "c", "p")
    pv_iv(mkt[:1000], S[:1000], K[:1000], T[:1000], r[:1000], flag[:1000], q=q[:1000], model="black_scholes_merton", return_as="numpy")  # JIT warm-up
    t, ivp = timed(lambda: pv_iv(mkt, S, K, T, r, flag, q=q, model="black_scholes_merton", return_as="numpy"), reps=1)
    ivp = np.asarray(ivp, dtype=float)
    rec("iv     py_vollib_vectorized (numba LetsBeRational)", t, f"median rel err {np.nanmedian(np.abs(ivp-sigma)/sigma):.1e}")
except Exception as ex:
    print("py_vollib_vectorized not available:", type(ex).__name__, str(ex)[:80])
os.remove(path)

# ---------------- American options (quantpolars only) ----------------
am = df.head(min(N, 1_000_000))
na = am.height
t, _ = timed(lambda: am.select(qp.baw_price("S", "K", "T", "r", "sigma", "q", "cp")))
rec(f"american quantpolars BAW ({na:,} rows)", t, n=na)
t, _ = timed(lambda: am.head(100_000).select(qp.crr_price("S", "K", "T", "r", "sigma", "q", "cp", steps=200)), reps=1)
rec("american quantpolars CRR 200 steps (100,000 rows)", t, n=100_000)

# ---------------- grouped Welch t-tests ----------------
g = pl.DataFrame({"g": rng.integers(0, 5000, N), "x": rng.normal(0, 1, N), "arm": rng.integers(0, 2, N)})
t, _ = timed(lambda: qp.two_t(g, "x", group_column="arm", group_by="g"))
rec("ttest  quantpolars two_t, 5,000 groups", t)
out = pl.DataFrame(rows, schema=["method", "seconds", "million_rows_per_s", "note"], orient="row")
print(out.write_csv())
