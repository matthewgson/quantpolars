"""quantpolars against other Python option-pricing packages, same rows, same machine.

Usage:  python benchmarks/bench_packages.py [--rows-vec N] [--rows-scalar N] [--rows-slow N]
                                             [--only PKG] [--label TEXT] [--csv]

Tasks: Black-Scholes price, delta, and implied volatility, on synthetic options
with a zero dividend yield (mibian has none). Vectorised packages run on
``--rows-vec`` rows, packages with a scalar API on ``--rows-scalar`` rows in a
Python loop, and the pure-Python ones on ``--rows-slow`` rows. Rates are rows
per second on the stated sample. Accuracy is against the SciPy closed form for
price and delta and against the true volatility for implied vol, on rows whose
extrinsic value is resolvable in double precision.

Packages tried: quantpolars (all threads, and one thread via a subprocess),
NumPy + SciPy (closed forms; implied vol by 60-step vectorised bisection),
vollib / py_vollib (Let's Be Rational, scalar, pure Python: its optional numba
path fails to compile on Python 3.13 with numba 0.61), QuantLib (scalar, solver
accuracy 1e-12), financepy (numba, vectorised; its solver raises on rows it
judges to be at intrinsic value, so it runs in chunks with a row-by-row
fallback and the raised rows are counted), pyfeng (NumPy, vectorised),
blackscholes (pure Python, scalar, no implied vol), mibian (pure Python,
scalar). py_vollib_vectorized does not import on Python 3.13 with a current
numba and is skipped.
"""
import argparse, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--rows-vec", type=int, default=2_000_000)
ap.add_argument("--rows-scalar", type=int, default=50_000)
ap.add_argument("--rows-slow", type=int, default=5_000)
ap.add_argument("--only", default=None)
ap.add_argument("--label", default=None)
ap.add_argument("--csv", action="store_true", help="print only the CSV rows")
a = ap.parse_args()

# ---------------------------------------------------------------- data ----
rng = np.random.default_rng(0)
N = a.rows_vec
S = np.full(N, 5900.0)
K = np.round(rng.uniform(5000, 6800, N) / 5) * 5
T = np.exp(rng.uniform(np.log(1 / (252 * 6.5)), np.log(1.0), N))
sigma = rng.uniform(0.05, 1.0, N)
isc = rng.random(N) < 0.5
r = 0.0425
v = sigma * np.sqrt(T); d1 = (np.log(S / K) + r * T) / v + 0.5 * v; d2 = d1 - v
call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
price = np.where(isc, call, put)
delta_ref = np.where(isc, norm.cdf(d1), norm.cdf(d1) - 1.0)
intrinsic = np.where(isc, np.maximum(S - K * np.exp(-r * T), 0), np.maximum(K * np.exp(-r * T) - S, 0))
keep = (price > 1e-3) & ((price - intrinsic) > 1e-6 * np.maximum(price, 1.0))
S, K, T, sigma, isc, price, delta_ref = (x[keep] for x in (S, K, T, sigma, isc, price, delta_ref))
N = len(S)
flag_c = np.where(isc, "c", "p")

rows = []
def rec(pkg, task, n, seconds, note=""):
    rows.append((pkg, task, n, seconds, n / seconds, note))
    if not a.csv:
        print(f"{pkg:28s} {task:12s} {n:>9,d} rows {seconds:9.3f} s {n / seconds / 1e6:9.3f} M rows/s  {note}")

def iv_note(iv, sig):
    iv = np.asarray(iv, dtype=float); e = np.abs(iv - sig) / sig
    return f"median rel err {np.nanmedian(e):.1e}, max {np.nanmax(e):.1e}" + (f", {np.isnan(iv).sum()} failed" if np.isnan(iv).any() else "")

def want(pkg):
    return a.only is None or a.only == pkg

# ------------------------------------------------------------ quantpolars ----
if want("quantpolars"):
    import polars as pl, quantpolars as qp
    label = a.label or f"quantpolars ({pl.thread_pool_size()} threads)"
    df = pl.DataFrame({"S": S, "K": K, "T": T, "sigma": sigma, "cp": flag_c, "mkt": price}).with_columns(r=pl.lit(r))
    t0 = time.perf_counter(); out = qp.black_scholes(df, "S", "K", "T", "r", "sigma", "cp"); el = time.perf_counter() - t0
    rec(label, "price", N, el, f"max abs err {np.max(np.abs(out['price'].to_numpy() - price)):.1e}")
    t0 = time.perf_counter(); out = qp.calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", greeks=("delta",)); el = time.perf_counter() - t0
    rec(label, "delta", N, el, f"max abs err {np.max(np.abs(out['delta'].to_numpy() - delta_ref)):.1e}")
    t0 = time.perf_counter(); out = qp.implied_volatility(df, "S", "K", "T", "r", "mkt", "cp"); el = time.perf_counter() - t0
    rec(label, "implied vol", N, el, iv_note(out["implied_vol"].to_numpy(), sigma))
    if a.only is None:
        env = dict(os.environ, POLARS_MAX_THREADS="1")
        res = subprocess.run([sys.executable, __file__, "--only", "quantpolars", "--label", "quantpolars (1 thread)",
                              "--rows-vec", str(a.rows_vec), "--csv"], env=env, capture_output=True, text=True)
        for line in res.stdout.strip().splitlines():
            p = line.split("\t")
            rows.append((p[0], p[1], int(p[2]), float(p[3]), float(p[4]), p[5]))
            print(f"{p[0]:28s} {p[1]:12s} {int(p[2]):>9,d} rows {float(p[3]):9.3f} s {float(p[4]) / 1e6:9.3f} M rows/s  {p[5]}")

# ------------------------------------------------------------ numpy/scipy ----
if want("numpy"):
    def np_price(S, K, T, sig, isc):
        v = sig * np.sqrt(T); d1 = (np.log(S / K) + r * T) / v + 0.5 * v; d2 = d1 - v
        return np.where(isc, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    t0 = time.perf_counter(); p = np_price(S, K, T, sigma, isc); el = time.perf_counter() - t0
    rec("NumPy + SciPy", "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
    t0 = time.perf_counter(); d1 = (np.log(S / K) + r * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T); d = np.where(isc, norm.cdf(d1), norm.cdf(d1) - 1); el = time.perf_counter() - t0
    rec("NumPy + SciPy", "delta", N, el, f"max abs err {np.max(np.abs(d - delta_ref)):.1e}")
    t0 = time.perf_counter()
    lo = np.full(N, 0.005); hi = np.full(N, 6.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi); too_low = np_price(S, K, T, mid, isc) < price
        lo = np.where(too_low, mid, lo); hi = np.where(too_low, hi, mid)
    el = time.perf_counter() - t0
    rec("NumPy + SciPy (bisection)", "implied vol", N, el, iv_note(0.5 * (lo + hi), sigma))

# ------------------------------------------------------------- financepy ----
if want("financepy"):
    try:
        from financepy.models.black_scholes_analytic import bs_value, bs_delta, bs_implied_volatility
        from financepy.utils.global_types import OptionTypes
        m = isc; mp = ~isc
        ot = np.where(isc, OptionTypes.EUROPEAN_CALL.value, OptionTypes.EUROPEAN_PUT.value)
        bs_value(S[:10], T[:10], K[:10], r, 0.0, sigma[:10], ot[:10])  # JIT warm-up
        bs_delta(S[:10], T[:10], K[:10], r, 0.0, sigma[:10], ot[:10])
        bs_implied_volatility(S[:10], T[:10], K[:10], r, 0.0, price[:10], ot[:10])
        t0 = time.perf_counter(); p = np.empty(N); p[m] = bs_value(S[m], T[m], K[m], r, 0.0, sigma[m], OptionTypes.EUROPEAN_CALL.value); p[mp] = bs_value(S[mp], T[mp], K[mp], r, 0.0, sigma[mp], OptionTypes.EUROPEAN_PUT.value); el = time.perf_counter() - t0
        rec("financepy (numba)", "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
        t0 = time.perf_counter(); d = np.empty(N); d[m] = bs_delta(S[m], T[m], K[m], r, 0.0, sigma[m], OptionTypes.EUROPEAN_CALL.value); d[mp] = bs_delta(S[mp], T[mp], K[mp], r, 0.0, sigma[mp], OptionTypes.EUROPEAN_PUT.value); el = time.perf_counter() - t0
        rec("financepy (numba)", "delta", N, el, f"max abs err {np.max(np.abs(d - delta_ref)):.1e}")
        # the vectorised solver raises on any row it considers below intrinsic:
        # run in chunks and fall back to one row at a time inside a failing chunk
        import io, contextlib
        iv = np.full(N, np.nan); nfail = 0
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            for lo_ in range(0, N, 20000):
                sl = slice(lo_, min(lo_ + 20000, N))
                try:
                    iv[sl] = bs_implied_volatility(S[sl], T[sl], K[sl], r, 0.0, price[sl], ot[sl])
                except Exception:
                    for i in range(sl.start, sl.stop):
                        try: iv[i] = bs_implied_volatility(S[i], T[i], K[i], r, 0.0, price[i], int(ot[i]))
                        except Exception: nfail += 1
        el = time.perf_counter() - t0
        rec("financepy (numba)", "implied vol", N, el, iv_note(iv, sigma) + f" ({nfail} raised)")
    except Exception as e:
        print("financepy skipped:", type(e).__name__, str(e)[:100])

# ---------------------------------------------------------------- pyfeng ----
if want("pyfeng"):
    try:
        import pyfeng as pf
        cp = np.where(isc, 1, -1)
        t0 = time.perf_counter(); p = pf.Bsm(sigma=sigma, intr=r, divr=0.0).price(K, S, T, cp=cp); el = time.perf_counter() - t0
        rec("pyfeng (NumPy)", "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
        t0 = time.perf_counter(); d = pf.Bsm(sigma=sigma, intr=r, divr=0.0).delta(K, S, T, cp=cp); el = time.perf_counter() - t0
        rec("pyfeng (NumPy)", "delta", N, el, f"max abs err {np.max(np.abs(d - delta_ref)):.1e}")
        t0 = time.perf_counter(); iv = pf.Bsm(sigma=0.2, intr=r, divr=0.0).impvol(price, K, S, T, cp=cp); el = time.perf_counter() - t0
        rec("pyfeng (NumPy)", "implied vol", N, el, iv_note(iv, sigma))
    except Exception as e:
        print("pyfeng skipped:", type(e).__name__, str(e)[:100])

# ------------------------------------------------------- scalar packages ----
n = min(a.rows_scalar, N)
if want("vollib"):
    try:
        try:
            from vollib.black_scholes_merton import black_scholes_merton as bsm
            from vollib.black_scholes_merton.implied_volatility import implied_volatility as piv
            from vollib.black_scholes_merton.greeks.analytical import delta as pdelta
            name = "vollib (Let's Be Rational)"
        except ImportError:
            from py_vollib.black_scholes_merton import black_scholes_merton as bsm
            from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as piv
            from py_vollib.black_scholes_merton.greeks.analytical import delta as pdelta
            name = "py_vollib (Let's Be Rational)"
        bsm("c", 100.0, 100.0, 1.0, 0.05, 0.2, 0.0); piv(10.45, 100.0, 100.0, 1.0, 0.05, 0.0, "c")
        t0 = time.perf_counter(); p = np.array([bsm(flag_c[i], S[i], K[i], T[i], r, sigma[i], 0.0) for i in range(n)]); el = time.perf_counter() - t0
        rec(name, "price", n, el, f"max abs err {np.max(np.abs(p - price[:n])):.1e}")
        t0 = time.perf_counter(); d = np.array([pdelta(flag_c[i], S[i], K[i], T[i], r, sigma[i], 0.0) for i in range(n)]); el = time.perf_counter() - t0
        rec(name, "delta", n, el, f"max abs err {np.max(np.abs(d - delta_ref[:n])):.1e}")
        def safe(i):
            try: return piv(price[i], S[i], K[i], T[i], r, 0.0, flag_c[i])
            except Exception: return np.nan
        t0 = time.perf_counter(); iv = np.array([safe(i) for i in range(n)]); el = time.perf_counter() - t0
        rec(name, "implied vol", n, el, iv_note(iv, sigma[:n]))
    except Exception as e:
        print("vollib skipped:", type(e).__name__, str(e)[:100])

if want("quantlib"):
    try:
        import QuantLib as ql
        F = S * np.exp(r * T); disc = np.exp(-r * T); otype = np.where(isc, ql.Option.Call, ql.Option.Put)
        t0 = time.perf_counter(); p = np.array([ql.blackFormula(int(otype[i]), K[i], F[i], sigma[i] * np.sqrt(T[i]), disc[i]) for i in range(n)]); el = time.perf_counter() - t0
        rec("QuantLib (scalar)", "price", n, el, f"max abs err {np.max(np.abs(p - price[:n])):.1e}")
        t0 = time.perf_counter()
        d = np.array([ql.BlackCalculator(ql.PlainVanillaPayoff(int(otype[i]), K[i]), F[i], sigma[i] * np.sqrt(T[i]), disc[i]).delta(S[i]) for i in range(n)])
        el = time.perf_counter() - t0
        rec("QuantLib (scalar)", "delta", n, el, f"max abs err {np.max(np.abs(d - delta_ref[:n])):.1e}")
        def safe(i):
            try: return ql.blackFormulaImpliedStdDev(int(otype[i]), K[i], F[i], price[i], disc[i], 0.0, ql.nullDouble(), 1.0e-12, 200) / np.sqrt(T[i])
            except Exception: return np.nan
        t0 = time.perf_counter(); iv = np.array([safe(i) for i in range(n)]); el = time.perf_counter() - t0
        rec("QuantLib (scalar)", "implied vol", n, el, iv_note(iv, sigma[:n]))
    except Exception as e:
        print("QuantLib skipped:", type(e).__name__, str(e)[:100])

n = min(a.rows_slow, N)
if want("blackscholes"):
    try:
        from blackscholes import BlackScholesCall, BlackScholesPut
        def obj(i):
            cls = BlackScholesCall if isc[i] else BlackScholesPut
            return cls(S=S[i], K=K[i], T=T[i], r=r, sigma=sigma[i])
        t0 = time.perf_counter(); p = np.array([obj(i).price() for i in range(n)]); el = time.perf_counter() - t0
        rec("blackscholes (pure Python)", "price", n, el, f"max abs err {np.max(np.abs(p - price[:n])):.1e}")
        t0 = time.perf_counter(); d = np.array([obj(i).delta() for i in range(n)]); el = time.perf_counter() - t0
        rec("blackscholes (pure Python)", "delta", n, el, f"max abs err {np.max(np.abs(d - delta_ref[:n])):.1e}")
    except Exception as e:
        print("blackscholes skipped:", type(e).__name__, str(e)[:100])

if want("mibian"):
    try:
        import mibian
        t0 = time.perf_counter(); p = np.array([(mibian.BS([S[i], K[i], r * 100, T[i] * 365], volatility=sigma[i] * 100).callPrice if isc[i] else mibian.BS([S[i], K[i], r * 100, T[i] * 365], volatility=sigma[i] * 100).putPrice) for i in range(n)]); el = time.perf_counter() - t0
        rec("mibian (pure Python)", "price", n, el, f"max abs err {np.max(np.abs(p - price[:n])):.1e}")
        t0 = time.perf_counter(); iv = np.array([(mibian.BS([S[i], K[i], r * 100, T[i] * 365], callPrice=price[i]) if isc[i] else mibian.BS([S[i], K[i], r * 100, T[i] * 365], putPrice=price[i])).impliedVolatility / 100 for i in range(n)]); el = time.perf_counter() - t0
        rec("mibian (pure Python)", "implied vol", n, el, iv_note(iv, sigma[:n]))
    except Exception as e:
        print("mibian skipped:", type(e).__name__, str(e)[:100])

if a.csv:
    for p in rows:
        print("\t".join(str(x) for x in p))
else:
    import csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages_results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["package", "task", "rows", "seconds", "rows_per_s", "note"]); w.writerows(rows)
    print("wrote", path)
