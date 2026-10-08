"""quantpolars against other Python option-pricing packages, same rows, same machine.

Usage:  python benchmarks/bench_packages.py [--rows-vec N] [--rows-scalar N] [--rows-slow N]
                                             [--rows-quantlib N] [--only PKG] [--name TEXT]
                                             [--repeat N] [--append] [--csv]

Tasks: Black-Scholes price, delta, and implied volatility, on synthetic options
with a zero dividend yield (mibian has none). Vectorised packages run on
``--rows-vec`` rows, packages with a scalar API on ``--rows-scalar`` rows in a
Python loop, and the pure-Python ones on ``--rows-slow`` rows. Rates are rows
per second on the stated sample. Accuracy is against the SciPy closed form for
price and delta and against the true volatility for implied vol, on rows whose
extrinsic value is resolvable in double precision.

Packages tried: quantpolars (all threads, and one thread via a subprocess with
POLARS_MAX_THREADS=1), NumPy + SciPy (closed forms; implied vol by 60-step
vectorised bisection; once on one thread and once split into chunks over a
thread pool, since NumPy releases the GIL), pyfeng (NumPy, vectorised; it has
no parallel option of its own, so it also runs split into chunks over a thread
pool), QuantLib (C++ through SWIG, one call per option, solver accuracy
1e-12; on all rows by default, once in one process, once on a thread pool to
show that its bindings keep the GIL, and once split over a process pool, the
way to use several cores with it), vollib / py_vollib (Let's Be Rational,
scalar, pure Python: its optional numba path fails to compile on Python 3.13),
financepy (numba, vectorised; its solver raises on rows it judges to be at
intrinsic value, so it runs in chunks with a row-by-row fallback and the raised
rows are counted), blackscholes (pure Python, scalar, no implied vol), mibian
(pure Python, scalar). py_vollib_vectorized does not compile under a current
numba and is skipped.
"""
import argparse, os, subprocess, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm


# QuantLib workers: top level so a process pool can pickle them.
def ql_price(otype, K, F, stdev, disc):
    import QuantLib as ql
    return np.array([ql.blackFormula(int(otype[i]), K[i], F[i], stdev[i], disc[i]) for i in range(len(K))])


def ql_delta(otype, K, F, stdev, disc, S):
    import QuantLib as ql
    return np.array([ql.BlackCalculator(ql.PlainVanillaPayoff(int(otype[i]), K[i]), F[i], stdev[i], disc[i]).delta(S[i])
                     for i in range(len(K))])


def ql_iv(otype, K, F, price, disc, T):
    import QuantLib as ql
    out = np.empty(len(K))
    for i in range(len(K)):
        try:
            out[i] = ql.blackFormulaImpliedStdDev(int(otype[i]), K[i], F[i], price[i], disc[i], 0.0,
                                                  ql.nullDouble(), 1.0e-12, 200) / np.sqrt(T[i])
        except Exception:
            out[i] = np.nan
    return out


def _call_star(job):
    fn, args = job
    return fn(*args)


def split_rows(arrays, chunks):
    bounds = np.linspace(0, len(arrays[0]), chunks + 1).astype(int)
    return [tuple(x[lo:hi] for x in arrays) for lo, hi in zip(bounds[:-1], bounds[1:])]


# Everything below runs only as a script, so process-pool workers (which
# re-import this file under another name) do not rerun the benchmark.
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-vec", type=int, default=2_000_000)
    ap.add_argument("--rows-scalar", type=int, default=50_000)
    ap.add_argument("--rows-slow", type=int, default=5_000)
    ap.add_argument("--rows-quantlib", type=int, default=0, help="rows for QuantLib (0 = all)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--name", default="quantpolars", help="label prefix for the quantpolars rows")
    ap.add_argument("--repeat", type=int, default=3, help="best of N timings for vectorised packages")
    ap.add_argument("--append", action="store_true", help="append to packages_results.csv instead of overwriting")
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

    def timed(f, repeat=None):
        """Best wall time of `repeat` calls (vectorised packages) and the last result."""
        best = float("inf")
        for _ in range(repeat or a.repeat):
            t0 = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t0)
        return out, best

    N_THREADS = os.cpu_count()
    def threaded(f, *arrays, chunks=None):
        """Split row arrays into chunks and evaluate `f` on a thread pool (NumPy releases the GIL)."""
        chunks = chunks or N_THREADS
        bounds = np.linspace(0, len(arrays[0]), chunks + 1).astype(int)
        parts = [tuple(x[lo:hi] for x in arrays) for lo, hi in zip(bounds[:-1], bounds[1:])]
        with ThreadPoolExecutor(chunks) as ex:
            return np.concatenate(list(ex.map(lambda p: f(*p), parts)))

    def want(pkg):
        return a.only is None or a.only == pkg

    # ------------------------------------------------------------ quantpolars ----
    if want("quantpolars"):
        import polars as pl, quantpolars as qp
        label = a.label or f"{a.name} ({pl.thread_pool_size()} threads)"
        df = pl.DataFrame({"S": S, "K": K, "T": T, "sigma": sigma, "cp": flag_c, "mkt": price}).with_columns(r=pl.lit(r))
        out, el = timed(lambda: qp.black_scholes(df, "S", "K", "T", "r", "sigma", "cp"))
        rec(label, "price", N, el, f"max abs err {np.max(np.abs(out['price'].to_numpy() - price)):.1e}")
        out, el = timed(lambda: qp.calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", greeks=("delta",)))
        rec(label, "delta", N, el, f"max abs err {np.max(np.abs(out['delta'].to_numpy() - delta_ref)):.1e}")
        out, el = timed(lambda: qp.implied_volatility(df, "S", "K", "T", "r", "mkt", "cp"))
        rec(label, "implied vol", N, el, iv_note(out["implied_vol"].to_numpy(), sigma))
        if a.label is None:
            env = dict(os.environ, POLARS_MAX_THREADS="1")
            res = subprocess.run([sys.executable, __file__, "--only", "quantpolars", "--label", f"{a.name} (1 thread)",
                                  "--rows-vec", str(a.rows_vec), "--repeat", str(a.repeat), "--csv"],
                                 env=env, capture_output=True, text=True)
            for line in res.stdout.strip().splitlines():
                p = line.split("\t")
                rows.append((p[0], p[1], int(p[2]), float(p[3]), float(p[4]), p[5]))
                print(f"{p[0]:28s} {p[1]:12s} {int(p[2]):>9,d} rows {float(p[3]):9.3f} s {float(p[4]) / 1e6:9.3f} M rows/s  {p[5]}")

    # ------------------------------------------------------------ numpy/scipy ----
    if want("numpy"):
        def np_price(S, K, T, sig, isc):
            v = sig * np.sqrt(T); d1 = (np.log(S / K) + r * T) / v + 0.5 * v; d2 = d1 - v
            return np.where(isc, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        def np_delta(S, K, T, sig, isc):
            d1 = (np.log(S / K) + r * T) / (sig * np.sqrt(T)) + 0.5 * sig * np.sqrt(T)
            return np.where(isc, norm.cdf(d1), norm.cdf(d1) - 1)
        def np_bisect(S, K, T, isc, price):
            lo = np.full(len(S), 0.005); hi = np.full(len(S), 6.0)
            for _ in range(60):
                mid = 0.5 * (lo + hi); too_low = np_price(S, K, T, mid, isc) < price
                lo = np.where(too_low, mid, lo); hi = np.where(too_low, hi, mid)
            return 0.5 * (lo + hi)
        for label, run in (("NumPy + SciPy (1 thread)", lambda f, *x: f(*x)),
                           (f"NumPy + SciPy ({N_THREADS} threads)", threaded)):
            p, el = timed(lambda: run(np_price, S, K, T, sigma, isc))
            rec(label, "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
            d, el = timed(lambda: run(np_delta, S, K, T, sigma, isc))
            rec(label, "delta", N, el, f"max abs err {np.max(np.abs(d - delta_ref)):.1e}")
            iv, el = timed(lambda: run(np_bisect, S, K, T, isc, price), repeat=1)
            rec(label.replace(")", ", bisection)"), "implied vol", N, el, iv_note(iv, sigma))

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
            def fp_price():
                p = np.empty(N); p[m] = bs_value(S[m], T[m], K[m], r, 0.0, sigma[m], OptionTypes.EUROPEAN_CALL.value); p[mp] = bs_value(S[mp], T[mp], K[mp], r, 0.0, sigma[mp], OptionTypes.EUROPEAN_PUT.value)
                return p
            p, el = timed(fp_price)
            rec("financepy (numba)", "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
            def fp_delta():
                d = np.empty(N); d[m] = bs_delta(S[m], T[m], K[m], r, 0.0, sigma[m], OptionTypes.EUROPEAN_CALL.value); d[mp] = bs_delta(S[mp], T[mp], K[mp], r, 0.0, sigma[mp], OptionTypes.EUROPEAN_PUT.value)
                return d
            d, el = timed(fp_delta)
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
            pf_price = lambda K, S, T, sig, cp: pf.Bsm(sigma=sig, intr=r, divr=0.0).price(K, S, T, cp=cp)
            pf_delta = lambda K, S, T, sig, cp: pf.Bsm(sigma=sig, intr=r, divr=0.0).delta(K, S, T, cp=cp)
            pf_iv = lambda price, K, S, T, cp: pf.Bsm(sigma=0.2, intr=r, divr=0.0).impvol(price, K, S, T, cp=cp)
            for label, run in (("pyfeng (NumPy, 1 thread)", lambda f, *x: f(*x)),
                               (f"pyfeng (NumPy, {N_THREADS} threads)", threaded)):
                p, el = timed(lambda: run(pf_price, K, S, T, sigma, cp))
                rec(label, "price", N, el, f"max abs err {np.max(np.abs(p - price)):.1e}")
                d, el = timed(lambda: run(pf_delta, K, S, T, sigma, cp))
                rec(label, "delta", N, el, f"max abs err {np.max(np.abs(d - delta_ref)):.1e}")
                iv, el = timed(lambda: run(pf_iv, price, K, S, T, cp))
                rec(label, "implied vol", N, el, iv_note(iv, sigma))
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
            from concurrent.futures import ProcessPoolExecutor
            nq = N if a.rows_quantlib <= 0 else min(a.rows_quantlib, N)
            F = S * np.exp(r * T); disc = np.exp(-r * T); stdev = sigma * np.sqrt(T)
            otype = np.where(isc, ql.Option.Call, ql.Option.Put).astype(np.int64)
            args = {
                "price": (ql_price, (otype[:nq], K[:nq], F[:nq], stdev[:nq], disc[:nq])),
                "delta": (ql_delta, (otype[:nq], K[:nq], F[:nq], stdev[:nq], disc[:nq], S[:nq])),
                "implied vol": (ql_iv, (otype[:nq], K[:nq], F[:nq], price[:nq], disc[:nq], T[:nq])),
            }
            def note(task, out):
                if task == "price":
                    return f"max abs err {np.max(np.abs(out - price[:nq])):.1e}"
                if task == "delta":
                    return f"max abs err {np.max(np.abs(out - delta_ref[:nq])):.1e}"
                return iv_note(out, sigma[:nq])
            # one process
            for task, (fn, arrs) in args.items():
                t0 = time.perf_counter(); out = fn(*arrs); el = time.perf_counter() - t0
                rec("QuantLib (1 process)", task, nq, el, note(task, out))
            # thread pool: shows whether the bindings release the GIL (price only)
            fn, arrs = args["price"]
            parts = split_rows(arrs, N_THREADS)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(N_THREADS) as ex:
                out = np.concatenate(list(ex.map(lambda p: fn(*p), parts)))
            el = time.perf_counter() - t0
            rec(f"QuantLib ({N_THREADS} threads)", "price", nq, el, "SWIG bindings hold the GIL")
            # process pool: the way to use every core with QuantLib from Python
            with ProcessPoolExecutor(N_THREADS) as ex:
                list(ex.map(_call_star, [(ql_price, p) for p in split_rows(args["price"][1], N_THREADS)]))  # start workers
                for task, (fn, arrs) in args.items():
                    jobs = [(fn, p) for p in split_rows(arrs, N_THREADS)]
                    t0 = time.perf_counter(); out = np.concatenate(list(ex.map(_call_star, jobs))); el = time.perf_counter() - t0
                    rec(f"QuantLib ({N_THREADS} processes)", task, nq, el, note(task, out))
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
        with open(path, "a" if a.append else "w", newline="") as fh:
            w = csv.writer(fh)
            if not a.append:
                w.writerow(["package", "task", "rows", "seconds", "rows_per_s", "note"])
            w.writerows(rows)
        print("wrote", path)
