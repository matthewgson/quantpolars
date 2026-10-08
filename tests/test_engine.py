# Rust-engine specifics: broadcasting, nulls, flag parsing, empty input, streaming, threads
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import quantpolars as qp

CALL, PUT = 10.450583572185565, 5.573526022256971


def test_flag_encodings_and_unknowns():
    flags = ["c", " put", "CALL", "x", None, "", "P", "true", "-1", "callx"]
    df = pl.DataFrame({"cp": flags, "S": [100.0] * len(flags)})
    out = df.select(qp.bs_price("S", 100.0, 1.0, 0.05, 0.2, option_type="cp"))["S"].to_list()
    want = [CALL, PUT, CALL, None, None, None, PUT, CALL, PUT, None]
    for got, w in zip(out, want):
        assert (got is None and w is None) or got == pytest.approx(w, abs=1e-12)


def test_numeric_and_boolean_flags():
    df = pl.DataFrame({"S": [100.0] * 4, "pm": [1, -1, 0, 7], "b": [True, False, None, True]})
    pm = df.select(qp.bs_price("S", 100.0, 1.0, 0.05, 0.2, option_type="pm"))["S"].to_list()
    assert pm[0] == pytest.approx(CALL) and pm[1] == pytest.approx(PUT) and pm[2] == pytest.approx(PUT)
    assert pm[3] is None
    b = df.select(qp.bs_price("S", 100.0, 1.0, 0.05, 0.2, option_type="b"))["S"].to_list()
    assert b[2] is None and b[0] == pytest.approx(CALL)


def test_scalar_broadcast_and_null_scalar():
    df = pl.DataFrame({"S": [100.0, None, 100.0], "K": [100.0, 100.0, None]})
    out = df.select(qp.bs_price("S", "K", 1.0, 0.05, 0.2)).to_series()
    assert out.null_count() == 2 and out[0] == pytest.approx(CALL, abs=1e-12)
    all_null = df.select(qp.bs_price("S", "K", pl.lit(None, dtype=pl.Float64), 0.05, 0.2)).to_series()
    assert all_null.null_count() == 3


def test_empty_frame():
    schema = {c: pl.Float64 for c in ["S", "K", "T", "r", "sigma"]}
    df = pl.DataFrame(schema=schema)
    assert qp.black_scholes(df, "S", "K", "T", "r", "sigma").shape == (0, 6)
    assert qp.calculate_greeks(df, "S", "K", "T", "r", "sigma").shape == (0, 10)
    assert qp.implied_volatility(df, "S", "K", "T", "r", 1.0).shape == (0, 6)


def test_greeks_struct_keeps_requested_order():
    df = pl.DataFrame({"S": [100.0, None]})
    out = df.select(qp.greeks_struct("S", 100.0, 1.0, 0.05, 0.2, greeks=["rho", "delta"]).alias("g"))
    assert out.schema["g"] == pl.Struct({"rho": pl.Float64, "delta": pl.Float64})
    assert out.unnest("g")["delta"][1] is None


def test_streaming_and_group_by_agree_with_eager():
    rng = np.random.default_rng(1)
    n = 50_000
    df = pl.DataFrame({"S": rng.uniform(50, 150, n), "K": rng.uniform(50, 150, n), "T": rng.uniform(0.01, 2, n),
                       "sigma": rng.uniform(0.1, 0.8, n), "g": rng.integers(0, 7, n),
                       "cp": rng.choice(["C", "P"], n)})
    expr = qp.bs_price("S", "K", "T", 0.03, "sigma", option_type="cp").alias("p")
    eager = df.select(expr)
    streamed = df.lazy().select(expr).collect(engine="streaming")
    assert eager.equals(streamed)
    by_group = df.group_by("g").agg(expr.sum()).sort("g")
    direct = df.with_columns(expr).group_by("g").agg(pl.col("p").sum()).sort("g")
    assert (by_group["p"] - direct["p"]).abs().max() < 1e-6


def test_iv_lazy_streaming_matches_eager():
    rng = np.random.default_rng(2)
    n = 20_000
    df = pl.DataFrame({"S": rng.uniform(80, 120, n), "K": rng.uniform(80, 120, n), "T": rng.uniform(0.05, 1, n),
                       "sigma": rng.uniform(0.1, 0.6, n)})
    df = qp.black_scholes(df, "S", "K", "T", 0.02, "sigma", out_col="px")
    eager = qp.implied_volatility(df, "S", "K", "T", 0.02, "px")
    lazy = qp.implied_volatility(df.lazy(), "S", "K", "T", 0.02, "px").collect(engine="streaming")
    assert eager.equals(lazy)


def test_results_identical_on_one_and_many_threads():
    code = ("import numpy as np, polars as pl, quantpolars as qp, sys;"
            "rng = np.random.default_rng(3); n = 300_000;"
            "df = pl.DataFrame({'S': rng.uniform(50, 150, n), 'K': rng.uniform(50, 150, n),"
            " 'T': rng.uniform(0.01, 2, n), 'v': rng.uniform(0.1, 0.8, n)});"
            "out = qp.calculate_greeks(qp.black_scholes(df, 'S', 'K', 'T', 0.03, 'v'), 'S', 'K', 'T', 0.03, 'v',"
            " greeks=qp.ALL_GREEKS);"
            "out = qp.implied_volatility(out, 'S', 'K', 'T', 0.03, 'price');"
            "sys.stdout.write(out.hash_rows().sum().__str__())")
    runs = []
    for threads in ("1", str(os.cpu_count() or 4)):
        env = dict(os.environ, POLARS_MAX_THREADS=threads)
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        runs.append(res.stdout)
    assert runs[0] == runs[1]
