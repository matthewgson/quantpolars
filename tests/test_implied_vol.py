# Implied volatility: round trips, invalid inputs, both frame types
import itertools

import numpy as np
import polars as pl
import pytest

from quantpolars import black76, black_scholes, implied_volatility, implied_volatility_black76


def _grid():
    hour = 1 / (252 * 6.5)
    rows = [dict(S=5900.0, K=float(K), T=T, r=0.0425, q=0.013, sigma=s, cp=cp)
            for K, T, s, cp in itertools.product([5000, 5500, 5800, 5880, 5900, 5920, 6000, 6300, 7000],
                                                 [hour, 3 * hour, 1 / 252, 5 / 252, 21 / 252, 0.5, 2.0],
                                                 [0.05, 0.1, 0.2, 0.4, 0.8, 1.5], ["C", "P"])]
    df = pl.DataFrame(rows)
    df = black_scholes(df, "S", "K", "T", "r", "sigma", "cp", q_col="q", out_col="mkt")
    # keep rows whose extrinsic value carries at least ten significant digits in the double price
    F = df["S"] * ((df["r"] - df["q"]) * df["T"]).exp()
    intrinsic = pl.when(pl.col("cp") == "C").then((F - pl.col("K")).clip(lower_bound=0)).otherwise((pl.col("K") - F).clip(lower_bound=0)) * (-pl.col("r") * pl.col("T")).exp()
    return df.filter((pl.col("mkt") - intrinsic) > 1e-6 * pl.col("mkt").clip(lower_bound=1.0))


def test_round_trip_recovers_sigma():
    df = _grid()
    out = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q", diagnostics=True)
    rel = ((out["implied_vol"] - out["sigma"]) / out["sigma"]).abs()
    assert out["implied_vol"].null_count() == 0
    assert rel.max() < 1e-9
    assert rel.median() < 1e-13
    # the solution reprices the input to machine precision
    assert out["iv_resid"].max() < 1e-11


def test_zero_dte_round_trip_with_cents():
    # one hour to expiry, prices rounded to a cent as on a tape: solver must still return the vol
    # that reprices the rounded quote exactly
    df = _grid().filter(pl.col("T") < 1 / 252).with_columns(mkt=(pl.col("mkt") * 100).round() / 100).filter(pl.col("mkt") > 0)
    out = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q", diagnostics=True)
    ok = out.filter(pl.col("implied_vol").is_not_null())
    assert ok.height > 0.8 * df.height
    assert ok["iv_resid"].max() < 1e-10
    back = black_scholes(ok, "S", "K", "T", "r", "implied_vol", "cp", q_col="q", out_col="back")
    assert (back["back"] - back["mkt"]).abs().max() < 1e-9


def test_invalid_prices_give_null():
    df = pl.DataFrame({"S": [100.0] * 5, "K": [90.0, 90.0, 110.0, 90.0, 90.0], "T": [0.5] * 4 + [0.0],
                       "r": [0.05] * 5, "mkt": [5.0, 200.0, 0.0, -1.0, 12.0], "cp": ["C"] * 5})
    out = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp")
    assert out["implied_vol"].to_list() == [None] * 5


def test_itm_and_otm_give_same_vol_by_parity():
    df = _grid()
    c = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q")
    assert (c["implied_vol"] - c["sigma"]).abs().max() < 1e-9


def test_black76_variant_and_lazy():
    df = _grid().with_columns(F=pl.col("S") * ((pl.col("r") - pl.col("q")) * pl.col("T")).exp())
    df = black76(df, "F", "K", "T", "r", "sigma", "cp", out_col="mkt76")
    lf = implied_volatility_black76(df.lazy(), "F", "K", "T", "r", "mkt76", "cp", out_col="iv76")
    assert isinstance(lf, pl.LazyFrame)
    out = lf.collect()
    assert ((out["iv76"] - out["sigma"]) / out["sigma"]).abs().max() < 1e-9


def test_temporaries_are_dropped_and_diagnostics_optional():
    df = _grid().head(10)
    out = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q")
    assert set(out.columns) == set(df.columns) | {"implied_vol"}
    out = implied_volatility(df, "S", "K", "T", "r", "mkt", "cp", q_col="q", diagnostics=True)
    assert {"iv_resid", "iv_bracket"} <= set(out.columns)


def test_legacy_positional_signature():
    df = pl.DataFrame({"S": [100.0], "K": [100.0], "T": [1.0], "r": [0.05], "mkt": [10.450583572185565]})
    assert implied_volatility(df, "S", "K", "T", "r", "mkt", "call")["implied_vol"][0] == pytest.approx(0.2, rel=1e-12)
