# Black-Scholes-Merton / Black-76 pricing
import itertools
import warnings

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quantpolars import black76, black_scholes, bs_price, crr_binomial, is_call_expr


def _bsm(S, K, T, r, q, sig, is_call):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call, put)


def _grid():
    rows = [dict(S=100.0, K=float(K), T=T, r=0.03, q=0.01, sigma=s, cp=cp)
            for K, T, s, cp in itertools.product([60, 90, 100, 110, 150], [1e-4, 0.01, 0.25, 1.0, 5.0],
                                                 [0.05, 0.2, 0.6, 1.5], ["C", "P"])]
    df = pl.DataFrame(rows)
    exact = _bsm(*(df[c].to_numpy() for c in ["S", "K", "T", "r", "q", "sigma"]), df["cp"].to_numpy() == "C")
    return df.with_columns(exact=pl.Series(exact))


def test_black_scholes_call():
    df = pl.DataFrame({'S': [100.0], 'K': [100.0], 'T': [1.0], 'r': [0.05], 'sigma': [0.2]})
    assert black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'call')['price'][0] == pytest.approx(10.450583572185565, abs=1e-12)


def test_black_scholes_put():
    df = pl.DataFrame({'S': [100.0], 'K': [100.0], 'T': [1.0], 'r': [0.05], 'sigma': [0.2]})
    assert black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'put')['price'][0] == pytest.approx(5.573526022256971, abs=1e-12)


def test_grid_matches_closed_form_with_dividends():
    df = _grid()
    out = black_scholes(df, "S", "K", "T", "r", "sigma", option_type="cp", q_col="q")
    err = (out["price"] - out["exact"]).abs()
    assert err.max() < 1e-12


def test_per_row_flag_encodings_agree():
    df = _grid().with_columns(
        b=pl.col("cp") == "C",
        pm=pl.when(pl.col("cp") == "C").then(1).otherwise(-1),
        w=pl.when(pl.col("cp") == "C").then(pl.lit("call")).otherwise(pl.lit("Put")),
    )
    base = black_scholes(df, "S", "K", "T", "r", "sigma", "cp", q_col="q")["price"]
    for flag in ["b", "pm", "w"]:
        assert (black_scholes(df, "S", "K", "T", "r", "sigma", flag, q_col="q")["price"] == base).all()


def test_unknown_flag_gives_null():
    df = pl.DataFrame({"S": [100.0], "K": [100.0], "T": [1.0], "r": [0.0], "sigma": [0.2], "cp": ["x"]})
    assert black_scholes(df, "S", "K", "T", "r", "sigma", "cp")["price"][0] is None
    assert df.select(is_call_expr("cp").alias("f"))["f"][0] is None


def test_expiry_and_zero_vol_return_intrinsic():
    df = pl.DataFrame({"S": [100.0, 100.0, 100.0, 100.0], "K": [90.0, 110.0, 90.0, 110.0],
                       "T": [0.0, 0.0, 1.0, 1.0], "r": [0.05] * 4, "sigma": [0.2, 0.2, 0.0, 0.0]})
    call = black_scholes(df, "S", "K", "T", "r", "sigma", "call")["price"].to_list()
    put = black_scholes(df, "S", "K", "T", "r", "sigma", "put")["price"].to_list()
    assert call == pytest.approx([10.0, 0.0, 100 - 90 * np.exp(-0.05), 0.0])
    assert put == pytest.approx([0.0, 10.0, 0.0, 110 * np.exp(-0.05) - 100])


def test_black76_equals_bsm_with_forward():
    df = _grid().with_columns(F=pl.col("S") * ((pl.col("r") - pl.col("q")) * pl.col("T")).exp())
    a = black76(df, "F", "K", "T", "r", "sigma", "cp")["price"]
    b = black_scholes(df, "S", "K", "T", "r", "sigma", "cp", q_col="q")["price"]
    assert (a - b).abs().max() < 1e-10


def test_lazy_in_lazy_out_and_expression_api():
    df = _grid()
    assert isinstance(black_scholes(df.lazy(), "S", "K", "T", "r", "sigma", "cp"), pl.LazyFrame)
    out = df.lazy().select(p=bs_price("S", "K", "T", "r", "sigma", "q", "cp")).collect()
    assert (out["p"] - df["exact"]).abs().max() < 1e-12


def test_put_call_parity():
    df = _grid()
    c = black_scholes(df, "S", "K", "T", "r", "sigma", "call", q_col="q")["price"]
    p = black_scholes(df, "S", "K", "T", "r", "sigma", "put", q_col="q")["price"]
    lhs = c - p
    rhs = df["S"] * (-df["q"] * df["T"]).exp() - df["K"] * (-df["r"] * df["T"]).exp()
    assert (lhs - rhs).abs().max() < 1e-10


def test_crr_binomial_is_a_real_tree():
    # 0.4 returned the Black-Scholes price with a DeprecationWarning; it is now a CRR tree.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        euro = crr_binomial(100, 100, 1, 0.05, 0.2, 100, 'call', american=False)
        am_put = crr_binomial(100, 100, 1, 0.05, 0.2, 2000, 'put', american=True)
    assert euro == pytest.approx(10.450583572185565, abs=0.05)
    assert am_put == pytest.approx(6.0903, abs=2e-3)  # standard American put benchmark


def test_null_inputs_propagate():
    df = pl.DataFrame({"S": [100.0, None, 100.0], "K": [100.0, 100.0, 100.0], "T": [None, 1.0, 1.0],
                       "r": [0.05] * 3, "sigma": [0.2, 0.2, None]})
    out = black_scholes(df, "S", "K", "T", "r", "sigma", "call")["price"]
    assert out.to_list() == [None, None, None]
