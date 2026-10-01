# Greeks vs closed forms and finite differences
import itertools

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quantpolars import ALL_GREEKS, bs_greeks, calculate_greeks, calculate_vega


def _grid():
    rows = [dict(S=100.0, K=float(K), T=T, r=0.03, q=0.01, sigma=s, cp=cp)
            for K, T, s, cp in itertools.product([70, 95, 100, 105, 140], [0.02, 0.25, 1.0, 3.0],
                                                 [0.1, 0.3, 0.8], ["C", "P"])]
    return pl.DataFrame(rows)


def _price(S, K, T, r, q, sig, isc):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    c = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    p = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(isc, c, p)


def _delta(S, K, T, r, q, sig, isc):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return np.where(isc, np.exp(-q * T) * norm.cdf(d1), -np.exp(-q * T) * norm.cdf(-d1))


def _vega(S, K, T, r, q, sig):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


@pytest.fixture(scope="module")
def grid():
    df = _grid()
    out = calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", q_col="q", greeks=ALL_GREEKS)
    arrs = {c: df[c].to_numpy() for c in ["S", "K", "T", "r", "q", "sigma"]}
    arrs["isc"] = df["cp"].to_numpy() == "C"
    return out, arrs


def _fd(fun, arrs, var, h=1e-5):
    a = dict(arrs); b = dict(arrs)
    a[var] = arrs[var] * (1 + h); b[var] = arrs[var] * (1 - h)
    return (fun(**a) - fun(**b)) / (2 * h * arrs[var])


def test_first_order_greeks_against_finite_differences(grid):
    out, a = grid
    price = lambda S, K, T, r, q, sigma, isc: _price(S, K, T, r, q, sigma, isc)
    for name, var, sign in [("delta", "S", 1), ("rho", "r", 1), ("theta", "T", -1)]:
        fd = sign * _fd(price, a, var)
        assert np.max(np.abs(out[name].to_numpy() - fd) / np.maximum(np.abs(fd), 1e-3)) < 1e-6, name
    ref = _vega(a["S"], a["K"], a["T"], a["r"], a["q"], a["sigma"])
    assert np.max(np.abs(out["vega"].to_numpy() - ref) / np.maximum(np.abs(ref), 1e-3)) < 1e-12


def test_second_order_greeks_against_finite_differences(grid):
    out, a = grid
    delta = lambda S, K, T, r, q, sigma, isc: _delta(S, K, T, r, q, sigma, isc)
    vega = lambda S, K, T, r, q, sigma, isc: _vega(S, K, T, r, q, sigma)
    price = lambda S, K, T, r, q, sigma, isc: _price(S, K, T, r, q, sigma, isc)
    checks = {"gamma": _fd(delta, a, "S"), "vanna": _fd(delta, a, "sigma"), "vomma": _fd(vega, a, "sigma"),
              "charm": -_fd(delta, a, "T"), "dual_delta": _fd(price, a, "K")}
    for name, fd in checks.items():
        assert np.max(np.abs(out[name].to_numpy() - fd) / np.maximum(np.abs(fd), 1e-3)) < 1e-5, name


def test_delta_bounds_and_call_put_relation(grid):
    out, a = grid
    d = out["delta"].to_numpy()
    assert np.all(d[a["isc"]] >= 0) and np.all(d[a["isc"]] <= 1)
    assert np.all(d[~a["isc"]] <= 0) and np.all(d[~a["isc"]] >= -1)
    # delta_call - delta_put = exp(-qT)
    df = _grid()
    c = calculate_greeks(df, "S", "K", "T", "r", "sigma", "call", q_col="q", greeks=("delta",))["delta"]
    p = calculate_greeks(df, "S", "K", "T", "r", "sigma", "put", q_col="q", greeks=("delta",))["delta"]
    assert ((c - p) - (-df["q"] * df["T"]).exp()).abs().max() < 1e-14


def test_expiry_greeks():
    df = pl.DataFrame({"S": [100.0, 100.0], "K": [90.0, 110.0], "T": [0.0, 0.0], "r": [0.05] * 2, "sigma": [0.2] * 2})
    out = calculate_greeks(df, "S", "K", "T", "r", "sigma", "call", greeks=ALL_GREEKS)
    assert out["delta"].to_list() == [1.0, 0.0]
    for g in ALL_GREEKS[1:]:
        assert out[g].to_list() == [0.0, 0.0], g


def test_theta_per_day_and_prefix_and_lazy():
    df = _grid()
    a = calculate_greeks(df, "S", "K", "T", "r", "sigma", "cp", q_col="q", greeks=("theta",))["theta"]
    b = calculate_greeks(df.lazy(), "S", "K", "T", "r", "sigma", "cp", q_col="q", greeks=("theta",),
                         theta_per_day=True, prefix="g_")
    assert isinstance(b, pl.LazyFrame)
    assert ((b.collect()["g_theta"] * 365) - a).abs().max() < 1e-10


def test_expression_api_and_vega_compat():
    df = _grid()
    exprs = bs_greeks("S", "K", "T", "r", "sigma", "q", "cp", greeks=("vega",))
    v1 = df.select(**exprs)["vega"]
    v2 = calculate_vega(df, "S", "K", "T", "r", "sigma", q_col="q")["vega"]
    assert (v1 - v2).abs().max() == 0


def test_unknown_greek_raises():
    with pytest.raises(ValueError):
        bs_greeks("S", "K", "T", "r", "sigma", greeks=("delta", "speed"))


def test_null_sigma_gives_null_greeks_not_expiry_values():
    df = pl.DataFrame({"S": [100.0, 100.0], "K": [90.0, 90.0], "T": [0.5, 0.5], "r": [0.05] * 2, "sigma": [None, 0.2]})
    out = calculate_greeks(df, "S", "K", "T", "r", "sigma", "call", greeks=ALL_GREEKS)
    assert all(out[g][0] is None for g in ALL_GREEKS)
    assert all(out[g][1] is not None for g in ALL_GREEKS)
