"""Implied volatility (Rust kernel, Jäckel's "Let's Be Rational").

Each row is inverted with Peter Jäckel's *Let's Be Rational* (2015) algorithm
via the ``implied-vol`` Rust crate: a rational initial guess accurate to a few
ulps over most of the domain followed by at most two Householder steps, which
reaches machine precision everywhere, including deep in the wings and for
minutes-to-expiry options. There is no iteration count to tune.

Rows with no implied volatility come back as null: a price at or below the
discounted intrinsic value, at or above the discounted forward (call) or strike
(put) bound, a non-positive price, or ``T <= 0``.
"""

from __future__ import annotations

import polars as pl

from ._plugin import ExprLike, FrameLike, OptionType, call, option_flag, to_expr
from .option_pricing import bs_price


def bs_iv(
    price: ExprLike, S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call",
) -> pl.Expr:
    """Black-Scholes-Merton implied volatility expression (null where none exists)."""
    return call("bs_iv", to_expr(price), to_expr(S), to_expr(K), to_expr(T), to_expr(r), to_expr(q),
                option_flag(option_type))


def _with_iv(df, iv: pl.Expr, S, K, T, r, P, q, option_type, out_col, diagnostics):
    out = df.with_columns(iv.alias(out_col))
    if diagnostics:
        repriced = bs_price(S, K, T, r, pl.col(out_col), q, option_type)
        out = out.with_columns(
            iv_resid=((repriced - to_expr(P)) / to_expr(P)).abs(),  # relative price error at the solution
            iv_bracket=pl.lit(None, dtype=pl.Float64),  # no bracket: kept for compatibility
        )
    return out


def implied_volatility(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    market_col: ExprLike, option_type: OptionType = "call",
    q_col: ExprLike = 0.0, out_col: str = "implied_vol", max_iter: int = 8,
    diagnostics: bool = False,
) -> FrameLike:
    """Add a Black-Scholes-Merton implied volatility column.

    Parameters
    ----------
    df : DataFrame or LazyFrame (lazy in, lazy out).
    s_col, k_col, t_col, r_col, market_col : spot, strike, years to expiry,
        rate, observed option price (column names, expressions or scalars).
    option_type : ``'call'``/``'put'`` or a per-row flag column.
    q_col : dividend yield (default 0).
    out_col : output column name.
    max_iter : ignored; kept for compatibility with 0.4 (Let's Be Rational always
        converges in at most two steps).
    diagnostics : also add ``iv_resid``, the relative price error at the solution,
        and ``iv_bracket`` (always null; kept for compatibility with 0.4).
    """
    iv = bs_iv(market_col, s_col, k_col, t_col, r_col, q_col, option_type)
    return _with_iv(df, iv, s_col, k_col, t_col, r_col, market_col, q_col, option_type, out_col, diagnostics)


def implied_volatility_black76(
    df: FrameLike, f_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    market_col: ExprLike, option_type: OptionType = "call",
    out_col: str = "implied_vol", max_iter: int = 8, diagnostics: bool = False,
) -> FrameLike:
    """Black-76 implied volatility from a forward price ``F`` (see ``implied_volatility``)."""
    iv = bs_iv(market_col, f_col, k_col, t_col, r_col, r_col, option_type)
    return _with_iv(df, iv, f_col, k_col, t_col, r_col, market_col, r_col, option_type, out_col, diagnostics)
