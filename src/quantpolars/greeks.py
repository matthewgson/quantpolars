"""Black-Scholes-Merton Greeks (Rust kernels).

``bs_greeks`` returns a dict of named expressions; ``calculate_greeks`` adds the
Greeks as columns to a DataFrame or LazyFrame. Either way the requested Greeks
come from one pass over the rows that computes d1, d2, N(d1), N(d2) and the
density once and skips whatever no requested Greek needs. All Greeks are
analytic.

Units
-----
* ``delta``  : per unit of spot (call in [0, 1], put in [-1, 0]).
* ``gamma``  : per unit of spot squared.
* ``vega``   : per unit of volatility (i.e. per 100 vol points); divide by
               100 for the market convention of "per vol point".
* ``theta``  : per year; ``theta_per_day=True`` divides by 365.
* ``rho``    : per unit of rate; divide by 100 for "per percentage point".
* ``vanna``  : d delta / d sigma (= d vega / d spot).
* ``vomma``  : d vega / d sigma (also called volga).
* ``charm``  : minus d delta / d T, i.e. delta decay per year as time passes.
* ``dual_delta`` : d price / d strike (minus the discounted exercise probability
                   for a call).

At or past expiry (``T <= 0``) or with zero volatility ``delta`` is the
exercise indicator and every other Greek is 0. Null inputs give null Greeks.
"""

from __future__ import annotations

from typing import Dict, Iterable

import polars as pl

from ._plugin import ExprLike, FrameLike, OptionType, call
from .option_pricing import _option_args

ALL_GREEKS = ("delta", "gamma", "vega", "theta", "rho", "vanna", "vomma", "charm", "dual_delta")
DEFAULT_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _check(greeks: Iterable[str]) -> tuple:
    greeks = tuple(greeks)
    unknown = set(greeks) - set(ALL_GREEKS)
    if unknown:
        raise ValueError(f"unknown greeks {sorted(unknown)}; choose from {ALL_GREEKS}")
    return greeks


def greeks_struct(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call", greeks: Iterable[str] = DEFAULT_GREEKS,
) -> pl.Expr:
    """The requested Greeks as one Struct expression (fields in the requested order)."""
    greeks = _check(greeks)
    return call("bs_greeks", *_option_args(S, K, T, r, sigma, q, option_type), kwargs={"greeks": list(greeks)})


def _field(struct: pl.Expr, name: str, theta_per_day: bool) -> pl.Expr:
    expr = struct.struct.field(name)
    return expr / 365.0 if (name == "theta" and theta_per_day) else expr


def bs_greeks(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call",
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False,
) -> Dict[str, pl.Expr]:
    """Dict of Greek-name -> expression for the requested Greeks."""
    greeks = _check(greeks)
    struct = greeks_struct(S, K, T, r, sigma, q, option_type, greeks)
    return {g: _field(struct, g, theta_per_day).alias(g) for g in greeks}


def calculate_greeks(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: OptionType = "call", q_col: ExprLike = 0.0,
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False, prefix: str = "",
) -> FrameLike:
    """Add Greek columns to a DataFrame or LazyFrame (lazy in, lazy out).

    ``greeks`` selects which of ``ALL_GREEKS`` to compute (default: delta,
    gamma, vega, theta, rho); ``prefix`` is prepended to each column name.
    """
    greeks = _check(greeks)
    tmp = "__qp_greeks"
    struct = greeks_struct(s_col, k_col, t_col, r_col, sigma_col, q_col, option_type, greeks)
    return (
        df.with_columns(struct.alias(tmp))
        .with_columns(_field(pl.col(tmp), g, theta_per_day).alias(prefix + g) for g in greeks)
        .drop(tmp)
    )


def calculate_vega(df: FrameLike, s_col: str, k_col: str, t_col: str, r_col: str,
                   sigma_col: str, q_col: ExprLike = 0.0) -> FrameLike:
    """Add a ``vega`` column (kept for backward compatibility)."""
    return calculate_greeks(df, s_col, k_col, t_col, r_col, sigma_col, "call", q_col, ("vega",))
