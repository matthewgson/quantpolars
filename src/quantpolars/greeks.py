"""Black-Scholes-Merton Greeks (Rust kernels).

``bs_greeks`` returns a dict of named expressions; ``calculate_greeks`` adds the
Greeks as columns to a DataFrame or LazyFrame. Either way the requested Greeks
come from one pass over the rows that computes d1, d2, N(d1), N(d2) and the
density once and skips whatever no requested Greek needs. All Greeks are
analytic.

Volatility
----------
By default the Greeks are evaluated at each option's **implied volatility**:
pass the observed option price (``price=`` / ``market_col=``) and the vol is
backed out per row with ``bs_iv`` before the Greeks are computed. To evaluate
at a volatility of your own instead, pass it explicitly as ``sigma`` /
``sigma_col``. Passing both, or neither, raises ``ValueError``.

Rows whose price admits no implied volatility (at or below intrinsic value, at
or above the no-arbitrage upper bound, non-positive) get null Greeks. Rows at or
past expiry get the expiry values below, as they do with an explicit ``sigma``.

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

from typing import Dict, Iterable, Optional

import polars as pl

from ._plugin import ExprLike, FrameLike, OptionType, call, to_expr
from .implied_vol import bs_iv
from .option_pricing import _option_args

ALL_GREEKS = ("delta", "gamma", "vega", "theta", "rho", "vanna", "vomma", "charm", "dual_delta")
DEFAULT_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _check(greeks: Iterable[str]) -> tuple:
    greeks = tuple(greeks)
    unknown = set(greeks) - set(ALL_GREEKS)
    if unknown:
        raise ValueError(f"unknown greeks {sorted(unknown)}; choose from {ALL_GREEKS}")
    return greeks


def _volatility(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: Optional[ExprLike],
    price: Optional[ExprLike], q: ExprLike, option_type: OptionType,
) -> ExprLike:
    """The volatility the Greeks are evaluated at: ``sigma`` if given, else the
    implied volatility of ``price``."""
    if sigma is not None and price is not None:
        raise ValueError(
            "pass either a price (Greeks at the implied volatility) or sigma "
            "(Greeks at that volatility), not both"
        )
    if sigma is not None:
        return sigma
    if price is None:
        raise ValueError(
            "pass the option price (price= / market_col=) to evaluate the Greeks at the "
            "implied volatility, or sigma (sigma= / sigma_col=) to use a volatility of your own"
        )
    iv = bs_iv(price, S, K, T, r, q, option_type)
    # No implied vol exists at or past expiry, but the Greeks there do not depend on
    # the vol; any value gives the expiry Greeks (delta = exercise indicator, rest 0).
    expired = (to_expr(T) <= 0) & to_expr(price).is_not_null()
    return pl.when(expired).then(0.0).otherwise(iv)


def greeks_struct(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: Optional[ExprLike] = None,
    q: ExprLike = 0.0, option_type: OptionType = "call", greeks: Iterable[str] = DEFAULT_GREEKS,
    *, price: Optional[ExprLike] = None,
) -> pl.Expr:
    """The requested Greeks as one Struct expression (fields in the requested order).

    Evaluated at the implied volatility of ``price`` (the observed option price),
    or at ``sigma`` if that is given instead.
    """
    greeks = _check(greeks)
    vol = _volatility(S, K, T, r, sigma, price, q, option_type)
    return call("bs_greeks", *_option_args(S, K, T, r, vol, q, option_type), kwargs={"greeks": list(greeks)})


def _field(struct: pl.Expr, name: str, theta_per_day: bool) -> pl.Expr:
    expr = struct.struct.field(name)
    return expr / 365.0 if (name == "theta" and theta_per_day) else expr


def bs_greeks(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: Optional[ExprLike] = None,
    q: ExprLike = 0.0, option_type: OptionType = "call",
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False,
    *, price: Optional[ExprLike] = None,
) -> Dict[str, pl.Expr]:
    """Dict of Greek-name -> expression for the requested Greeks.

    Evaluated at the implied volatility of ``price`` (the observed option price),
    or at ``sigma`` if that is given instead.
    """
    greeks = _check(greeks)
    struct = greeks_struct(S, K, T, r, sigma, q, option_type, greeks, price=price)
    return {g: _field(struct, g, theta_per_day).alias(g) for g in greeks}


def calculate_greeks(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: Optional[ExprLike] = None, option_type: OptionType = "call", q_col: ExprLike = 0.0,
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False, prefix: str = "",
    *, market_col: Optional[ExprLike] = None,
) -> FrameLike:
    """Add Greek columns to a DataFrame or LazyFrame (lazy in, lazy out).

    Pass ``market_col`` (the observed option price) to evaluate the Greeks at
    each row's implied volatility; pass ``sigma_col`` only to use a volatility
    of your own instead. ``greeks`` selects which of ``ALL_GREEKS`` to compute
    (default: delta, gamma, vega, theta, rho); ``prefix`` is prepended to each
    column name.
    """
    greeks = _check(greeks)
    tmp = "__qp_greeks"
    struct = greeks_struct(s_col, k_col, t_col, r_col, sigma_col, q_col, option_type, greeks, price=market_col)
    return (
        df.with_columns(struct.alias(tmp))
        .with_columns(_field(pl.col(tmp), g, theta_per_day).alias(prefix + g) for g in greeks)
        .drop(tmp)
    )


def calculate_vega(df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
                   sigma_col: Optional[ExprLike] = None, q_col: ExprLike = 0.0, *,
                   market_col: Optional[ExprLike] = None, option_type: OptionType = "call") -> FrameLike:
    """Add a ``vega`` column (kept for backward compatibility).

    As ``calculate_greeks``: at the implied volatility of ``market_col``, or at
    ``sigma_col`` if given. ``option_type`` only matters for backing out the
    implied volatility (vega itself is the same for calls and puts).
    """
    return calculate_greeks(df, s_col, k_col, t_col, r_col, sigma_col, option_type, q_col, ("vega",),
                            market_col=market_col)
