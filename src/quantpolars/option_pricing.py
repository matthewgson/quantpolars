"""Black-Scholes-Merton, Black-76 and American option pricing.

The numerics run in Rust as Polars expression plugins, so every function here
returns (or adds) an ordinary Polars expression: it works in eager, lazy and
streaming queries and inside ``group_by``, never holds the GIL, and uses all
cores (the thread count follows ``POLARS_MAX_THREADS``).

Two layers:

* **Expression layer** (``bs_price``, ``black76_price``, ``crr_price``,
  ``baw_price``): return a ``pl.Expr`` for your own ``select``/``with_columns``.
* **Frame layer** (``black_scholes``, ``black76``): add a ``price`` column to a
  DataFrame or LazyFrame (lazy in, lazy out).

Conventions
-----------
* ``T`` is time to expiry in years, ``r`` the continuously compounded riskless
  rate, ``q`` the continuous dividend (or foreign-rate / carry) yield, ``sigma``
  the annualised volatility. All may be column names, expressions or scalars.
* ``option_type`` may be a literal ``'call'``/``'put'`` or a column holding one
  flag per row. Accepted per-row encodings: ``'C'``/``'P'``, ``'c'``/``'p'``,
  ``'call'``/``'put'`` (any case), booleans (True = call) and ``+1``/``-1``.
  An unrecognised flag gives a null price.
* At or past expiry (``T <= 0``) or with zero volatility the price is the
  (discounted) intrinsic value, never NaN. Null inputs give null outputs.
"""

from __future__ import annotations

from typing import Union

import polars as pl

from ._normal import norm_cdf, norm_pdf  # noqa: F401  (re-exported for compatibility)
from ._plugin import ExprLike, FrameLike, OptionType, call, option_flag, to_expr

_CALL_TOKENS = ("c", "call", "true", "1", "1.0", "+1")
_PUT_TOKENS = ("p", "put", "false", "-1", "-1.0", "0")


def is_call_expr(option_type: Union[str, pl.Expr, bool]) -> pl.Expr:
    """Boolean expression that is True on call rows, False on puts, null otherwise.

    ``option_type`` is a literal ``'call'``/``'put'`` (any case, or ``'c'``/
    ``'p'``), a bool, an expression, or the name of a column holding per-row
    flags in any of the encodings listed in the module docstring. (The pricing
    functions parse flags in Rust; this helper is for your own queries.)
    """
    if isinstance(option_type, bool):
        return pl.lit(option_type)
    if isinstance(option_type, str):
        token = option_type.strip().lower()
        if token in ("call", "c"):
            return pl.lit(True)
        if token in ("put", "p"):
            return pl.lit(False)
        option_type = pl.col(option_type)
    s = option_type.cast(pl.String).str.strip_chars().str.to_lowercase()
    return (
        pl.when(s.is_in(list(_CALL_TOKENS))).then(True)
        .when(s.is_in(list(_PUT_TOKENS))).then(False)
        .otherwise(None)
    )


def _option_args(S, K, T, r, sigma, q, option_type):
    """Argument order expected by the Rust kernels: s, k, t, r, q, sigma, is_call."""
    return (to_expr(S), to_expr(K), to_expr(T), to_expr(r), to_expr(q), to_expr(sigma),
            option_flag(option_type))


# -----------------------------------------------------------------------------
# Expression layer
# -----------------------------------------------------------------------------
def bs_price(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call",
) -> pl.Expr:
    """Black-Scholes-Merton price of a European option on spot ``S``."""
    return call("bs_price", *_option_args(S, K, T, r, sigma, q, option_type))


def black76_price(
    F: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    option_type: OptionType = "call",
) -> pl.Expr:
    """Black (1976) price of a European option on a forward ``F``.

    price = exp(-rT) * theta * (F N(theta d1) - K N(theta d2)), i.e. Black-Scholes-Merton
    with spot ``F`` and dividend yield ``r``.
    """
    return call("bs_price", *_option_args(F, K, T, r, sigma, r, option_type))


def crr_price(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call", steps: int = 200, american: bool = True,
) -> pl.Expr:
    """Cox-Ross-Rubinstein binomial tree price (American exercise by default).

    Cost is O(steps^2) per row; rows are priced in parallel. NaN when ``steps``
    is too small for the rate and volatility (risk-neutral probability outside
    [0, 1]).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return call("crr_price", *_option_args(S, K, T, r, sigma, q, option_type),
                kwargs={"steps": int(steps), "american": bool(american)})


def baw_price(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: OptionType = "call",
) -> pl.Expr:
    """Barone-Adesi & Whaley (1987) American option approximation.

    Fast and accurate for short and medium maturities; its error grows with
    maturity (use ``crr_price`` when that matters).
    """
    return call("baw_price", *_option_args(S, K, T, r, sigma, q, option_type))


# -----------------------------------------------------------------------------
# Frame layer
# -----------------------------------------------------------------------------
def black_scholes(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: OptionType = "call",
    q_col: ExprLike = 0.0, out_col: str = "price",
) -> FrameLike:
    """Add a Black-Scholes-Merton ``price`` column to a DataFrame or LazyFrame.

    Parameters
    ----------
    df : DataFrame or LazyFrame (lazy in, lazy out).
    s_col, k_col, t_col, r_col, sigma_col : column names, expressions or scalars.
    option_type : ``'call'``/``'put'`` or the name of a per-row flag column.
    q_col : dividend-yield column, expression or scalar (default 0).
    out_col : name of the added column.
    """
    return df.with_columns(bs_price(s_col, k_col, t_col, r_col, sigma_col, q_col, option_type).alias(out_col))


def black76(
    df: FrameLike, f_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: OptionType = "call", out_col: str = "price",
) -> FrameLike:
    """Add a Black-76 ``price`` column (option on a forward) to a frame."""
    return df.with_columns(black76_price(f_col, k_col, t_col, r_col, sigma_col, option_type).alias(out_col))


# -----------------------------------------------------------------------------
# Scalar helpers
# -----------------------------------------------------------------------------
def crr_binomial(S, K, T, r, sigma, N=200, option_type="call", american=False, q=0.0) -> float:
    """Cox-Ross-Rubinstein binomial price of a single option (``N`` steps).

    For whole frames use the ``crr_price`` expression.
    """
    return pl.select(crr_price(S, K, T, r, sigma, q, option_type, steps=N, american=american)).item()


def baw_american_call(S, K, T, r, sigma, q=0.0) -> float:
    """Barone-Adesi-Whaley American call price of a single option.

    For whole frames (calls or puts) use the ``baw_price`` expression.
    """
    return pl.select(baw_price(S, K, T, r, sigma, q, "call")).item()
