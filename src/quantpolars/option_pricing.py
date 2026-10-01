"""Black-Scholes-Merton and Black-76 pricing as pure Polars expressions.

Two layers:

* **Expression layer** (``bs_price``, ``black76_price``): return a ``pl.Expr``
  that can be dropped into any ``select``/``with_columns``. Convenient for a
  single column inside a larger query; every intermediate (d1, d2, N(d1), ...)
  is recomputed inside the one expression.
* **Frame layer** (``black_scholes``, ``black76``): add a ``price`` column to a
  DataFrame or LazyFrame (lazy in, lazy out). The intermediates are
  materialised once as temporary columns, and an eager input is evaluated with
  Polars' streaming engine, which parallelises across rows. This is the fast
  path for large frames: on a 128-core machine it is one to two orders of
  magnitude faster than the single-expression form.

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

import warnings
from typing import Union

import polars as pl

from ._normal import ExprLike, norm_cdf, norm_pdf, to_expr

FrameLike = Union[pl.DataFrame, pl.LazyFrame]

_CALL_TOKENS = ("c", "call", "true", "1", "1.0", "+1")
_PUT_TOKENS = ("p", "put", "false", "-1", "-1.0", "0")


def is_call_expr(option_type: Union[str, pl.Expr, bool]) -> pl.Expr:
    """Boolean expression that is True on call rows, False on puts, null otherwise.

    ``option_type`` is a literal ``'call'``/``'put'`` (any case, or ``'c'``/
    ``'p'``), a bool, an expression, or the name of a column holding per-row
    flags in any of the encodings listed in the module docstring.
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


def _theta(is_call: pl.Expr) -> pl.Expr:
    """+1 for calls, -1 for puts, null for an unrecognised flag."""
    return pl.when(is_call).then(1.0).when(~is_call).then(-1.0).otherwise(None)


def collect_frame(lf: pl.LazyFrame, lazy_in: bool) -> FrameLike:
    """Return the LazyFrame as is, or collect it with the streaming engine."""
    if lazy_in:
        return lf
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # older Polars without the engine keyword
        return lf.collect()


# -----------------------------------------------------------------------------
# Expression layer
# -----------------------------------------------------------------------------
def black76_price(
    F: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    option_type: Union[str, pl.Expr, bool] = "call",
) -> pl.Expr:
    """Black (1976) price of a European option on a forward ``F``.

    price = exp(-rT) * theta * (F N(theta d1) - K N(theta d2)),
    d1 = ln(F/K)/v + v/2, d2 = d1 - v, v = sigma sqrt(T), theta = +1 call / -1 put.
    """
    F, K, T, r, sigma = map(to_expr, (F, K, T, r, sigma))
    th = _theta(is_call_expr(option_type))
    disc = (-r * T).exp()
    v = sigma * T.sqrt()
    x = (F / K).log()
    d1 = x / v + 0.5 * v
    d2 = x / v - 0.5 * v
    live = th * (F * norm_cdf(th * d1) - K * norm_cdf(th * d2))
    intrinsic = (th * (F - K)).clip(lower_bound=0.0)
    missing = F.is_null() | K.is_null() | T.is_null() | sigma.is_null() | r.is_null() | th.is_null()
    alive = (T > 0) & (sigma > 0)
    return pl.when(missing).then(None).when(alive).then(disc * live).otherwise(disc * intrinsic)


def bs_price(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: Union[str, pl.Expr, bool] = "call",
) -> pl.Expr:
    """Black-Scholes-Merton price of a European option on spot ``S``.

    Equivalent to ``black76_price`` with forward ``F = S exp((r - q) T)``.
    """
    S, T, r, q = map(to_expr, (S, T, r, q))
    F = S * ((r - q) * T).exp()
    return black76_price(F, K, T, r, sigma, option_type)


# -----------------------------------------------------------------------------
# Frame layer: materialised intermediates, streaming evaluation
# -----------------------------------------------------------------------------
_CORE_COLS = ("_bs_S", "_bs_F", "_bs_K", "_bs_T", "_bs_sig", "_bs_r", "_bs_q", "_bs_th",
              "_bs_v", "_bs_d1", "_bs_d2", "_bs_dr", "_bs_dq", "_bs_n1", "_bs_n2", "_bs_pdf1",
              "_bs_missing", "_bs_alive")


def _with_core(
    lf: pl.LazyFrame, S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike, option_type, forward: bool,
) -> pl.LazyFrame:
    """Append the shared Black-Scholes intermediates as temporary columns.

    With ``forward=True`` the ``S`` argument is the forward and ``q`` is unused.
    """
    S, K, T, r, sigma, q = map(to_expr, (S, K, T, r, sigma, q))
    th = _theta(is_call_expr(option_type))
    if forward:
        lf = lf.with_columns(_bs_F=S, _bs_K=K, _bs_T=T, _bs_sig=sigma, _bs_r=r, _bs_q=pl.lit(0.0), _bs_th=th)
        lf = lf.with_columns(_bs_S=pl.col("_bs_F") * (-(pl.col("_bs_r")) * pl.col("_bs_T")).exp())
    else:
        lf = lf.with_columns(_bs_S=S, _bs_K=K, _bs_T=T, _bs_sig=sigma, _bs_r=r, _bs_q=q, _bs_th=th)
        lf = lf.with_columns(_bs_F=pl.col("_bs_S") * ((pl.col("_bs_r") - pl.col("_bs_q")) * pl.col("_bs_T")).exp())
    c = pl.col
    lf = lf.with_columns(
        _bs_v=c("_bs_sig") * c("_bs_T").sqrt(),
        _bs_dr=(-c("_bs_r") * c("_bs_T")).exp(),
        _bs_dq=(-c("_bs_q") * c("_bs_T")).exp(),
        _bs_missing=(c("_bs_F").is_null() | c("_bs_K").is_null() | c("_bs_T").is_null()
                     | c("_bs_sig").is_null() | c("_bs_r").is_null() | c("_bs_th").is_null()),
        _bs_alive=(c("_bs_T") > 0) & (c("_bs_sig") > 0),
    )
    lf = lf.with_columns(_bs_d1=(c("_bs_F") / c("_bs_K")).log() / c("_bs_v") + 0.5 * c("_bs_v"))
    lf = lf.with_columns(_bs_d2=c("_bs_d1") - c("_bs_v"))
    lf = lf.with_columns(
        _bs_n1=norm_cdf(c("_bs_th") * c("_bs_d1")),
        _bs_n2=norm_cdf(c("_bs_th") * c("_bs_d2")),
        _bs_pdf1=norm_pdf(c("_bs_d1")),
    )
    return lf


def _price_from_core() -> pl.Expr:
    c = pl.col
    live = c("_bs_th") * (c("_bs_F") * c("_bs_n1") - c("_bs_K") * c("_bs_n2"))
    intrinsic = (c("_bs_th") * (c("_bs_F") - c("_bs_K"))).clip(lower_bound=0.0)
    return (
        pl.when(c("_bs_missing")).then(None)
        .when(c("_bs_alive")).then(c("_bs_dr") * live)
        .otherwise(c("_bs_dr") * intrinsic)
    )


def black_scholes(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: Union[str, pl.Expr, bool] = "call",
    q_col: ExprLike = 0.0, out_col: str = "price",
) -> FrameLike:
    """Add a Black-Scholes-Merton ``price`` column to a DataFrame or LazyFrame.

    Parameters
    ----------
    df : DataFrame or LazyFrame (a LazyFrame is returned lazily; a DataFrame is
        evaluated with the streaming engine).
    s_col, k_col, t_col, r_col, sigma_col : column names, expressions or scalars.
    option_type : ``'call'``/``'put'`` or the name of a per-row flag column.
    q_col : dividend-yield column, expression or scalar (default 0).
    out_col : name of the added column.
    """
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = _with_core(df.lazy(), s_col, k_col, t_col, r_col, sigma_col, q_col, option_type, forward=False)
    lf = lf.with_columns(_price_from_core().alias(out_col)).drop(list(_CORE_COLS), strict=False)
    return collect_frame(lf, lazy_in)


def black76(
    df: FrameLike, f_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: Union[str, pl.Expr, bool] = "call", out_col: str = "price",
) -> FrameLike:
    """Add a Black-76 ``price`` column (option on a forward) to a frame."""
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = _with_core(df.lazy(), f_col, k_col, t_col, r_col, sigma_col, 0.0, option_type, forward=True)
    lf = lf.with_columns(_price_from_core().alias(out_col)).drop(list(_CORE_COLS), strict=False)
    return collect_frame(lf, lazy_in)


# -----------------------------------------------------------------------------
# Legacy scalar helpers kept for API compatibility. They were never real tree
# or BAW implementations; they return the Black-Scholes price and warn.
# -----------------------------------------------------------------------------
def _scalar_bs(S, K, T, r, sigma, option_type):
    df = pl.DataFrame({"S": [float(S)], "K": [float(K)], "T": [float(T)],
                       "r": [float(r)], "sigma": [float(sigma)]})
    return black_scholes(df, "S", "K", "T", "r", "sigma", option_type)["price"][0]


def crr_binomial(S, K, T, r, sigma, N, option_type="call", american=False):
    """Deprecated placeholder: returns the Black-Scholes European price.

    A Cox-Ross-Rubinstein tree is not implemented; ``N`` and ``american`` are
    ignored. Kept so that old code keeps running.
    """
    warnings.warn("crr_binomial is a placeholder returning the Black-Scholes price; "
                  "N and american are ignored.", DeprecationWarning, stacklevel=2)
    return _scalar_bs(S, K, T, r, sigma, option_type)


def baw_american_call(S, K, T, r, sigma, b=0):
    """Deprecated placeholder: returns the Black-Scholes European call price."""
    warnings.warn("baw_american_call is a placeholder returning the Black-Scholes price.",
                  DeprecationWarning, stacklevel=2)
    return _scalar_bs(S, K, T, r, sigma, "call")
