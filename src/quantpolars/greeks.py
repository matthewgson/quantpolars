"""Black-Scholes-Merton Greeks as pure Polars expressions.

``bs_greeks`` returns a dict of named expressions (one self-contained
expression per Greek); ``calculate_greeks`` adds the Greeks as columns to a
DataFrame or LazyFrame, sharing the intermediates between them and evaluating
an eager input with the streaming engine. All Greeks are analytic.

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

from typing import Dict, Iterable, Union

import polars as pl

from ._normal import ExprLike, norm_cdf, norm_pdf, to_expr
from .option_pricing import _CORE_COLS, FrameLike, _theta, _with_core, collect_frame, is_call_expr

ALL_GREEKS = ("delta", "gamma", "vega", "theta", "rho", "vanna", "vomma", "charm", "dual_delta")
DEFAULT_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _greek_exprs(S, K, T, r, sigma, q, th, F, v, disc_r, disc_q, d1, d2, pdf1, n_th_d1, n_th_d2,
                 missing, alive, greeks, theta_per_day) -> Dict[str, pl.Expr]:
    sqrt_t = v / sigma  # = sqrt(T) when sigma > 0; only used on alive rows

    def guard(expr: pl.Expr, dead) -> pl.Expr:
        return pl.when(missing).then(None).when(alive).then(expr).otherwise(dead)

    out: Dict[str, pl.Expr] = {}
    if "delta" in greeks:
        dead = disc_q * th * ((th * (F - K)) > 0).cast(pl.Float64)
        out["delta"] = guard(th * disc_q * n_th_d1, dead)
    if "gamma" in greeks:
        out["gamma"] = guard(disc_q * pdf1 / (S * v), 0.0)
    if "vega" in greeks:
        out["vega"] = guard(S * disc_q * pdf1 * sqrt_t, 0.0)
    if "theta" in greeks:
        theta = (
            -S * disc_q * pdf1 * sigma / (2.0 * sqrt_t)
            - th * r * K * disc_r * n_th_d2
            + th * q * S * disc_q * n_th_d1
        )
        if theta_per_day:
            theta = theta / 365.0
        out["theta"] = guard(theta, 0.0)
    if "rho" in greeks:
        out["rho"] = guard(th * K * T * disc_r * n_th_d2, 0.0)
    if "vanna" in greeks:
        out["vanna"] = guard(-disc_q * pdf1 * d2 / sigma, 0.0)
    if "vomma" in greeks:
        out["vomma"] = guard(S * disc_q * pdf1 * sqrt_t * d1 * d2 / sigma, 0.0)
    if "charm" in greeks:
        charm = (
            th * q * disc_q * n_th_d1
            - disc_q * pdf1 * (2.0 * (r - q) * T - d2 * v) / (2.0 * T * v)
        )
        out["charm"] = guard(charm, 0.0)
    if "dual_delta" in greeks:
        out["dual_delta"] = guard(-th * disc_r * n_th_d2, 0.0)
    return out


def _check(greeks: Iterable[str]):
    greeks = tuple(greeks)
    unknown = set(greeks) - set(ALL_GREEKS)
    if unknown:
        raise ValueError(f"unknown greeks {sorted(unknown)}; choose from {ALL_GREEKS}")
    return greeks


def bs_greeks(
    S: ExprLike, K: ExprLike, T: ExprLike, r: ExprLike, sigma: ExprLike,
    q: ExprLike = 0.0, option_type: Union[str, pl.Expr, bool] = "call",
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False,
) -> Dict[str, pl.Expr]:
    """Dict of Greek-name -> self-contained expression for the requested Greeks."""
    greeks = _check(greeks)
    S, K, T, r, sigma, q = map(to_expr, (S, K, T, r, sigma, q))
    th = _theta(is_call_expr(option_type))
    sqrt_t = T.sqrt()
    v = sigma * sqrt_t
    disc_r = (-r * T).exp()
    disc_q = (-q * T).exp()
    F = S * ((r - q) * T).exp()
    d1 = (F / K).log() / v + 0.5 * v
    d2 = d1 - v
    missing = S.is_null() | K.is_null() | T.is_null() | sigma.is_null() | r.is_null() | q.is_null() | th.is_null()
    alive = (T > 0) & (sigma > 0)
    return _greek_exprs(S, K, T, r, sigma, q, th, F, v, disc_r, disc_q, d1, d2, norm_pdf(d1),
                        norm_cdf(th * d1), norm_cdf(th * d2), missing, alive, greeks, theta_per_day)


def calculate_greeks(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    sigma_col: ExprLike, option_type: Union[str, pl.Expr, bool] = "call", q_col: ExprLike = 0.0,
    greeks: Iterable[str] = DEFAULT_GREEKS, theta_per_day: bool = False, prefix: str = "",
) -> FrameLike:
    """Add Greek columns to a DataFrame or LazyFrame (lazy in, lazy out).

    ``greeks`` selects which of ``ALL_GREEKS`` to compute (default: delta,
    gamma, vega, theta, rho); ``prefix`` is prepended to each column name. The
    shared intermediates are materialised once, and an eager input is evaluated
    with the streaming engine.
    """
    greeks = _check(greeks)
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = _with_core(df.lazy(), s_col, k_col, t_col, r_col, sigma_col, q_col, option_type, forward=False)
    c = pl.col
    exprs = _greek_exprs(c("_bs_S"), c("_bs_K"), c("_bs_T"), c("_bs_r"), c("_bs_sig"), c("_bs_q"),
                         c("_bs_th"), c("_bs_F"), c("_bs_v"), c("_bs_dr"), c("_bs_dq"),
                         c("_bs_d1"), c("_bs_d2"), c("_bs_pdf1"), c("_bs_n1"), c("_bs_n2"),
                         c("_bs_missing"), c("_bs_alive"), greeks, theta_per_day)
    lf = lf.with_columns(**{prefix + name: e for name, e in exprs.items()}).drop(list(_CORE_COLS), strict=False)
    return collect_frame(lf, lazy_in)


def calculate_vega(df: FrameLike, s_col: str, k_col: str, t_col: str, r_col: str,
                   sigma_col: str, q_col: ExprLike = 0.0) -> FrameLike:
    """Add a ``vega`` column (kept for backward compatibility)."""
    return calculate_greeks(df, s_col, k_col, t_col, r_col, sigma_col, "call", q_col, ("vega",))
