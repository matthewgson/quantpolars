"""Implied volatility as a vectorised Polars query.

The solver follows the structure of Jaeckel's "Let's Be Rational" (2015)
without its asymptotic branches:

1. **Normalise.** Every option is mapped to its out-of-the-money twin by
   put-call parity, then expressed in the dimensionless Black form
   ``b(v) = exp(y/2) N(y/v + v/2) - exp(-y/2) N(y/v - v/2)`` with
   ``y = -|ln(F/K)|`` and ``v = sigma sqrt(T)``. The target is
   ``b* = undiscounted OTM price / sqrt(F K)``, which must lie in
   ``(0, exp(y/2))``; prices outside that range have no implied volatility and
   come back as null.
2. **Start** from the closed-form Stefanica-Radoicic (2017) approximation,
   which inverts Black-Scholes under Polya's normal-CDF approximation and is
   within a few percent of the root almost everywhere.
3. **Iterate** safeguarded Halley steps on ``ln b(v) - ln b*``. With
   ``h = y/v``, ``t = v/2`` and ``E = erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)``,
   ``ln b = -(h^2 + t^2)/2 + ln(E/2)``, its derivative is ``2/(sqrt(2 pi) E)``
   and the second derivative follows from ``b'' = b' d1 d2 / v``; every step is
   a handful of arithmetic operations plus two ``erfcx`` evaluations. Working
   on the logarithm keeps the deep tails solvable where ``b`` itself
   underflows. A bracket ``[lo, hi]`` on ``v`` is maintained; a step that
   leaves the bracket is replaced by bisection, so the iteration cannot
   diverge, and a step below the floating-point spacing of ``v`` stops it.
   Each step is two ``with_columns`` on the lazy query, so the plan grows
   linearly in the number of iterations and runs under the streaming engine.

``b(v)`` is evaluated as ``exp(-(h^2+t^2)/2) [erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)] / 2``,
which keeps relative precision deep in the tails where ``N(d1)`` and ``N(d2)``
underflow.
"""

from __future__ import annotations

import math
from typing import Union

import polars as pl

from ._normal import INV_SQRT2PI, SQRT2, SQRT2PI, ExprLike, _erfcx_nonneg, erfcx, to_expr
from .option_pricing import FrameLike, collect_frame, is_call_expr

V_MIN = 0.0
V_MAX = 30.0   # total volatility sigma*sqrt(T) upper bracket; b(30) is within 1e-50 of b_max
_TWO_OVER_PI = 2.0 / math.pi
_ONE_MINUS_TWO_OVER_PI = 1.0 - _TWO_OVER_PI


def _normalised_black(y: pl.Expr, v: pl.Expr):
    """Return (b, vega_n, d1, d2, scale, E) for the OTM call with log-moneyness y <= 0.

    b = scale * E / 2,  vega_n = scale / sqrt(2 pi),  scale = exp(-(h^2 + t^2)/2),
    E = erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)  (positive, free of underflow).
    """
    h = y / v
    t = 0.5 * v
    d1 = h + t
    d2 = h - t
    scale = (-0.5 * (h * h + t * t)).exp()
    # d2 = h - t < 0 whenever y <= 0, so its erfcx argument never needs the reflection
    E = erfcx(-d1 / SQRT2) - _erfcx_nonneg(-d2 / SQRT2)
    b = 0.5 * scale * E
    vega_n = INV_SQRT2PI * scale
    return b, vega_n, d1, d2, scale, E


def _polya_cdf(z: pl.Expr) -> pl.Expr:
    """Polya's approximation to the standard normal CDF."""
    tail = (1.0 - (-_TWO_OVER_PI * z * z).exp()).sqrt()
    return 0.5 + 0.5 * pl.when(z >= 0).then(tail).otherwise(-tail)


def _sr_initial_guess(y: pl.Expr, c_k: pl.Expr) -> pl.Expr:
    """Stefanica-Radoicic (2017) closed-form total volatility for an OTM call.

    ``y = ln(F/K) <= 0`` and ``c_k`` = undiscounted call price / K.
    """
    ey = y.exp()
    R = 2.0 * c_k - ey + 1.0
    ea = (_ONE_MINUS_TWO_OVER_PI * y).exp()
    eb = (_TWO_OVER_PI * y).exp()
    A = (ea - 1.0 / ea) ** 2
    B = 4.0 * (eb + 1.0 / eb) - 2.0 * (1.0 / ey) * (ea + 1.0 / ea) * (ey * ey + 1.0 - R * R)
    C = (1.0 / (ey * ey)) * (R * R - (ey - 1.0) ** 2) * ((ey + 1.0) ** 2 - R * R)
    beta = 2.0 * C / (B + (B * B + 4.0 * A * C).clip(lower_bound=0.0).sqrt())
    gamma = (-0.5 * math.pi) * beta.log()
    gamma = pl.max_horizontal(gamma, -y)           # d1^2 = gamma + y must be >= 0
    # price at which d1 = 0 (v^2 = -2y); above it d1 > 0
    c0 = 0.5 * ey - _polya_cdf(-((-2.0 * y).sqrt()))
    sp = (gamma + y).sqrt()
    sm = (gamma - y).sqrt()
    return pl.when(c_k <= c0).then(sm - sp).otherwise(sm + sp)


def _implied_total_vol(
    lf: pl.LazyFrame, F: pl.Expr, K: pl.Expr, p_undisc: pl.Expr, is_call: pl.Expr,
    max_iter: int, diagnostics: bool,
) -> pl.LazyFrame:
    """Append ``_iv_v`` (total volatility) and ``_iv_ok``; drops other temporaries."""
    lf = lf.with_columns(_iv_call=is_call, _iv_x=(F / K).log(), _iv_p=p_undisc)
    call, x, p = pl.col("_iv_call"), pl.col("_iv_x"), pl.col("_iv_p")
    # OTM twin by put-call parity (undiscounted): C - P = F - K
    p_otm = (
        pl.when(call & (x > 0)).then(p - (F - K))
        .when(~call & (x < 0)).then(p + (F - K))
        .otherwise(p)
    )
    y = -x.abs()
    b_target = p_otm / (F * K).sqrt()
    b_max = (0.5 * y).exp()
    lf = lf.with_columns(
        _iv_y=y, _iv_b=b_target,
        _iv_ok=(b_target > 0) & (b_target < b_max) & (F > 0) & (K > 0) & call.is_not_null(),
        _iv_ck=p_otm / K,
    )
    yv, bv, ck = pl.col("_iv_y"), pl.col("_iv_b"), pl.col("_iv_ck")
    # Closed-form start where the price is not lost in the cancellation inside
    # the formula; otherwise a tail start from ln b ~ -y^2 / (2 v^2), floored by
    # the at-the-money linearisation v ~ sqrt(2 pi) b.
    guess = _sr_initial_guess(yv, ck)
    ln_b = bv.log()
    v_tail = pl.when(ln_b < 0).then((-yv) / (-2.0 * ln_b).sqrt()).otherwise(0.0)
    fallback = pl.max_horizontal(v_tail, math.sqrt(2.0 * math.pi) * bv)
    sr_ok = (2.0 * ck / (1.0 - yv.exp() + 1e-300) > 1e-6) & guess.is_finite() & (guess > 0)
    v0 = pl.when(sr_ok).then(guess).otherwise(fallback)
    lf = lf.with_columns(
        _iv_v=v0.clip(lower_bound=1e-8, upper_bound=V_MAX - 1e-8),
        _iv_lo=pl.lit(V_MIN), _iv_hi=pl.lit(V_MAX),
    )
    vv = pl.col("_iv_v")
    for _ in range(max_iter):
        b, vega_n, d1, d2, scale, E = _normalised_black(yv, vv)
        # iterate on g(v) = ln b(v) - ln b*: g' = 2 / (sqrt(2 pi) E), g'' = g' (d1 d2 / v - g').
        # ln b is computable where b itself underflows, which is what makes the
        # deep tails solvable; the Halley step uses newton = -g/g' and curv = g''/g'
        g = (0.5 * E).log() + (-0.5 * ((yv / vv) ** 2 + 0.25 * vv * vv)) - ln_b
        gp = 2.0 / (SQRT2PI * E)
        lf = lf.with_columns(_iv_f=g, _iv_newton=-g / gp, _iv_curv=(d1 * d2 / vv - gp))
        f, newton, curv = pl.col("_iv_f"), pl.col("_iv_newton"), pl.col("_iv_curv")
        denom = 1.0 + 0.5 * newton * curv
        step = pl.when(denom > 0.5).then(newton / denom).otherwise(newton)
        lo = pl.when(f < 0).then(vv).otherwise(pl.col("_iv_lo"))
        hi = pl.when(f < 0).then(pl.col("_iv_hi")).otherwise(vv)
        v_new = vv + step
        # a step below the floating-point spacing of v means the root is found;
        # test it before the bracket test, because v itself is now a bracket edge
        converged = (f == 0) | (step.abs() <= 4e-16 * vv)
        inside = (v_new > lo) & (v_new < hi)
        lf = lf.with_columns(
            _iv_lo=lo, _iv_hi=hi,
            _iv_v=pl.when(converged).then(vv).when(inside).then(v_new).otherwise(0.5 * (lo + hi)),
        )
    if diagnostics:
        b, vega_n, d1, d2, scale, E = _normalised_black(yv, vv)
        lf = lf.with_columns(
            iv_resid=((b - bv) / bv).abs(),   # relative price error at the solution
            iv_bracket=pl.col("_iv_hi") - pl.col("_iv_lo"),
        )
    return lf.drop(["_iv_call", "_iv_x", "_iv_p", "_iv_y", "_iv_b", "_iv_ck", "_iv_lo", "_iv_hi",
                    "_iv_f", "_iv_newton", "_iv_curv"], strict=False)


def implied_volatility(
    df: FrameLike, s_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    market_col: ExprLike, option_type: Union[str, pl.Expr, bool] = "call",
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
    max_iter : number of safeguarded Halley iterations (8 reaches machine
        precision from the closed-form start in all but pathological cases).
    diagnostics : also add ``iv_resid`` (relative price error at the solution)
        and ``iv_bracket`` (width of the final bracket on ``sigma sqrt T``).

    Rows whose price lies outside the no-arbitrage range, or with ``T <= 0``,
    get a null.
    """
    S, K, T, r, P, q = map(to_expr, (s_col, k_col, t_col, r_col, market_col, q_col))
    is_call = is_call_expr(option_type)
    F = S * ((r - q) * T).exp()
    p_undisc = P * (r * T).exp()
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = df.lazy().with_columns(_iv_F=F, _iv_T=T)
    lf = _implied_total_vol(lf, pl.col("_iv_F"), K, p_undisc, is_call, max_iter, diagnostics)
    lf = lf.with_columns(
        pl.when(pl.col("_iv_ok") & (pl.col("_iv_T") > 0))
        .then(pl.col("_iv_v") / pl.col("_iv_T").sqrt())
        .otherwise(None)
        .alias(out_col)
    ).drop(["_iv_F", "_iv_T", "_iv_ok", "_iv_v"])
    return collect_frame(lf, lazy_in)


def implied_volatility_black76(
    df: FrameLike, f_col: ExprLike, k_col: ExprLike, t_col: ExprLike, r_col: ExprLike,
    market_col: ExprLike, option_type: Union[str, pl.Expr, bool] = "call",
    out_col: str = "implied_vol", max_iter: int = 8, diagnostics: bool = False,
) -> FrameLike:
    """Black-76 implied volatility from a forward price ``F`` (see ``implied_volatility``)."""
    F, K, T, r, P = map(to_expr, (f_col, k_col, t_col, r_col, market_col))
    is_call = is_call_expr(option_type)
    p_undisc = P * (r * T).exp()
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = df.lazy().with_columns(_iv_F=F, _iv_T=T)
    lf = _implied_total_vol(lf, pl.col("_iv_F"), K, p_undisc, is_call, max_iter, diagnostics)
    lf = lf.with_columns(
        pl.when(pl.col("_iv_ok") & (pl.col("_iv_T") > 0))
        .then(pl.col("_iv_v") / pl.col("_iv_T").sqrt())
        .otherwise(None)
        .alias(out_col)
    ).drop(["_iv_F", "_iv_T", "_iv_ok", "_iv_v"])
    return collect_frame(lf, lazy_in)
