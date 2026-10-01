"""Standard normal distribution functions as pure Polars expressions.

Everything here compiles to Polars/Rust expression trees: no Python UDFs, no
NumPy, no SciPy. The functions therefore run inside lazy and streaming
queries and parallelise across cores like any other Polars expression.

Accuracy
--------
``erf``, ``erfc`` and ``erfcx`` implement W. J. Cody's rational Chebyshev
approximations (Cody 1969, Netlib ``specfun/calerf``). They are accurate to
roughly 1e-16 relative error over the whole real line, including the tails,
which is what option pricing needs: a far out-of-the-money option price is a
tail probability, and an approximation with a 1e-7 *absolute* error (such as
Abramowitz-Stegun 7.1.26) has a relative error of order one there.

``norm_cdf`` is evaluated through ``erfc`` so that the lower tail keeps full
relative precision; ``norm_ppf`` implements Wichura's AS241 (``PPND16``),
accurate to about 1e-16.
"""

from __future__ import annotations

import math
from typing import Union

import polars as pl

ExprLike = Union[str, pl.Expr, float, int]

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
INV_SQRT2PI = 1.0 / SQRT2PI
INV_SQRTPI = 0.56418958354775628695  # 1/sqrt(pi)

# -----------------------------------------------------------------------------
# Cody (1969) coefficients, from Netlib specfun CALERF.
# -----------------------------------------------------------------------------
_A = (3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
      3.20937758913846947e03, 1.85777706184603153e-1)
_B = (2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03,
      2.84423683343917062e03)
_C = (5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
      2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03,
      2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8)
_D = (1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02,
      1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03,
      3.43936767414372164e03, 1.23033935480374942e03)
_P = (3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
      1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2)
_Q = (2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1,
      6.05183413124413191e-2, 2.33520497626869185e-3)

_THRESH = 0.46875
_XNEG = -26.628       # erfcx overflows below this


def to_expr(x: ExprLike) -> pl.Expr:
    """Column name -> pl.col, number -> pl.lit, expression -> itself."""
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    return pl.lit(float(x))


def _keep_name(x: ExprLike, expr: pl.Expr) -> pl.Expr:
    """If the input was a column name, the output keeps that name."""
    return expr.alias(x) if isinstance(x, str) else expr


def _cody_pieces(y: pl.Expr):
    """Return (erf_small, erfcx_mid, erfcx_large) for y = |x| >= 0.

    * ``erf_small``  : erf(y) on the region y <= 0.46875 (as a function of y)
    * ``erfcx_mid``  : exp(y^2) erfc(y) on 0.46875 < y <= 4
    * ``erfcx_large``: exp(y^2) erfc(y) on y > 4
    Each piece is only meaningful on its own region.
    """
    # Region 1: erf(y) = y * R(y^2)
    ysq = y * y
    xnum = _A[4] * ysq
    xden = ysq
    for i in range(3):
        xnum = (xnum + _A[i]) * ysq
        xden = (xden + _B[i]) * ysq
    erf_small = y * (xnum + _A[3]) / (xden + _B[3])

    # Region 2: erfcx(y) = R(y)
    xnum = _C[8] * y
    xden = y
    for i in range(7):
        xnum = (xnum + _C[i]) * y
        xden = (xden + _D[i]) * y
    erfcx_mid = (xnum + _C[7]) / (xden + _D[7])

    # Region 3: erfcx(y) = (1/sqrt(pi) - R(1/y^2)/y^2) / y
    inv = 1.0 / ysq
    xnum = _P[5] * inv
    xden = inv
    for i in range(4):
        xnum = (xnum + _P[i]) * inv
        xden = (xden + _Q[i]) * inv
    erfcx_large = (INV_SQRTPI - inv * (xnum + _P[4]) / (xden + _Q[4])) / y
    return erf_small, erfcx_mid, erfcx_large


def _exp_neg_sq(y: pl.Expr) -> pl.Expr:
    """exp(-y^2) computed as Cody does, to keep precision for large y."""
    ysq = (y * 16.0).floor() / 16.0
    delta = (y - ysq) * (y + ysq)
    return (-ysq * ysq).exp() * (-delta).exp()


def _erfcx_nonneg(y: pl.Expr) -> pl.Expr:
    """erfcx(y) for y >= 0 only (no reflection branch); used by the IV solver."""
    erf_small, erfcx_mid, erfcx_large = _cody_pieces(y)
    return (
        pl.when(y <= _THRESH).then((y * y).exp() * (1.0 - erf_small))
        .when(y <= 4.0).then(erfcx_mid)
        .otherwise(erfcx_large)
    )


def erfcx(x: ExprLike) -> pl.Expr:
    """Scaled complementary error function exp(x^2) erfc(x), double precision."""
    x_in, x = x, to_expr(x)
    y = x.abs()
    # erfcx on the positive half line
    pos = _erfcx_nonneg(y)
    # reflection: erfcx(-y) = 2 exp(y^2) - erfcx(y)
    ysq = (y * 16.0).floor() / 16.0
    delta = (y - ysq) * (y + ysq)
    two_exp = 2.0 * (ysq * ysq).exp() * delta.exp()
    return _keep_name(x_in, pl.when(x >= 0).then(pos).otherwise(two_exp - pos))


def erfc(x: ExprLike) -> pl.Expr:
    """Complementary error function, double precision over the real line."""
    x_in, x = x, to_expr(x)
    y = x.abs()
    erf_small, erfcx_mid, erfcx_large = _cody_pieces(y)
    pos = (
        pl.when(y <= _THRESH).then(1.0 - erf_small)
        .when(y <= 4.0).then(_exp_neg_sq(y) * erfcx_mid)
        .otherwise(_exp_neg_sq(y) * erfcx_large)
    )
    return _keep_name(x_in, pl.when(x >= 0).then(pos).otherwise(2.0 - pos))


def erf(x: ExprLike) -> pl.Expr:
    """Error function, double precision over the real line."""
    x_in, x = x, to_expr(x)
    y = x.abs()
    erf_small, erfcx_mid, erfcx_large = _cody_pieces(y)
    pos = (
        pl.when(y <= _THRESH).then(erf_small)
        .when(y <= 4.0).then(1.0 - _exp_neg_sq(y) * erfcx_mid)
        .otherwise(1.0 - _exp_neg_sq(y) * erfcx_large)
    )
    return _keep_name(x_in, pl.when(x >= 0).then(pos).otherwise(-pos))


def norm_cdf(x: ExprLike) -> pl.Expr:
    """Standard normal CDF, N(x) = erfc(-x / sqrt 2) / 2.

    Evaluated through ``erfc`` so the lower tail keeps full relative precision
    (N(-38) is returned as 2.9e-316, not 0).
    """
    return _keep_name(x, 0.5 * erfc(-to_expr(x) / SQRT2))


def norm_sf(x: ExprLike) -> pl.Expr:
    """Standard normal survival function 1 - N(x), accurate in the upper tail."""
    return _keep_name(x, 0.5 * erfc(to_expr(x) / SQRT2))


def norm_pdf(x: ExprLike) -> pl.Expr:
    """Standard normal density."""
    x_in, x = x, to_expr(x)
    return _keep_name(x_in, INV_SQRT2PI * (-0.5 * x * x).exp())


# -----------------------------------------------------------------------------
# AS241 (Wichura 1988) PPND16: inverse normal CDF to ~1e-16.
# -----------------------------------------------------------------------------
_PA = (3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
       1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
       3.3430575583588128105e+4, 2.5090809287301226727e+3)
_PB = (4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
       2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
       5.2264952788528545610e+3)
_PC = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
       3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
       2.27238449892691845833e-2, 7.74545014278341407640e-4)
_PD = (2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
       1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
       1.05075007164441684324e-9)
_PE = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
       2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
       2.71155556874348757815e-5, 2.01033439929228813265e-7)
_PF = (5.9983220655588793769e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
       7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
       2.04426310338993978564e-15)


def _horner(coefs, r: pl.Expr) -> pl.Expr:
    """coefs[0] + coefs[1] r + ... evaluated by Horner's rule."""
    acc = pl.lit(coefs[-1])
    for c in reversed(coefs[:-1]):
        acc = acc * r + c
    return acc


def norm_ppf(p: ExprLike) -> pl.Expr:
    """Inverse standard normal CDF (quantile function), AS241 PPND16.

    Returns null outside (0, 1); +/-inf at the endpoints is not generated.
    """
    p_in, p = p, to_expr(p)
    q = p - 0.5
    # central region
    r_c = 0.180625 - q * q
    central = q * _horner(_PA, r_c) / (_horner(_PB, r_c) * r_c + 1.0)
    # tails
    r_t = pl.when(q < 0).then(p).otherwise(1.0 - p)
    r_t = (-(r_t.log())).sqrt()
    r1 = r_t - 1.6
    tail_near = _horner(_PC, r1) / (_horner(_PD, r1) * r1 + 1.0)
    r2 = r_t - 5.0
    tail_far = _horner(_PE, r2) / (_horner(_PF, r2) * r2 + 1.0)
    tail = pl.when(r_t <= 5.0).then(tail_near).otherwise(tail_far)
    tail = pl.when(q < 0).then(-tail).otherwise(tail)
    return _keep_name(p_in, (
        pl.when((p <= 0) | (p >= 1)).then(None)
        .when(q.abs() <= 0.425).then(central)
        .otherwise(tail)
    ))
