"""Standard normal distribution functions as Polars expressions (Rust kernels).

Accuracy
--------
``erf``, ``erfc`` and ``erfcx`` use W. J. Cody's rational Chebyshev
approximations (as implemented in Jäckel's *Let's Be Rational* code); they agree
with SciPy to about 1e-16 relative error over the whole real line, including the
tails. ``norm_cdf`` keeps full relative precision in the lower tail
(``norm_cdf(-37)`` is 5.7e-300, not zero); in the central region it uses the
same rational polynomial as SciPy's ``ndtr``. ``norm_ppf`` uses Jäckel's
rational approximations from *Let's Be Rational* (about 1e-16 relative).
"""

from __future__ import annotations

import math

import polars as pl

from ._plugin import ExprLike, call, keep_name, to_expr

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
INV_SQRT2PI = 1.0 / SQRT2PI


def _special(func: str, x: ExprLike) -> pl.Expr:
    return keep_name(x, call("special", to_expr(x), kwargs={"func": func}))


def erf(x: ExprLike) -> pl.Expr:
    """Error function, double precision over the real line."""
    return _special("erf", x)


def erfc(x: ExprLike) -> pl.Expr:
    """Complementary error function, double precision over the real line."""
    return _special("erfc", x)


def erfcx(x: ExprLike) -> pl.Expr:
    """Scaled complementary error function exp(x^2) erfc(x), double precision."""
    return _special("erfcx", x)


def norm_cdf(x: ExprLike) -> pl.Expr:
    """Standard normal CDF, with full relative precision in the lower tail."""
    return keep_name(x, call("norm_cdf", to_expr(x)))


def norm_sf(x: ExprLike) -> pl.Expr:
    """Standard normal survival function 1 - N(x), accurate in the upper tail."""
    return _special("norm_sf", x)


def norm_pdf(x: ExprLike) -> pl.Expr:
    """Standard normal density."""
    x_expr = to_expr(x)
    return keep_name(x, INV_SQRT2PI * (-0.5 * x_expr * x_expr).exp())


def norm_ppf(p: ExprLike) -> pl.Expr:
    """Inverse standard normal CDF (quantile function), double precision.

    Returns null outside (0, 1); +/-inf at the endpoints is not generated.
    """
    return _special("norm_ppf", p)
