# Normal-distribution expressions vs SciPy
import numpy as np
import polars as pl
import pytest
from scipy import special, stats

from quantpolars import erf, erfc, erfcx, norm_cdf, norm_pdf, norm_ppf, norm_sf

X = np.concatenate([np.linspace(-37, 37, 20001), np.linspace(-0.5, 0.5, 2001),
                    [0.0, 0.46875, -0.46875, 4.0, -4.0, 26.0, -26.0]])
DF = pl.DataFrame({"x": X})


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(b) & (np.abs(b) > 1e-300)
    return np.max(np.abs(a[m] - b[m]) / np.abs(b[m]))


def test_erf_matches_scipy():
    got = DF.select(erf("x"))["x"].to_numpy()
    assert np.max(np.abs(got - special.erf(X))) < 1e-15


def test_erfc_relative_precision_in_tails():
    got = DF.select(erfc("x"))["x"].to_numpy()
    assert _rel(got, special.erfc(X)) < 1e-12


def test_erfcx_matches_scipy():
    got = DF.select(erfcx("x"))["x"].to_numpy()
    assert _rel(got, special.erfcx(X)) < 1e-12


def test_norm_cdf_sf_pdf():
    out = DF.select(cdf=norm_cdf("x"), sf=norm_sf("x"), pdf=norm_pdf("x"))
    assert _rel(out["cdf"], stats.norm.cdf(X)) < 1e-12
    assert _rel(out["sf"], stats.norm.sf(X)) < 1e-12
    assert _rel(out["pdf"], stats.norm.pdf(X)) < 1e-14
    # the lower tail keeps relative precision instead of flushing to zero
    assert out["cdf"][0] > 0 and out["cdf"][0] == pytest.approx(stats.norm.cdf(-37), rel=1e-10)


def test_norm_ppf_matches_scipy():
    p = np.concatenate([np.linspace(1e-12, 1 - 1e-12, 20001), 10.0 ** np.linspace(-300, -1, 500)])
    got = pl.DataFrame({"p": p}).select(norm_ppf("p").alias("q"))["q"].to_numpy()
    ref = stats.norm.ppf(p)
    assert np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)) < 1e-13


def test_norm_ppf_outside_unit_interval_is_null():
    out = pl.DataFrame({"p": [0.0, 1.0, -0.1, 1.5, 0.5]}).select(norm_ppf("p").alias("q"))
    assert out["q"].to_list()[:4] == [None] * 4
    assert out["q"][4] == pytest.approx(0.0, abs=1e-15)


def test_accepts_expressions_and_scalars():
    out = DF.select(a=norm_cdf(pl.col("x") * 2), b=norm_cdf(1.0))
    assert out["a"][0] == pytest.approx(stats.norm.cdf(X[0] * 2), rel=1e-10)
    assert out["b"][0] == pytest.approx(stats.norm.cdf(1.0), rel=1e-14)
