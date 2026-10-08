# Tests for American option models (CRR binomial, Barone-Adesi-Whaley)

import polars as pl
import pytest

import quantpolars as qp

BS_CALL = 10.450583572185565


def test_crr_european_converges_to_black_scholes():
    assert qp.crr_binomial(100, 100, 1, 0.05, 0.2, 2000) == pytest.approx(BS_CALL, abs=2e-3)


def test_crr_american_put_reference():
    # Standard benchmark: S=K=100, T=1, r=5%, sigma=20% -> 6.0903
    assert qp.crr_binomial(100, 100, 1, 0.05, 0.2, 2000, "put", american=True) == pytest.approx(6.0903, abs=2e-3)


def test_american_call_without_dividends_equals_european():
    eu = qp.crr_binomial(100, 100, 1, 0.05, 0.2, 500, "call", american=False)
    am = qp.crr_binomial(100, 100, 1, 0.05, 0.2, 500, "call", american=True)
    assert am == pytest.approx(eu, abs=1e-12)
    assert qp.baw_american_call(100, 100, 1, 0.05, 0.2) == pytest.approx(BS_CALL, abs=1e-12)


def test_baw_close_to_tree_on_a_frame():
    df = pl.DataFrame({
        "S": [80.0, 100.0, 120.0, 100.0], "K": 100.0, "T": [0.25, 0.5, 1.0, 1.0],
        "r": 0.08, "q": [0.0, 0.04, 0.12, 0.0], "sigma": [0.2, 0.3, 0.25, 0.4],
        "cp": ["put", "put", "call", "put"],
    })
    out = df.with_columns(
        baw=qp.baw_price("S", "K", "T", "r", "sigma", "q", "cp"),
        tree=qp.crr_price("S", "K", "T", "r", "sigma", "q", "cp", steps=2000),
        euro=qp.bs_price("S", "K", "T", "r", "sigma", "q", "cp"),
    )
    # BAW is an approximation; its error grows with maturity.
    assert ((out["baw"] - out["tree"]).abs() < 0.01 * out["tree"]).all()
    # Early exercise premium is never negative.
    assert (out["tree"] >= out["euro"] - 1e-9).all()


def test_deep_itm_put_is_exercised():
    df = pl.DataFrame({"S": [50.0]})
    out = df.select(qp.baw_price("S", 100.0, 1.0, 0.05, 0.2, option_type="put"))
    assert out.item() == 50.0


def test_crr_validates_steps():
    with pytest.raises(ValueError):
        qp.crr_price("S", "K", "T", "r", "sigma", steps=0)
