# Test for option_pricing

import pytest
import polars as pl
from quantpolars import black_scholes, crr_binomial

def test_black_scholes_call():
    df = pl.DataFrame({
        'S': [100.0],
        'K': [100.0],
        'T': [1.0],
        'r': [0.05],
        'sigma': [0.2]
    })
    result_df = black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'call')
    price = result_df['price'][0]
    expected = 10.450583572185565  # Approximate value
    assert abs(price - expected) < 1e-5  # Adjusted tolerance for approximation

def test_black_scholes_put():
    df = pl.DataFrame({
        'S': [100.0],
        'K': [100.0],
        'T': [1.0],
        'r': [0.05],
        'sigma': [0.2]
    })
    result_df = black_scholes(df, 'S', 'K', 'T', 'r', 'sigma', 'put')
    price = result_df['price'][0]
    expected = 5.573526022256971  # Approximate value
    assert abs(price - expected) < 1e-5

def test_crr_binomial():
    # For now, test with single value
    price = crr_binomial(100, 100, 1, 0.05, 0.2, 100, 'call', american=False)
    expected = 10.450583572185565  # Should converge to BS
    assert abs(price - expected) < 0.1