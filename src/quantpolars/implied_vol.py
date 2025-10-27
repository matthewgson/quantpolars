# Implied volatility calculation

import polars as pl
from .option_pricing import black_scholes
from .greeks import calculate_vega

def implied_volatility(df: pl.DataFrame, s_col: str, k_col: str, t_col: str, r_col: str, market_col: str, option_type: str = 'call') -> pl.DataFrame:
    """
    Calculate implied volatility using Newton-Raphson on a Polars DataFrame.

    Parameters:
    df: Polars DataFrame
    s_col: column name for spot price
    k_col: column name for strike price
    t_col: column name for time to maturity
    r_col: column name for risk-free rate
    market_col: column name for observed option price
    option_type: 'call' or 'put'

    Returns:
    DataFrame with added 'implied_vol' column
    """
    # Initial guess
    sigma = pl.lit(0.2)

    # Unroll 3 Newton-Raphson iterations for speed
    for _ in range(3):
        temp_df = df.with_columns(sigma_iter=sigma)
        temp_df = black_scholes(temp_df, s_col, k_col, t_col, r_col, 'sigma_iter', option_type)
        temp_df = calculate_vega(temp_df, s_col, k_col, t_col, r_col, 'sigma_iter')
        sigma = pl.col('sigma_iter') - (pl.col('price') - pl.col(market_col)) / pl.col('vega')
        temp_df = temp_df.drop(['price', 'vega', 'sigma_iter'])

    return df.with_columns(implied_vol=sigma)