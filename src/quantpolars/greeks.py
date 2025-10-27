# Greeks calculation

import polars as pl

def norm_pdf(x: pl.Expr) -> pl.Expr:
    """
    Probability density function of the standard normal distribution.
    """
    return (1 / (2 * pl.pi).sqrt()) * (-0.5 * x.pow(2)).exp()

def calculate_greeks(df: pl.DataFrame, s_col: str, k_col: str, t_col: str, r_col: str, sigma_col: str, option_type: str = 'call') -> pl.DataFrame:
    """
    Calculate option Greeks using Black-Scholes model on a Polars DataFrame.

    Parameters:
    df: Polars DataFrame
    s_col: column name for spot price
    k_col: column name for strike price
    t_col: column name for time to maturity
    r_col: column name for risk-free rate
    sigma_col: column name for volatility
    option_type: 'call' or 'put'

    Returns:
    DataFrame with added Greek columns: delta, gamma, theta, vega, rho
    """
    df = df.with_columns(
        d1 = (pl.col(s_col).log() - pl.col(k_col).log() + (pl.col(r_col) + 0.5 * pl.col(sigma_col).pow(2)) * pl.col(t_col)) / (pl.col(sigma_col) * pl.col(t_col).sqrt()),
        d2 = pl.col('d1') - pl.col(sigma_col) * pl.col(t_col).sqrt()
    )

    pdf_d1 = norm_pdf(pl.col('d1'))

    if option_type == 'call':
        delta = pl.when(pl.col(t_col) <= 0).then(1).otherwise(
            pl.when(pl.col(s_col) > pl.col(k_col)).then(1).otherwise(0)
        ).otherwise(
            norm_cdf(pl.col('d1'))
        )
        theta = - (pl.col(s_col) * pl.col(sigma_col) * pdf_d1) / (2 * pl.col(t_col).sqrt()) - pl.col(r_col) * pl.col(k_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(pl.col('d2'))
        rho = pl.col(k_col) * pl.col(t_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(pl.col('d2'))
    else:
        delta = pl.when(pl.col(t_col) <= 0).then(-1).otherwise(
            pl.when(pl.col(s_col) < pl.col(k_col)).then(-1).otherwise(0)
        ).otherwise(
            norm_cdf(pl.col('d1')) - 1
        )
        theta = - (pl.col(s_col) * pl.col(sigma_col) * pdf_d1) / (2 * pl.col(t_col).sqrt()) + pl.col(r_col) * pl.col(k_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(-pl.col('d2'))
        rho = -pl.col(k_col) * pl.col(t_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(-pl.col('d2'))

    gamma = pdf_d1 / (pl.col(s_col) * pl.col(sigma_col) * pl.col(t_col).sqrt())
    vega = pl.col(s_col) * pl.col(t_col).sqrt() * pdf_d1

    df = df.with_columns(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    return df.drop(['d1', 'd2'])

def calculate_vega(df: pl.DataFrame, s_col: str, k_col: str, t_col: str, r_col: str, sigma_col: str) -> pl.DataFrame:
    """
    Calculate vega for options on a Polars DataFrame.
    """
    df = df.with_columns(
        d1 = (pl.col(s_col).log() - pl.col(k_col).log() + (pl.col(r_col) + 0.5 * pl.col(sigma_col).pow(2)) * pl.col(t_col)) / (pl.col(sigma_col) * pl.col(t_col).sqrt())
    )
    pdf_d1 = norm_pdf(pl.col('d1'))
    vega = pl.col(s_col) * pl.col(t_col).sqrt() * pdf_d1
    return df.with_columns(vega=vega).drop('d1')

# Import norm_cdf from option_pricing
from .option_pricing import norm_cdf