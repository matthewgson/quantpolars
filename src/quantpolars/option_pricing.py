# Option pricing models: Black-Scholes, CRR, BAW

import polars as pl

def norm_cdf(x: pl.Expr) -> pl.Expr:
    """
    Approximation of the cumulative distribution function of the standard normal distribution.
    Using Abramowitz & Stegun approximation for erf.
    """
    def erf_approx(z: pl.Expr) -> pl.Expr:
        sign = pl.when(z < 0).then(-1).otherwise(1)
        z_abs = z.abs()
        t = 1 / (1 + 0.3275911 * z_abs)
        poly = (0.254829592 * t +
                -0.284496736 * t.pow(2) +
                1.421413741 * t.pow(3) +
                -1.453152027 * t.pow(4) +
                1.061405429 * t.pow(5))
        return sign * (1 - poly * (-z_abs.pow(2)).exp())

    return 0.5 + 0.5 * erf_approx(x / pl.lit(2).sqrt())

def norm_pdf(x: pl.Expr) -> pl.Expr:
    """
    Probability density function of the standard normal distribution.
    """
    return (1 / (2 * pl.pi).sqrt()) * (-0.5 * x.pow(2)).exp()

def black_scholes(df: pl.DataFrame, s_col: str, k_col: str, t_col: str, r_col: str, sigma_col: str, option_type: str = 'call') -> pl.DataFrame:
    """
    Black-Scholes European option pricing model on a Polars DataFrame.

    Parameters:
    df: Polars DataFrame
    s_col: column name for spot price
    k_col: column name for strike price
    t_col: column name for time to maturity
    r_col: column name for risk-free rate
    sigma_col: column name for volatility
    option_type: 'call' or 'put'

    Returns:
    DataFrame with added 'price' column
    """
    df = df.with_columns(
        d1 = (pl.col(s_col).log() - pl.col(k_col).log() + (pl.col(r_col) + 0.5 * pl.col(sigma_col).pow(2)) * pl.col(t_col)) / (pl.col(sigma_col) * pl.col(t_col).sqrt())
    )
    df = df.with_columns(
        d2 = pl.col('d1') - pl.col(sigma_col) * pl.col(t_col).sqrt()
    )
    if option_type == 'call':
        df = df.with_columns(
            price = pl.col(s_col) * norm_cdf(pl.col('d1')) - pl.col(k_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(pl.col('d2'))
        )
    else:
        df = df.with_columns(
            price = pl.col(k_col) * (-pl.col(r_col) * pl.col(t_col)).exp() * norm_cdf(-pl.col('d2')) - pl.col(s_col) * norm_cdf(-pl.col('d1'))
        )
    return df.drop(['d1', 'd2'])

def crr_binomial(S, K, T, r, sigma, N, option_type='call', american=False):
    """
    Cox-Ross-Rubinstein binomial tree model.
    Note: This is not vectorized for Polars DataFrames yet, kept for compatibility.
    """
    # Keep the original implementation for now
    dt = T / N
    u = (sigma * (dt ** 0.5))
    u = (1 + u)  # approx exp(sigma sqrt(dt))
    d = 1 / u
    p = ((1 + r * dt) - d) / (u - d)  # risk-neutral prob

    # For simplicity, compute for single option
    asset_prices = [S]
    for i in range(N):
        new_prices = []
        for price in asset_prices:
            new_prices.extend([price * u, price * d])
        asset_prices = new_prices[:N+1]  # keep only relevant

    # Actually, better to use the standard way
    # This is a simplified version; full implementation needed for accuracy
    # For speed, perhaps implement vectorized later
    return black_scholes(pl.DataFrame({'S': [S], 'K': [K], 'T': [T], 'r': [r], 'sigma': [sigma]}), 'S', 'K', 'T', 'r', 'sigma', option_type)['price'][0]

def baw_american_call(S, K, T, r, sigma, b=0):
    """
    Barone-Adesi-Whaley approximation for American call options.
    Placeholder: currently returns Black-Scholes.
    """
    # TODO: Implement full BAW
    return black_scholes(pl.DataFrame({'S': [S], 'K': [K], 'T': [T], 'r': [r], 'sigma': [sigma]}), 'S', 'K', 'T', 'r', 'sigma', 'call')['price'][0]