//! Student-t tail probabilities for the t-tests.

use crate::bsm::ncdf;
use statrs::function::beta::checked_beta_reg;

/// P(T > t) for Student-t with `df` degrees of freedom, computed from the
/// regularised incomplete beta directly so small p-values keep full precision
/// (no `1 - cdf` cancellation).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(df > 0.0)` also rejects NaN
pub fn t_sf(t: f64, df: f64) -> f64 {
    if t.is_nan() || !(df > 0.0) {
        return f64::NAN;
    }
    if df.is_infinite() {
        return ncdf(-t);
    }
    if t.is_infinite() {
        return if t > 0.0 { 0.0 } else { 1.0 };
    }
    // P(|T| > |t|) = I_{df/(df+t^2)}(df/2, 1/2)
    let x = df / (df + t * t);
    let two_tail = checked_beta_reg(0.5 * df, 0.5, x).unwrap_or(f64::NAN);
    if t >= 0.0 { 0.5 * two_tail } else { 1.0 - 0.5 * two_tail }
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Alternative {
    TwoSided,
    Greater,
    Less,
}

pub fn t_pvalue(t: f64, df: f64, alt: Alternative) -> f64 {
    match alt {
        Alternative::TwoSided => (2.0 * t_sf(t.abs(), df)).min(1.0),
        Alternative::Greater => t_sf(t, df),
        Alternative::Less => t_sf(-t, df),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scipy() {
        // scipy.stats.t.sf values
        assert!((t_sf(2.0, 10.0) - 0.036_694_017_385_370_196).abs() < 1e-14);
        assert!((t_sf(-1.5, 3.5) - 0.891_090_906_492_327_5).abs() < 1e-12);
        assert!((t_sf(8.0, 2.5) - 0.003_828_322_065_521_722_5).abs() < 1e-12);
        // Deep tail keeps relative precision instead of rounding to 0.
        let p = t_sf(30.0, 60.0);
        assert!((p / 3.980_833_974_284_883e-38 - 1.0).abs() < 1e-8, "{p}");
    }
}
