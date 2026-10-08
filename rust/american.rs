//! American option models: Cox-Ross-Rubinstein binomial tree and the
//! Barone-Adesi & Whaley (1987) quadratic approximation.

use crate::bsm::{ncdf, npdf, price as bsm_price};

/// CRR binomial price with continuous dividend yield `q`. `buf` is scratch
/// space reused across rows so the hot loop does not allocate.
#[allow(clippy::too_many_arguments)]
pub fn crr(
    s: f64, k: f64, t: f64, r: f64, q: f64, v: f64, call: bool,
    steps: usize, american: bool, buf: &mut Vec<f64>,
) -> f64 {
    if !(s >= 0.0 && k >= 0.0 && t >= 0.0 && v >= 0.0) || steps == 0 {
        return f64::NAN;
    }
    if t == 0.0 || v == 0.0 {
        let euro = bsm_price(s, k, t, r, q, v, call);
        let intrinsic = if call { s - k } else { k - s };
        return if american { euro.max(intrinsic) } else { euro };
    }

    let n = steps;
    let dt = t / n as f64;
    let ln_u = v * dt.sqrt();
    let (u, d) = (ln_u.exp(), (-ln_u).exp());
    let p = (((r - q) * dt).exp() - d) / (u - d);
    if !(0.0..=1.0).contains(&p) {
        // Too few steps for this carry/vol: the tree admits arbitrage.
        return f64::NAN;
    }
    let disc = (-r * dt).exp();
    let (pu, pd) = (disc * p, disc * (1.0 - p));
    let u2 = u * u;
    let sign = if call { 1.0 } else { -1.0 };

    buf.clear();
    buf.extend((0..=n).map(|j| {
        let st = s * ((2.0 * j as f64 - n as f64) * ln_u).exp();
        (sign * (st - k)).max(0.0)
    }));

    for i in (0..n).rev() {
        if american {
            let mut st = s * (-(i as f64) * ln_u).exp();
            for j in 0..=i {
                let cont = pu * buf[j + 1] + pd * buf[j];
                buf[j] = cont.max(sign * (st - k));
                st *= u2;
            }
        } else {
            for j in 0..=i {
                buf[j] = pu * buf[j + 1] + pd * buf[j];
            }
        }
    }
    buf[0]
}

/// Generalised BSM with cost of carry `b` (b = r - q).
#[inline]
fn gbs(s: f64, k: f64, t: f64, r: f64, b: f64, v: f64, call: bool) -> f64 {
    bsm_price(s, k, t, r, r - b, v, call)
}

const BAW_TOL: f64 = 1e-8;
const BAW_MAX_ITER: usize = 200;

/// Barone-Adesi & Whaley American approximation (Haug, 2007, sec. 3.3.2).
pub fn baw(s: f64, k: f64, t: f64, r: f64, q: f64, v: f64, call: bool) -> f64 {
    if !(s >= 0.0 && k > 0.0 && t >= 0.0 && v >= 0.0) {
        return f64::NAN;
    }
    let b = r - q;
    let euro = gbs(s, k, t, r, b, v, call);
    let intrinsic = if call { s - k } else { k - s };
    if t == 0.0 || v == 0.0 {
        return euro.max(intrinsic);
    }
    // Early exercise is never optimal for a call when b >= r, or for a put when r <= 0.
    if (call && b >= r) || (!call && r <= 0.0) {
        return euro;
    }

    let sqrt_t = t.sqrt();
    let vst = v * sqrt_t;
    let nn = 2.0 * b / (v * v);
    let m = 2.0 * r / (v * v);
    let kk = 1.0 - (-r * t).exp();
    let carry = ((b - r) * t).exp();
    let d1 = |x: f64| ((x / k).ln() + (b + 0.5 * v * v) * t) / vst;
    let disc = ((nn - 1.0) * (nn - 1.0) + 4.0 * m / kk).sqrt();

    if call {
        let q2 = (-(nn - 1.0) + disc) / 2.0;
        // Seed for the critical price (Barone-Adesi & Whaley eq. 30)
        let q2_inf = (-(nn - 1.0) + ((nn - 1.0) * (nn - 1.0) + 4.0 * m).sqrt()) / 2.0;
        let s_inf = k / (1.0 - 1.0 / q2_inf);
        let h2 = -(b * t + 2.0 * vst) * k / (s_inf - k);
        let mut si = k + (s_inf - k) * (1.0 - h2.exp());

        for _ in 0..BAW_MAX_ITER {
            let x = d1(si);
            let rhs = gbs(si, k, t, r, b, v, true) + (1.0 - carry * ncdf(x)) * si / q2;
            if ((si - k) - rhs).abs() / k < BAW_TOL {
                break;
            }
            let slope = carry * ncdf(x) * (1.0 - 1.0 / q2) + (1.0 - carry * npdf(x) / vst) / q2;
            si = (k + rhs - slope * si) / (1.0 - slope);
        }
        if !si.is_finite() || si <= 0.0 {
            return f64::NAN;
        }
        if s < si {
            let a2 = (si / q2) * (1.0 - carry * ncdf(d1(si)));
            euro + a2 * (s / si).powf(q2)
        } else {
            intrinsic
        }
    } else {
        let q1 = (-(nn - 1.0) - disc) / 2.0;
        let q1_inf = (-(nn - 1.0) - ((nn - 1.0) * (nn - 1.0) + 4.0 * m).sqrt()) / 2.0;
        let s_inf = k / (1.0 - 1.0 / q1_inf);
        let h1 = (b * t - 2.0 * vst) * k / (k - s_inf);
        let mut si = s_inf + (k - s_inf) * h1.exp();

        for _ in 0..BAW_MAX_ITER {
            let x = d1(si);
            let rhs = gbs(si, k, t, r, b, v, false) - (1.0 - carry * ncdf(-x)) * si / q1;
            if ((k - si) - rhs).abs() / k < BAW_TOL {
                break;
            }
            let slope = -carry * ncdf(-x) * (1.0 - 1.0 / q1) - (1.0 + carry * npdf(x) / vst) / q1;
            si = (k - rhs + slope * si) / (1.0 + slope);
        }
        if !si.is_finite() || si <= 0.0 {
            return f64::NAN;
        }
        if s > si {
            let a1 = -(si / q1) * (1.0 - carry * ncdf(-d1(si)));
            euro + a1 * (s / si).powf(q1)
        } else {
            intrinsic
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T = 0.1, K = 100, r = 0.10, b = 0 (q = r), sigma = 0.15 (Haug 2007, Table 3-1 inputs).
    // References solve the BAW critical-price equation to 1e-14 with Brent's method.
    #[test]
    fn baw_reference_values() {
        let cases = [
            (90.0, true, 0.020_635_595), (100.0, true, 1.876_920_687), (110.0, true, 10.006_060_055),
        ];
        for (s, call, want) in cases {
            let got = baw(s, 100.0, 0.1, 0.10, 0.10, 0.15, call);
            assert!((got - want).abs() < 1e-6, "S={s} call={call}: {got} vs {want}");
        }
        // Below the put's critical price the option is exercised immediately.
        assert_eq!(baw(80.0, 100.0, 0.1, 0.10, 0.10, 0.15, false), 20.0);
    }

    #[test]
    fn baw_is_close_to_binomial_american() {
        let mut buf = Vec::new();
        for &(s, q, call) in &[(100.0, 0.0, false), (90.0, 0.03, false), (110.0, 0.08, true), (100.0, 0.05, true)] {
            let a = baw(s, 100.0, 0.5, 0.06, q, 0.3, call);
            let b = crr(s, 100.0, 0.5, 0.06, q, 0.3, call, 5000, true, &mut buf);
            assert!((a - b).abs() < 0.05, "S={s} q={q} call={call}: baw {a} vs crr {b}");
        }
    }

    #[test]
    fn crr_converges_to_bsm_and_prices_american_put() {
        let mut buf = Vec::new();
        let euro = crr(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, true, 2000, false, &mut buf);
        assert!((euro - 10.450_583_572_185_565).abs() < 2e-3);
        // Well-known American put reference: S=K=100, T=1, r=5%, sigma=20% -> 6.0903
        let am = crr(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, false, 2000, true, &mut buf);
        assert!((am - 6.0903).abs() < 2e-3, "{am}");
        // American call without dividends equals the European call.
        let amc = crr(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, true, 2000, true, &mut buf);
        assert!((amc - euro).abs() < 1e-10);
    }
}
