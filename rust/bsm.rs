//! Generalised Black-Scholes-Merton (continuous dividend yield `q`):
//! prices, analytic Greeks and implied volatility. Black-76 is the special
//! case `s = F`, `q = r`.
//!
//! Conventions: theta is per year (calendar time), vega per 1.00 of vol,
//! rho per 1.00 of rate. A row is "dead" when `t <= 0` or `sigma <= 0`
//! (at/after expiry or no volatility): its price is the discounted intrinsic
//! value of the forward, delta the exercise indicator, every other Greek 0.

use implied_vol::{DefaultSpecialFn, ImpliedBlackVolatility, SpecialFn};

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

// erf(x) = x T(x^2) / U(x^2) for |x| < 1 (Cephes ndtr.c, S. L. Moshier; the same
// central branch SciPy's `ndtr` uses). Coefficients kept exactly as published.
#[allow(clippy::excessive_precision)]
const ERF_T: [f64; 5] = [
    9.604_973_739_870_516_387_49E0, 9.002_601_972_038_426_892_17E1, 2.232_005_345_946_843_192_26E3,
    7.003_325_141_128_050_754_73E3, 5.559_230_130_103_949_627_68E4,
];
#[allow(clippy::excessive_precision)]
const ERF_U: [f64; 5] = [
    3.356_171_416_475_030_996_47E1, 5.213_579_497_801_526_797_95E2, 4.594_323_829_709_801_279_87E3,
    2.262_900_006_138_909_342_46E4, 4.926_739_426_086_359_210_86E4,
];

// erfc(a) = exp(-a^2) P(a) / Q(a) for 1 <= a < 8 (Cephes ndtr.c).
#[allow(clippy::excessive_precision)]
const ERFC_P: [f64; 9] = [
    2.461_969_814_735_305_125_24E-10, 5.641_895_648_310_688_219_77E-1, 7.463_210_564_422_699_126_87E0,
    4.863_719_709_856_813_666_14E1, 1.965_208_329_560_770_982_42E2, 5.264_451_949_954_773_586_31E2,
    9.345_285_271_719_576_075_40E2, 1.027_551_886_895_157_102_72E3, 5.575_353_353_693_993_275_26E2,
];
#[allow(clippy::excessive_precision)]
const ERFC_Q: [f64; 8] = [
    1.322_819_511_547_449_925_08E1, 8.670_721_408_859_897_423_29E1, 3.549_377_788_878_198_910_62E2,
    9.757_085_017_432_054_897_53E2, 1.823_909_166_879_097_362_89E3, 2.246_337_608_187_109_817_92E3,
    1.656_663_091_941_613_501_82E3, 5.575_353_408_177_276_755_46E2,
];

#[inline(always)]
fn erf_central(z: f64) -> f64 {
    let z2 = z * z;
    let num = (((ERF_T[0] * z2 + ERF_T[1]) * z2 + ERF_T[2]) * z2 + ERF_T[3]) * z2 + ERF_T[4];
    let den = ((((z2 + ERF_U[0]) * z2 + ERF_U[1]) * z2 + ERF_U[2]) * z2 + ERF_U[3]) * z2 + ERF_U[4];
    z * num / den
}

#[inline(always)]
fn erfc_mid(a: f64) -> f64 {
    let p = ERFC_P.iter().fold(0.0, |acc, &c| acc * a + c);
    let q = ERFC_Q.iter().fold(1.0, |acc, &c| acc * a + c);
    (-a * a).exp() * p / q
}

/// Scratch for [`ncdf_slice`]; holds at least as many slots as the slice.
pub struct CdfScratch {
    idx: Vec<u32>,
    x: Vec<f64>,
}

impl CdfScratch {
    pub fn new(len: usize) -> Self {
        Self { idx: vec![0; len], x: vec![0.0; len] }
    }
}

/// In-place standard normal CDF over a slice, built for throughput: the hot
/// loops have no data-dependent branches (which mispredict on mixed inputs).
/// 1. every value gets the central rational polynomial (straight-line code);
/// 2. values with |x| >= sqrt 2 are gathered branch-free and redone with the
///    single-exp Cephes erfc, the sign chosen by a select;
/// 3. the deep lower tail (x < -4 sqrt 2) gets Jäckel's full-precision CDF.
///
/// Agrees with [`ncdf`] to ~2e-15 relative (the accuracy of SciPy's `ndtr`).
pub fn ncdf_slice(xs: &mut [f64], scratch: &mut CdfScratch) {
    const H: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let (idx, tx) = (&mut scratch.idx[..xs.len()], &mut scratch.x[..xs.len()]);
    let mut m = 0;
    for (j, &x) in xs.iter().enumerate() {
        idx[m] = j as u32;
        tx[m] = x;
        m += usize::from((x * H).abs() >= 1.0);
    }
    for x in xs.iter_mut() {
        *x = 0.5 + 0.5 * erf_central(*x * H);
    }
    let mut d = 0;
    for q in 0..m {
        let x = tx[q];
        let a = (x * H).abs();
        // Clamp so +inf gives exp(-a^2) = 0 instead of 0 * inf/inf; erfc(30) underflows anyway.
        let y = 0.5 * erfc_mid(a.min(30.0));
        xs[idx[q] as usize] = if x > 0.0 { 1.0 - y } else { y };
        idx[d] = idx[q];
        tx[d] = x;
        d += usize::from(x < 0.0 && a >= 4.0);
    }
    for q in 0..d {
        xs[idx[q] as usize] = DefaultSpecialFn::norm_cdf(tx[q]);
    }
}

/// Standard normal CDF. Central region (|x| < sqrt 2): one rational polynomial,
/// no exp() (~2 ns, within ~1e-15 relative). Elsewhere Jäckel's implementation of
/// Cody's erfc, which keeps full relative precision deep into the tails.
#[inline(always)]
pub fn ncdf(x: f64) -> f64 {
    let z = x * std::f64::consts::FRAC_1_SQRT_2;
    if z.abs() < 1.0 {
        0.5 + 0.5 * erf_central(z)
    } else {
        DefaultSpecialFn::norm_cdf(x)
    }
}

#[inline(always)]
pub fn npdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// exp(-x * t), skipping the exp() call for the common x = 0 (e.g. no dividends).
#[inline(always)]
fn disc(x: f64, t: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (-x * t).exp() }
}

/// max(x, 0) that keeps NaN (f64::max would turn NaN into 0).
#[inline(always)]
fn pos(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

/// Alive rows take the closed forms; everything else is "dead" (see module docs).
#[inline(always)]
pub fn is_alive(t: f64, v: f64) -> bool {
    t > 0.0 && v > 0.0
}

/// European price.
#[inline]
pub fn price(s: f64, k: f64, t: f64, r: f64, q: f64, v: f64, call: bool) -> f64 {
    let sign = if call { 1.0 } else { -1.0 };
    let dr = disc(r, t);
    if !is_alive(t, v) {
        let fwd = s * ((r - q) * t).exp();
        return dr * pos(sign * (fwd - k));
    }
    let vst = v * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / vst;
    let d2 = d1 - vst;
    sign * (s * disc(q, t) * ncdf(sign * d1) - k * dr * ncdf(sign * d2))
}

/// Greek names in output order; indices into [`Need`] and the kernels' arrays.
pub const GREEKS: [&str; 9] = [
    "delta", "gamma", "vega", "theta", "rho", "vanna", "vomma", "charm", "dual_delta",
];
pub const N_GREEKS: usize = GREEKS.len();

/// Which Greeks a caller needs (by index into [`GREEKS`]), so unused work is skipped.
#[derive(Clone, Copy)]
pub struct Need([bool; N_GREEKS]);

impl Need {
    pub fn from_indices(idx: &[usize]) -> Self {
        let mut need = [false; N_GREEKS];
        for &i in idx {
            need[i] = true;
        }
        Need(need)
    }
    #[inline(always)]
    fn any(&self, idx: &[usize]) -> bool {
        idx.iter().any(|&i| self.0[i])
    }
    #[inline(always)]
    fn has(&self, i: usize) -> bool {
        self.0[i]
    }
    /// N(+-d1): delta, theta, charm.  N(+-d2): theta, rho, dual_delta.
    /// density: gamma, vega, theta, vanna, vomma, charm.
    fn n1(&self) -> bool {
        self.any(&[0, 3, 7])
    }
    fn n2(&self) -> bool {
        self.any(&[3, 4, 8])
    }
    fn pdf(&self) -> bool {
        self.any(&[1, 2, 3, 5, 6, 7])
    }
}

/// Greeks of a dead row: delta = exercise indicator (discounted by the yield).
#[inline]
fn dead_greeks(s: f64, k: f64, t: f64, r: f64, q: f64, sign: f64) -> [f64; N_GREEKS] {
    let fwd = s * ((r - q) * t).exp();
    let mut g = [0.0; N_GREEKS];
    g[0] = if sign * (fwd - k) > 0.0 { sign * disc(q, t) } else { 0.0 };
    g
}

/// The closed forms for an alive row given its shared intermediates.
/// `d1s`, `n1`, `n2` are sign*d1, N(sign*d1), N(sign*d2); unused ones may be 0.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn alive_greeks(
    s: f64, k: f64, t: f64, r: f64, q: f64, v: f64, sign: f64,
    d1s: f64, n1: f64, n2: f64, need: Need,
) -> [f64; N_GREEKS] {
    let sqrt_t = t.sqrt();
    let vst = v * sqrt_t;
    let dq = disc(q, t);
    let dr = if need.n2() { disc(r, t) } else { 0.0 };
    let d1 = sign * d1s;
    let d2 = d1 - vst;
    let pdf = if need.pdf() { npdf(d1) } else { 0.0 };
    let mut g = [0.0; N_GREEKS];
    if need.has(0) {
        g[0] = sign * dq * n1;
    }
    if need.has(1) {
        g[1] = dq * pdf / (s * vst);
    }
    if need.has(2) {
        g[2] = s * dq * pdf * sqrt_t;
    }
    if need.has(3) {
        g[3] = -s * dq * pdf * v / (2.0 * sqrt_t) - sign * (r * k * dr * n2 - q * s * dq * n1);
    }
    if need.has(4) {
        g[4] = sign * k * t * dr * n2;
    }
    if need.has(5) {
        g[5] = -dq * pdf * d2 / v;
    }
    if need.has(6) {
        g[6] = s * dq * pdf * sqrt_t * d1 * d2 / v;
    }
    if need.has(7) {
        g[7] = sign * q * dq * n1 - dq * pdf * (2.0 * (r - q) * t - d2 * vst) / (2.0 * t * vst);
    }
    if need.has(8) {
        g[8] = -sign * dr * n2;
    }
    g
}

/// Greeks for one row; entries not in `need` are 0. Scalar reference path,
/// also used for dead rows by the chunked kernel.
#[allow(clippy::too_many_arguments)]
pub fn greeks_sel(s: f64, k: f64, t: f64, r: f64, q: f64, v: f64, call: bool, need: Need) -> [f64; N_GREEKS] {
    let sign = if call { 1.0 } else { -1.0 };
    if !is_alive(t, v) {
        return dead_greeks(s, k, t, r, q, sign);
    }
    let vst = v * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / vst;
    let n1 = if need.n1() { ncdf(sign * d1) } else { 0.0 };
    let n2 = if need.n2() { ncdf(sign * (d1 - vst)) } else { 0.0 };
    alive_greeks(s, k, t, r, q, v, sign, sign * d1, n1, n2, need)
}

/// One block of rows for the chunked kernels; every slice has the same length.
#[derive(Clone, Copy)]
pub struct Block<'a> {
    pub s: &'a [f64],
    pub k: &'a [f64],
    pub t: &'a [f64],
    pub r: &'a [f64],
    pub q: &'a [f64],
    pub v: &'a [f64],
    pub call: &'a [bool],
}

impl Block<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.s.len()
    }

    #[inline(always)]
    fn row(&self, j: usize) -> (f64, f64, f64, f64, f64, f64, bool) {
        (self.s[j], self.k[j], self.t[j], self.r[j], self.q[j], self.v[j], self.call[j])
    }

    /// Re-slice everything to exactly `len()` so the compiler can drop bounds checks.
    #[inline(always)]
    fn tight(self) -> Self {
        let n = self.len();
        Block {
            s: &self.s[..n], k: &self.k[..n], t: &self.t[..n], r: &self.r[..n],
            q: &self.q[..n], v: &self.v[..n], call: &self.call[..n],
        }
    }
}

/// Per-task buffers for the chunked kernels.
pub struct ChunkScratch {
    d1: Vec<f64>,
    n1: Vec<f64>,
    n2: Vec<f64>,
    cdf: CdfScratch,
}

impl ChunkScratch {
    pub fn new(len: usize) -> Self {
        Self { d1: vec![0.0; len], n1: vec![0.0; len], n2: vec![0.0; len], cdf: CdfScratch::new(len) }
    }
}

/// sign*d1 and sign*d2 (sign = +1 call, -1 put). Garbage on dead rows, which
/// the combine pass replaces.
#[inline(always)]
fn signed_d(row: (f64, f64, f64, f64, f64, f64, bool)) -> (f64, f64) {
    let (s, k, t, r, q, v, call) = row;
    let vst = v * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / vst;
    let sign = if call { 1.0 } else { -1.0 };
    (sign * d1, sign * (d1 - vst))
}

/// Prices for one block of rows into `out` (same length as the block).
/// Three passes (d1/d2, batched CDFs, combine) keep the CDF's branches off the
/// critical path; dead rows fall back to the scalar [`price`].
pub fn price_chunk(out: &mut [f64], sc: &mut ChunkScratch, b: Block) {
    let b = b.tight();
    let len = b.len();
    let out = &mut out[..len];
    let (n1, n2) = (&mut sc.n1[..len], &mut sc.n2[..len]);
    for j in 0..len {
        (n1[j], n2[j]) = signed_d(b.row(j));
    }
    ncdf_slice(n1, &mut sc.cdf);
    ncdf_slice(n2, &mut sc.cdf);
    for (j, o) in out.iter_mut().enumerate() {
        let (s, k, t, r, q, v, call) = b.row(j);
        *o = if is_alive(t, v) {
            let sign = if call { 1.0 } else { -1.0 };
            sign * (s * disc(q, t) * n1[j] - k * disc(r, t) * n2[j])
        } else {
            price(s, k, t, r, q, v, call)
        };
    }
}

/// Selected Greeks for a block: `outs[m]` receives Greek `which[m]` (index into
/// [`GREEKS`]). Same three-pass layout as [`price_chunk`].
pub fn greeks_chunk(outs: &mut [&mut [f64]], which: &[usize], need: Need, sc: &mut ChunkScratch, b: Block) {
    let b = b.tight();
    let len = b.len();
    let (d1, n1, n2) = (&mut sc.d1[..len], &mut sc.n1[..len], &mut sc.n2[..len]);
    for j in 0..len {
        (d1[j], n2[j]) = signed_d(b.row(j));
    }
    if need.n1() {
        n1.copy_from_slice(d1);
        ncdf_slice(n1, &mut sc.cdf);
    }
    if need.n2() {
        ncdf_slice(n2, &mut sc.cdf);
    }
    // Delta alone (the common request): a tight loop with no per-Greek dispatch.
    if let ([out], [0]) = (&mut *outs, which) {
        let out = &mut out[..len];
        for (j, o) in out.iter_mut().enumerate() {
            let (s, k, t, r, q, v, call) = b.row(j);
            *o = if is_alive(t, v) {
                let sign = if call { 1.0 } else { -1.0 };
                sign * disc(q, t) * n1[j]
            } else {
                greeks_sel(s, k, t, r, q, v, call, need)[0]
            };
        }
        return;
    }
    for j in 0..len {
        let (s, k, t, r, q, v, call) = b.row(j);
        let sign = if call { 1.0 } else { -1.0 };
        let g = if is_alive(t, v) {
            alive_greeks(s, k, t, r, q, v, sign, d1[j], n1[j], n2[j], need)
        } else {
            dead_greeks(s, k, t, r, q, sign)
        };
        for (o, &w) in outs.iter_mut().zip(which) {
            o[j] = g[w];
        }
    }
}

/// Implied volatility from a (discounted) market price via Jäckel's
/// "Let's Be Rational". NaN when no volatility reprices the quote: the price
/// is outside the no-arbitrage band (at or below intrinsic, at or above the
/// forward bound), `t <= 0`, or inputs are invalid.
#[inline]
pub fn implied_vol(price: f64, s: f64, k: f64, t: f64, r: f64, q: f64, call: bool) -> f64 {
    if !(price > 0.0 && s > 0.0 && k > 0.0 && t > 0.0 && price.is_finite()) {
        return f64::NAN;
    }
    let forward = s * ((r - q) * t).exp();
    let undiscounted = price * (r * t).exp();
    let iv = ImpliedBlackVolatility::builder()
        .option_price(undiscounted)
        .forward(forward)
        .strike(k)
        .expiry(t)
        .is_call(call)
        .build()
        .and_then(|iv| iv.calculate::<DefaultSpecialFn>())
        .unwrap_or(f64::NAN);
    // Zero (price exactly intrinsic) and infinite vols do not reprice a quote.
    if iv > 0.0 && iv.is_finite() { iv } else { f64::NAN }
}

/// Special functions exposed as expressions.
#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Special {
    Erf,
    Erfc,
    Erfcx,
    NormSf,
    NormPpf,
}

#[inline]
pub fn special(f: Special, x: f64) -> f64 {
    match f {
        Special::Erf => DefaultSpecialFn::erf(x),
        Special::Erfc => DefaultSpecialFn::erfc(x),
        Special::Erfcx => DefaultSpecialFn::erfcx(x),
        Special::NormSf => ncdf(-x),
        Special::NormPpf => {
            if x > 0.0 && x < 1.0 { DefaultSpecialFn::inverse_norm_cdf(x) } else { f64::NAN }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [usize; N_GREEKS] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    fn all() -> Need {
        Need::from_indices(&ALL)
    }

    #[test]
    fn ncdf_matches_reference_everywhere() {
        // Central polynomial agrees with Cody's erfc across its whole range.
        for i in 0..=200_000 {
            let x = -1.4143 + 2.8286 * f64::from(i) / 200_000.0;
            let (a, b) = (ncdf(x), DefaultSpecialFn::norm_cdf(x));
            assert!(((a - b) / b).abs() < 5e-15, "x={x}: {a} vs {b}");
        }
        assert_eq!(ncdf(0.0), 0.5);
        assert!((ncdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-16);
        assert!((ncdf(-30.0) / 4.906_713_927_148_187e-198 - 1.0).abs() < 1e-14);
        assert!(ncdf(f64::NAN).is_nan());
    }

    #[test]
    fn slice_cdf_matches_scalar() {
        let mut xs: Vec<f64> = (0..=400_000).map(|i| -40.0 + 52.0 * f64::from(i) / 400_000.0).collect();
        xs.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0]);
        let mut ys = xs.clone();
        ncdf_slice(&mut ys, &mut CdfScratch::new(xs.len()));
        for (&x, &y) in xs.iter().zip(&ys) {
            let want = DefaultSpecialFn::norm_cdf(x);
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert!(y == want || ((y - want) / want).abs() < 5e-15, "x={x}: {y} vs {want}");
            }
        }
    }

    #[test]
    fn chunk_kernels_match_scalar() {
        // Mixed calls/puts and dead rows (t <= 0, sigma <= 0, NaN) included.
        let rows: Vec<(f64, f64, f64, f64, f64, f64, bool)> = (0..5000)
            .map(|i| {
                let f = f64::from(i);
                let t = match i % 97 { 0 => 0.0, 1 => -0.1, _ => 0.001 + (f * 0.37) % 2.0 };
                let v = match i % 89 { 0 => 0.0, 1 => f64::NAN, 2 => -0.2, _ => 0.05 + (f * 0.13) % 0.9 };
                (100.0, 60.0 + (f * 7.3) % 90.0, t, 0.03, 0.01 * f64::from(i % 3), v, i % 2 == 0)
            })
            .collect();
        let n = rows.len();
        let col = |f: fn(&(f64, f64, f64, f64, f64, f64, bool)) -> f64| rows.iter().map(f).collect::<Vec<f64>>();
        let (s, k, t, r, q, v) = (col(|x| x.0), col(|x| x.1), col(|x| x.2), col(|x| x.3), col(|x| x.4), col(|x| x.5));
        let call: Vec<bool> = rows.iter().map(|x| x.6).collect();
        let block = Block { s: &s, k: &k, t: &t, r: &r, q: &q, v: &v, call: &call };
        let mut sc = ChunkScratch::new(n);
        let mut out = vec![0.0; n];
        price_chunk(&mut out, &mut sc, block);
        let mut cols = vec![vec![0.0; n]; N_GREEKS];
        {
            let mut outs: Vec<&mut [f64]> = cols.iter_mut().map(|c| c.as_mut_slice()).collect();
            greeks_chunk(&mut outs, &ALL, all(), &mut sc, block);
        }
        let mut delta_only = vec![0.0; n];
        greeks_chunk(&mut [delta_only.as_mut_slice()], &[0], Need::from_indices(&[0]), &mut sc, block);
        // The delta-only fast path must agree exactly with the general path.
        assert!(delta_only.iter().zip(&cols[0]).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
        let close = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        for (j, &(s, k, t, r, q, v, c)) in rows.iter().enumerate() {
            assert!(close(out[j], price(s, k, t, r, q, v, c)), "price row {j}: {} vs {}", out[j], price(s, k, t, r, q, v, c));
            let g = greeks_sel(s, k, t, r, q, v, c, all());
            for m in 0..N_GREEKS {
                assert!(close(cols[m][j], g[m]), "{} row {j}: {} vs {}", GREEKS[m], cols[m][j], g[m]);
            }
        }
    }

    #[test]
    fn hull_reference_values() {
        let c = price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, true);
        let p = price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, false);
        assert!((c - 10.450_583_572_185_565).abs() < 1e-12);
        assert!((p - 5.573_526_022_256_971).abs() < 1e-12);
    }

    #[test]
    fn dead_rows_are_intrinsic() {
        assert_eq!(price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, true), 10.0);
        assert_eq!(price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, false), 0.0);
        let zero_vol = price(100.0, 90.0, 1.0, 0.05, 0.0, 0.0, true);
        assert!((zero_vol - (100.0 - 90.0 * (-0.05f64).exp())).abs() < 1e-12);
        let g = greeks_sel(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, true, all());
        assert_eq!(g[0], 1.0);
        assert!(g[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn iv_round_trip() {
        for &(k, t, v, call) in &[(80.0, 0.1, 0.15, true), (120.0, 2.0, 0.6, false), (100.0, 0.5, 0.3, true)] {
            let px = price(100.0, k, t, 0.03, 0.01, v, call);
            let iv = implied_vol(px, 100.0, k, t, 0.03, 0.01, call);
            // Deep ITM short-dated options carry little time value, so the
            // recoverable precision is limited by the price itself.
            assert!((iv - v).abs() < 1e-8, "{iv} vs {v}");
        }
        assert!(implied_vol(0.0, 100.0, 110.0, 0.5, 0.05, 0.0, true).is_nan());
        assert!(implied_vol(5.0, 100.0, 90.0, 0.5, 0.05, 0.0, true).is_nan()); // below intrinsic
    }

    #[test]
    fn greeks_match_finite_differences() {
        let (s, k, t, r, q, v) = (105.0, 100.0, 0.75, 0.04, 0.015, 0.25);
        let h = 1e-4;
        for call in [true, false] {
            let g = greeks_sel(s, k, t, r, q, v, call, all());
            let p = |s: f64, k: f64, t: f64, r: f64, v: f64| price(s, k, t, r, q, v, call);
            let d = |s: f64, t: f64, v: f64| greeks_sel(s, k, t, r, q, v, call, all())[0];
            let vega = |v: f64| greeks_sel(s, k, t, r, q, v, call, all())[2];
            let fd = [
                (p(s + h, k, t, r, v) - p(s - h, k, t, r, v)) / (2.0 * h),
                (d(s + h, t, v) - d(s - h, t, v)) / (2.0 * h),
                (p(s, k, t, r, v + h) - p(s, k, t, r, v - h)) / (2.0 * h),
                -(p(s, k, t + h, r, v) - p(s, k, t - h, r, v)) / (2.0 * h),
                (p(s, k, t, r + h, v) - p(s, k, t, r - h, v)) / (2.0 * h),
                (d(s, t, v + h) - d(s, t, v - h)) / (2.0 * h),
                (vega(v + h) - vega(v - h)) / (2.0 * h),
                -(d(s, t + h, v) - d(s, t - h, v)) / (2.0 * h),
                (p(s, k + h, t, r, v) - p(s, k - h, t, r, v)) / (2.0 * h),
            ];
            for (m, (a, b)) in g.iter().zip(fd).enumerate() {
                assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()), "{}: {a} vs {b}", GREEKS[m]);
            }
        }
    }
}
