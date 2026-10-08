//! Input handling shared by every kernel: dtype coercion, scalar broadcasting,
//! null propagation, and a chunked parallel row runner.
//!
//! Nulls never reach the kernels: the output validity is the AND of the input
//! validity bitmaps, computed once with word-wide bitmap operations, and the
//! kernels run branch-free over the raw values (whatever sits in a null slot is
//! computed and then masked).

use std::sync::LazyLock;

use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use rayon::prelude::*;

/// Worker pool sized like Polars' own (`POLARS_MAX_THREADS`, else all cores), so
/// one setting caps the threads used by both Polars and these kernels.
static POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let n = std::env::var("POLARS_MAX_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("quantpolars-{i}"))
        .build()
        .expect("failed to build quantpolars thread pool")
});

/// Common output length of the inputs; each input must have that length or length 1
/// (scalars broadcast, including against an empty column).
pub fn output_len(inputs: &[Series]) -> PolarsResult<usize> {
    let n = if inputs.iter().any(|s| s.is_empty()) {
        0
    } else {
        inputs.iter().map(|s| s.len()).max().unwrap_or(0)
    };
    for s in inputs {
        if s.len() != n && s.len() != 1 {
            polars_bail!(ShapeMismatch:
                "input '{}' has length {}, expected {} or 1", s.name(), s.len(), n);
        }
    }
    Ok(n)
}

/// Owned, contiguous Float64 versions of the inputs (no copy when they already are).
pub struct F64Cols(Vec<Float64Chunked>);

impl F64Cols {
    pub fn new(inputs: &[Series]) -> PolarsResult<Self> {
        inputs
            .iter()
            .map(|s| {
                let s = s.cast(&DataType::Float64)?;
                Ok(s.f64()?.rechunk().into_owned())
            })
            .collect::<PolarsResult<Vec<_>>>()
            .map(Self)
    }

    pub fn views(&self) -> Vec<F64View<'_>> {
        self.0
            .iter()
            .map(|ca| {
                if ca.is_empty() {
                    return F64View { values: &[], validity: None, bcast: false };
                }
                let arr = ca.downcast_as_array();
                F64View {
                    values: arr.values().as_slice(),
                    validity: arr.validity().filter(|b| b.unset_bits() > 0),
                    bcast: ca.len() == 1,
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
pub struct F64View<'a> {
    values: &'a [f64],
    validity: Option<&'a Bitmap>,
    bcast: bool,
}

impl F64View<'_> {
    /// Raw value at row `i` (meaningless if the row is null; the output is masked).
    #[inline(always)]
    pub fn at(&self, i: usize) -> f64 {
        self.values[if self.bcast { 0 } else { i }]
    }

    /// Buffer to back [`F64View::block`] for a broadcast input (empty otherwise).
    pub fn fill_buffer(&self, len: usize) -> Vec<f64> {
        if self.bcast { vec![self.values.first().copied().unwrap_or(0.0); len] } else { Vec::new() }
    }

    /// Rows `base..base + len` as a contiguous slice; broadcast inputs read from `fill`.
    #[inline(always)]
    pub fn block<'b>(&'b self, base: usize, len: usize, fill: &'b [f64]) -> &'b [f64] {
        if self.bcast { &fill[..len] } else { &self.values[base..base + len] }
    }
}

/// Option side per row. Accepted encodings (strings trimmed, any case):
/// call = "c", "call", "true", "1", "1.0", "+1", Boolean true, numeric 1;
/// put  = "p", "put", "false", "-1", "-1.0", "0", Boolean false, numeric -1 or 0.
/// Anything else is null.
pub struct CallFlags {
    kind: FlagKind,
    validity: Option<Bitmap>,
    bcast: bool,
}

enum FlagKind {
    Bits(Bitmap),
    Parsed(Vec<bool>),
}

/// First-byte lookup for one-character flags: 1 = call, 0 = put, 2 = unrecognised.
static FLAG_LUT: [u8; 256] = {
    let mut lut = [2u8; 256];
    lut[b'c' as usize] = 1;
    lut[b'C' as usize] = 1;
    lut[b'1' as usize] = 1;
    lut[b'p' as usize] = 0;
    lut[b'P' as usize] = 0;
    lut[b'0' as usize] = 0;
    lut
};

/// Some(is_call) for a recognised token, None otherwise.
fn parse_token(raw: &str) -> Option<bool> {
    let t = raw.trim();
    if t.len() > 5 {
        return None;
    }
    match t.to_ascii_lowercase().as_str() {
        "c" | "call" | "true" | "1" | "1.0" | "+1" => Some(true),
        "p" | "put" | "false" | "-1" | "-1.0" | "0" => Some(false),
        _ => None,
    }
}

/// Validity from per-row "recognised" flags, AND-ed with the column's own nulls.
fn flag_validity(known: Option<Vec<bool>>, nulls: Option<&Bitmap>) -> Option<Bitmap> {
    let parsed: Option<Bitmap> = known.map(|k| k.into_iter().collect());
    match (parsed, nulls.filter(|b| b.unset_bits() > 0)) {
        (Some(a), Some(b)) => Some(&a & b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b.clone()),
        (None, None) => None,
    }
}

impl CallFlags {
    pub fn new(s: &Series) -> PolarsResult<Self> {
        let bcast = s.len() == 1;
        match s.dtype() {
            DataType::Boolean => {
                let ca = s.bool()?.rechunk();
                let arr = ca.downcast_as_array();
                Ok(Self {
                    kind: FlagKind::Bits(arr.values().clone()),
                    validity: flag_validity(None, arr.validity()),
                    bcast,
                })
            }
            DataType::String => {
                let ca = s.str()?.rechunk();
                let arr = ca.downcast_as_array();
                // 0 = put, 1 = call, 2 = unrecognised. One-character flags ("C"/"P",
                // the common case) come from a lookup on the first byte of the 4-byte
                // prefix stored inline in each view; longer strings are parsed.
                // Done in parallel: on large inputs this pass is otherwise serial.
                let views = arr.views().as_slice();
                let codes: Vec<u8> = POOL.install(|| {
                    views
                        .par_iter()
                        .enumerate()
                        .with_min_len(1 << 14)
                        .map(|(i, view)| {
                            if view.length == 1 {
                                FLAG_LUT[usize::from(view.prefix.to_ne_bytes()[0])]
                            } else {
                                parse_token(arr.value(i)).map_or(2, u8::from)
                            }
                        })
                        .collect()
                });
                let (flags, known) = POOL.install(|| {
                    let flags: Vec<bool> = codes.par_iter().map(|&c| c == 1).collect();
                    let any_unknown = codes.par_iter().any(|&c| c == 2);
                    (flags, any_unknown.then(|| codes.par_iter().map(|&c| c != 2).collect::<Vec<bool>>()))
                });
                Ok(Self { kind: FlagKind::Parsed(flags), validity: flag_validity(known, arr.validity()), bcast })
            }
            dt if dt.is_primitive_numeric() => {
                let ca = s.cast(&DataType::Float64)?;
                let ca = ca.f64()?.rechunk();
                let arr = ca.downcast_as_array();
                let known: Vec<bool> = arr.values().iter().map(|&x| x == 1.0 || x == -1.0 || x == 0.0).collect();
                let flags = arr.values().iter().map(|&x| x == 1.0).collect();
                let known = (!known.iter().all(|&k| k)).then_some(known);
                Ok(Self { kind: FlagKind::Parsed(flags), validity: flag_validity(known, arr.validity()), bcast })
            }
            DataType::Null => Ok(Self {
                kind: FlagKind::Parsed(vec![false; s.len()]),
                validity: Some(Bitmap::new_zeroed(s.len())),
                bcast,
            }),
            _ => Self::new(&s.cast(&DataType::String).map_err(|_| {
                polars_err!(InvalidOperation:
                    "option type must be Boolean, numeric (+1/-1) or String ('call'/'put'), got {}", s.dtype())
            })?),
        }
    }

    #[inline(always)]
    pub fn at(&self, i: usize) -> bool {
        let j = if self.bcast { 0 } else { i };
        match &self.kind {
            FlagKind::Bits(b) => b.get_bit(j),
            FlagKind::Parsed(v) => v[j],
        }
    }

    /// Rows `base..base + buf.len()` as a contiguous slice, materialised into
    /// `buf` unless already stored that way.
    #[inline]
    pub fn block<'b>(&'b self, base: usize, buf: &'b mut [bool]) -> &'b [bool] {
        match &self.kind {
            FlagKind::Parsed(v) if !self.bcast => &v[base..base + buf.len()],
            _ => {
                for (j, b) in buf.iter_mut().enumerate() {
                    *b = self.at(base + j);
                }
                buf
            }
        }
    }
}

/// Output validity: AND of every input's validity, broadcasting length-1 inputs.
pub fn combine_validity(n: usize, views: &[F64View], flags: Option<&CallFlags>) -> Option<Bitmap> {
    let parts = views
        .iter()
        .map(|v| (v.validity, v.bcast))
        .chain(flags.map(|f| (f.validity.as_ref(), f.bcast)));
    let mut acc: Option<Bitmap> = None;
    for (validity, bcast) in parts {
        let Some(v) = validity else { continue };
        if bcast {
            if !v.get_bit(0) {
                return Some(Bitmap::new_zeroed(n));
            }
            continue;
        }
        acc = Some(match acc {
            None => v.clone(),
            Some(a) => &a & v,
        });
    }
    acc
}

/// Evaluate `f` for rows `0..n` in parallel chunks of `chunk` rows. `init` builds
/// per-task scratch state (e.g. a binomial-tree buffer).
pub fn run_rows<S, I, F>(
    name: PlSmallStr, n: usize, chunk: usize, validity: Option<Bitmap>, init: I, f: F,
) -> Series
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> f64 + Sync + Send,
{
    let mut values = vec![0.0f64; n];
    POOL.install(|| {
        values
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each_init(&init, |state, (ci, vals)| {
                let base = ci * chunk;
                for (j, v) in vals.iter_mut().enumerate() {
                    *v = f(state, base + j);
                }
            })
    });
    Float64Chunked::from_vec_validity(name, values, validity).into_series()
}

/// Evaluate `f(state, base, out)` on consecutive row blocks `base..base + out.len()`
/// in parallel; for kernels that process a whole block at once.
pub fn run_chunks<S, I, F>(
    name: PlSmallStr, n: usize, chunk: usize, validity: Option<Bitmap>, init: I, f: F,
) -> Series
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [f64]) + Sync + Send,
{
    let mut values = vec![0.0f64; n];
    POOL.install(|| {
        values
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each_init(&init, |state, (ci, vals)| f(state, ci * chunk, vals))
    });
    Float64Chunked::from_vec_validity(name, values, validity).into_series()
}

/// Like [`run_chunks`] with `k` output columns: `f(state, base, outs)` fills
/// `outs[m][..]` for rows `base..`.
pub fn run_chunks_multi<S, I, F>(n: usize, k: usize, chunk: usize, init: I, f: F) -> Vec<Vec<f64>>
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [&mut [f64]]) + Sync + Send,
{
    let mut outs: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; n]).collect();
    // One set of disjoint `&mut` slices per block: safe parallel writes.
    let mut tasks: Vec<Vec<&mut [f64]>> = (0..n.div_ceil(chunk)).map(|_| Vec::with_capacity(k)).collect();
    for out in outs.iter_mut() {
        for (task, slice) in tasks.iter_mut().zip(out.chunks_mut(chunk)) {
            task.push(slice);
        }
    }
    POOL.install(|| {
        tasks
            .into_par_iter()
            .enumerate()
            .for_each_init(&init, |state, (ci, mut slices)| f(state, ci * chunk, &mut slices))
    });
    outs
}
