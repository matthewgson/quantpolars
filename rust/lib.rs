//! Native Polars expression plugins for quantpolars.
//!
//! Every function here is registered from Python with
//! `polars.plugins.register_plugin_function`, so it runs inside the Polars
//! engine (lazy, streaming, group_by contexts) without touching the GIL.

mod american;
mod args;
mod bsm;
mod stats;

use polars::prelude::*;
use pyo3::prelude::*;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use args::{combine_validity, output_len, run_chunks, run_chunks_multi, run_rows, CallFlags, F64Cols};

// No `PolarsAllocator` here: pyo3-polars 0.28 looks up a capsule path that recent
// Polars no longer exports, and its fallback re-enters Python during module init
// (segfault on import). Buffers cross the boundary via the Arrow C Data Interface,
// whose release callbacks free memory with the allocator that produced it.

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))
}

// Rows per parallel task, tuned to per-row cost. BLOCK keeps the chunked
// kernels' scratch buffers (4 x 32 KiB) inside L1/L2.
const BLOCK: usize = 4096;
const CHEAP: usize = 16_384;
const MEDIUM: usize = 1024;

/// Per-task state for the chunked kernels: scratch plus contiguous backing for
/// broadcast (scalar) inputs and materialised call flags.
struct Task {
    sc: bsm::ChunkScratch,
    fills: [Vec<f64>; 6],
    flags: Vec<bool>,
}

impl Task {
    fn new(views: &[args::F64View; 6]) -> Self {
        Task {
            sc: bsm::ChunkScratch::new(BLOCK),
            fills: std::array::from_fn(|m| views[m].fill_buffer(BLOCK)),
            flags: vec![false; BLOCK],
        }
    }

    /// Rows `base..base + len` of the six float inputs and the flags.
    fn block<'b>(
        &'b mut self, views: &'b [args::F64View; 6], cp: &'b CallFlags, base: usize, len: usize,
    ) -> (&'b mut bsm::ChunkScratch, bsm::Block<'b>) {
        let Task { sc, fills, flags } = self;
        let f = |m: usize| views[m].block(base, len, &fills[m]);
        let block = bsm::Block {
            s: f(0), k: f(1), t: f(2), r: f(3), q: f(4), v: f(5),
            call: cp.block(base, &mut flags[..len]),
        };
        (sc, block)
    }
}

/// Inputs: s, k, t, r, q, sigma (Float64-castable), is_call (Boolean or String).
/// Binds the six float views, the call flags, the row count and the output validity.
macro_rules! option_inputs {
    ($inputs:expr => $n:ident, $valid:ident, $cols:ident, $cp:ident, [$($x:ident),+]) => {
        let $n = output_len($inputs)?;
        let $cols = F64Cols::new(&$inputs[..6])?;
        let views = $cols.views();
        let [$($x),+] = [views[0], views[1], views[2], views[3], views[4], views[5]];
        let $cp = CallFlags::new(&$inputs[6])?;
        let $valid = combine_validity($n, &views, Some(&$cp));
    };
}

#[polars_expr(output_type = Float64)]
fn bs_price(inputs: &[Series]) -> PolarsResult<Series> {
    option_inputs!(inputs => n, valid, _cols, cp, [s, k, t, r, q, v]);
    let views = [s, k, t, r, q, v];
    Ok(run_chunks(inputs[0].name().clone(), n, BLOCK, valid, || Task::new(&views), |task, base, out| {
        let (sc, block) = task.block(&views, &cp, base, out.len());
        bsm::price_chunk(out, sc, block)
    }))
}

#[derive(Deserialize)]
struct GreeksKwargs {
    greeks: Vec<String>,
}

impl GreeksKwargs {
    /// Indices into [`bsm::GREEKS`], in the requested order.
    fn indices(&self) -> PolarsResult<Vec<usize>> {
        self.greeks
            .iter()
            .map(|g| {
                bsm::GREEKS.iter().position(|n| n == g).ok_or_else(
                    || polars_err!(InvalidOperation: "unknown greek '{}'; choose from {:?}", g, bsm::GREEKS),
                )
            })
            .collect()
    }
}

fn greeks_field(fields: &[Field], kwargs: GreeksKwargs) -> PolarsResult<Field> {
    let inner = kwargs
        .indices()?
        .into_iter()
        .map(|i| Field::new(bsm::GREEKS[i].into(), DataType::Float64))
        .collect();
    Ok(Field::new(fields[0].name().clone(), DataType::Struct(inner)))
}

#[polars_expr(output_type_func_with_kwargs = greeks_field)]
fn bs_greeks(inputs: &[Series], kwargs: GreeksKwargs) -> PolarsResult<Series> {
    option_inputs!(inputs => n, valid, _cols, cp, [s, k, t, r, q, v]);
    let idx = kwargs.indices()?;
    let need = bsm::Need::from_indices(&idx);
    let name = inputs[0].name().clone();
    let views = [s, k, t, r, q, v];
    let outs = run_chunks_multi(n, idx.len(), BLOCK, || Task::new(&views), |task, base, outs| {
        let len = outs.first().map_or(0, |o| o.len());
        let (sc, block) = task.block(&views, &cp, base, len);
        bsm::greeks_chunk(outs, &idx, need, sc, block)
    });
    let fields: Vec<Series> = idx
        .iter()
        .zip(outs)
        .map(|(&j, vals)| {
            Float64Chunked::from_vec_validity(bsm::GREEKS[j].into(), vals, valid.clone()).into_series()
        })
        .collect();
    Ok(StructChunked::from_series(name, n, fields.iter())?.into_series())
}

/// NaN results become nulls (on top of the inputs' nulls).
fn nan_to_null(out: Series) -> PolarsResult<Series> {
    let ca = out.f64()?;
    let arr = ca.downcast_as_array();
    if !arr.values().iter().any(|x| x.is_nan()) {
        return Ok(out);
    }
    let not_nan: Bitmap = arr.values().iter().map(|x| !x.is_nan()).collect();
    let validity = match arr.validity() {
        Some(v) => v & &not_nan,
        None => not_nan,
    };
    let values = arr.values().as_slice().to_vec();
    Ok(Float64Chunked::from_vec_validity(ca.name().clone(), values, Some(validity)).into_series())
}

/// Inputs: price, s, k, t, r, q, is_call. Null where no volatility reprices the quote.
#[polars_expr(output_type = Float64)]
fn bs_iv(inputs: &[Series]) -> PolarsResult<Series> {
    option_inputs!(inputs => n, valid, _cols, cp, [px, s, k, t, r, q]);
    nan_to_null(run_rows(inputs[0].name().clone(), n, MEDIUM, valid, || (), |_, i| {
        bsm::implied_vol(px.at(i), s.at(i), k.at(i), t.at(i), r.at(i), q.at(i), cp.at(i))
    }))
}

#[derive(Deserialize)]
struct CrrKwargs {
    steps: usize,
    american: bool,
}

#[polars_expr(output_type = Float64)]
fn crr_price(inputs: &[Series], kwargs: CrrKwargs) -> PolarsResult<Series> {
    option_inputs!(inputs => n, valid, _cols, cp, [s, k, t, r, q, v]);
    let steps = kwargs.steps;
    // Aim for roughly 1M tree-node updates per task.
    let chunk = (2_000_000 / (steps * steps + 1)).clamp(1, CHEAP);
    Ok(run_rows(
        inputs[0].name().clone(),
        n,
        chunk,
        valid,
        || Vec::with_capacity(steps + 1),
        |buf, i| {
            let (s, k, t, r, q, v, c) = (s.at(i), k.at(i), t.at(i), r.at(i), q.at(i), v.at(i), cp.at(i));
            american::crr(s, k, t, r, q, v, c, steps, kwargs.american, buf)
        },
    ))
}

#[polars_expr(output_type = Float64)]
fn baw_price(inputs: &[Series]) -> PolarsResult<Series> {
    option_inputs!(inputs => n, valid, _cols, cp, [s, k, t, r, q, v]);
    Ok(run_rows(inputs[0].name().clone(), n, MEDIUM, valid, || (), |_, i| {
        american::baw(s.at(i), k.at(i), t.at(i), r.at(i), q.at(i), v.at(i), cp.at(i))
    }))
}

#[derive(Deserialize)]
struct PValueKwargs {
    alternative: stats::Alternative,
}

/// Inputs: t statistic, degrees of freedom.
#[polars_expr(output_type = Float64)]
fn t_pvalue(inputs: &[Series], kwargs: PValueKwargs) -> PolarsResult<Series> {
    let n = output_len(inputs)?;
    let cols = F64Cols::new(inputs)?;
    let views = cols.views();
    let (t, df) = (views[0], views[1]);
    let valid = combine_validity(n, &views, None);
    Ok(run_rows(inputs[0].name().clone(), n, MEDIUM, valid, || (), |_, i| {
        stats::t_pvalue(t.at(i), df.at(i), kwargs.alternative)
    }))
}

#[polars_expr(output_type = Float64)]
fn norm_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    let n = output_len(inputs)?;
    let cols = F64Cols::new(inputs)?;
    let views = cols.views();
    let x = views[0];
    let valid = combine_validity(n, &views, None);
    Ok(run_chunks(
        inputs[0].name().clone(),
        n,
        BLOCK,
        valid,
        || bsm::CdfScratch::new(BLOCK),
        |sc, base, out| {
            for (j, o) in out.iter_mut().enumerate() {
                *o = x.at(base + j);
            }
            bsm::ncdf_slice(out, sc);
        },
    ))
}

#[derive(Deserialize)]
struct SpecialKwargs {
    func: bsm::Special,
}

/// erf, erfc, erfcx, norm_sf, norm_ppf (null outside (0, 1)).
#[polars_expr(output_type = Float64)]
fn special(inputs: &[Series], kwargs: SpecialKwargs) -> PolarsResult<Series> {
    let n = output_len(inputs)?;
    let cols = F64Cols::new(inputs)?;
    let views = cols.views();
    let x = views[0];
    let valid = combine_validity(n, &views, None);
    let out = run_rows(inputs[0].name().clone(), n, CHEAP, valid, || (), |_, i| bsm::special(kwargs.func, x.at(i)));
    match kwargs.func {
        bsm::Special::NormPpf => nan_to_null(out),
        _ => Ok(out),
    }
}
