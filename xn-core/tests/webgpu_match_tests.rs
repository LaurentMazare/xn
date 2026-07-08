#![cfg(feature = "webgpu")]
//! Extensive cross-backend equivalence tests for the WebGPU backend.
//!
//! Every op is run through a backend-generic `run_*` helper, so the exact same
//! computation executes on the CPU (the reference oracle) and on WebGPU from
//! bit-identical inputs; the results are compared up to a numerical tolerance.
//! When the crate is also built with `--features cuda`, each check additionally
//! compares WebGPU against CUDA directly (GPU-vs-GPU), so the backend is pinned
//! against both a scalar reference and another GPU implementation.
//!
//! Inputs are pseudo-random (a small deterministic xorshift PRNG) across many
//! shapes, including sizes that straddle the 256-lane workgroup boundary and
//! matmul dims that are not tile-aligned, to exercise the tail/edge paths.

use xn::{Backend, CPU, Result, Tensor, WithDTypeF};

// A single shared WebGPU device (see webgpu_tests.rs for why it is shared).
fn wg() -> xn::webgpu_backend::Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<xn::webgpu_backend::Device> = OnceLock::new();
    DEVICE.get_or_init(|| xn::webgpu_backend::Device::new(0).expect("init webgpu device")).clone()
}

#[cfg(feature = "cuda")]
fn cuda() -> xn::cuda_backend::Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<xn::cuda_backend::Device> = OnceLock::new();
    DEVICE.get_or_init(|| xn::cuda_backend::Device::new(0).expect("init cuda device")).clone()
}

// -----------------------------------------------------------------------------
// Deterministic input generation + comparison
// -----------------------------------------------------------------------------

/// xorshift64* PRNG -> f32 in [lo, hi).
fn rnd(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x1234_5678);
    if s == 0 {
        s = 0xDEAD_BEEF;
    }
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        })
        .collect()
}

/// Compare two f32 slices with combined relative + absolute tolerance; report
/// the worst offender on failure.
fn cmp(label: &str, reference: &[f32], got: &[f32], rtol: f32, atol: f32) {
    assert_eq!(reference.len(), got.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0;
    for (i, (&r, &g)) in reference.iter().zip(got).enumerate() {
        if r == g || (r.is_nan() && g.is_nan()) {
            continue;
        }
        let over = (r - g).abs() - (atol + rtol * r.abs().max(g.abs()));
        if over > worst {
            worst = over;
            worst_i = i;
        }
    }
    if worst > 0.0 {
        let (r, g) = (reference[worst_i], got[worst_i]);
        panic!(
            "{label}: mismatch at {worst_i}: ref={r} got={g} |d|={} (over tol by {worst})",
            (r - g).abs()
        );
    }
}

/// Run a backend-generic runner `$run` (a generic `fn(&B, ...) -> Vec<f32>`) on
/// the CPU oracle and WebGPU (and CUDA when built in), and assert equivalence.
/// The device is prepended to `$args`; because `$run` is a generic fn path it
/// monomorphizes per backend (a closure could not, since it has one param type).
macro_rules! verify {
    ($label:expr, $rtol:expr, $atol:expr, $run:path $(, $arg:expr)* $(,)?) => {{
        let reference = $run(&CPU $(, $arg)*);
        let got_wg = $run(&wg() $(, $arg)*);
        cmp($label, &reference, &got_wg, $rtol, $atol);
        #[cfg(feature = "cuda")]
        {
            // GPU-vs-GPU: two independent f32 implementations, so allow a
            // slightly looser floor than the CPU-oracle tolerance.
            let got_cu = $run(&cuda() $(, $arg)*);
            cmp(
                concat!($label, " [webgpu vs cuda]"),
                &got_cu,
                &got_wg,
                { let r: f32 = $rtol; r.max(3e-3) },
                { let a: f32 = $atol; a.max(1e-4) },
            );
        }
    }};
}

fn t<B: Backend>(dev: &B, data: &[f32], shape: &[usize]) -> Tensor<f32, B> {
    Tensor::from_vec(data.to_vec(), shape.to_vec(), dev).unwrap()
}

// -----------------------------------------------------------------------------
// Unary
// -----------------------------------------------------------------------------

macro_rules! def_unary {
    ($fname:ident, $m:ident) => {
        fn $fname<B: Backend>(dev: &B, d: &[f32], sh: &[usize]) -> Vec<f32> {
            t(dev, d, sh).$m().unwrap().to_vec().unwrap()
        }
    };
}
def_unary!(run_relu, relu);
def_unary!(run_silu, silu);
def_unary!(run_sqr, sqr);
def_unary!(run_sqrt, sqrt);
def_unary!(run_exp, exp);
def_unary!(run_log, log);
def_unary!(run_abs, abs);
def_unary!(run_neg, neg);
def_unary!(run_gelu, gelu_erf);
def_unary!(run_tanh, tanh);
def_unary!(run_sigmoid, sigmoid);
def_unary!(run_cos, cos);
def_unary!(run_sin, sin);
def_unary!(run_rsqrt, rsqrt);

const SHAPES_1D: &[&[usize]] =
    &[&[1], &[7], &[64], &[255], &[256], &[257], &[1000], &[2, 3, 4], &[3, 5, 7]];

#[test]
fn unary_ops() -> Result<()> {
    for (i, &sh) in SHAPES_1D.iter().enumerate() {
        let n: usize = sh.iter().product();
        let seed = i as u64 + 1;
        let d = rnd(seed, n, -4.0, 4.0);
        verify!("relu", 1e-6, 1e-6, run_relu, &d, sh);
        verify!("silu", 2e-6, 1e-6, run_silu, &d, sh);
        verify!("sqr", 1e-6, 1e-6, run_sqr, &d, sh);
        verify!("abs", 1e-6, 1e-6, run_abs, &d, sh);
        verify!("neg", 0.0, 0.0, run_neg, &d, sh);
        verify!("gelu_erf", 1e-5, 1e-5, run_gelu, &d, sh);
        verify!("tanh", 2e-6, 1e-6, run_tanh, &d, sh);
        verify!("sigmoid", 2e-6, 1e-6, run_sigmoid, &d, sh);
        verify!("cos", 2e-6, 1e-6, run_cos, &d, sh);
        verify!("sin", 2e-6, 1e-6, run_sin, &d, sh);
        let de = rnd(seed + 100, n, -3.0, 3.0);
        verify!("exp", 1e-5, 1e-6, run_exp, &de, sh);
        let dp = rnd(seed + 200, n, 0.05, 5.0);
        verify!("sqrt", 1e-6, 1e-6, run_sqrt, &dp, sh);
        verify!("log", 2e-6, 1e-6, run_log, &dp, sh);
        verify!("rsqrt", 2e-6, 1e-6, run_rsqrt, &dp, sh);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Binary (contiguous) + scale
// -----------------------------------------------------------------------------

fn run_bin<B: Backend>(dev: &B, a: &[f32], b: &[f32], sh: &[usize], op: &str) -> Vec<f32> {
    let (x, y) = (t(dev, a, sh), t(dev, b, sh));
    let r = match op {
        "add" => x.add(&y),
        "sub" => x.sub(&y),
        "mul" => x.mul(&y),
        "div" => x.div(&y),
        "max" => x.maximum(&y),
        "min" => x.minimum(&y),
        _ => unreachable!(),
    };
    r.unwrap().to_vec().unwrap()
}

fn run_scale<B: Backend>(dev: &B, a: &[f32], sh: &[usize], s: f32) -> Vec<f32> {
    t(dev, a, sh).scale(s).unwrap().to_vec().unwrap()
}

#[test]
fn binary_ops() -> Result<()> {
    for (i, &sh) in SHAPES_1D.iter().enumerate() {
        let n: usize = sh.iter().product();
        let a = rnd(i as u64 + 1, n, -4.0, 4.0);
        let b = rnd(i as u64 + 500, n, 0.5, 4.0);
        for op in ["add", "sub", "mul", "div", "max", "min"] {
            verify!("binary", 2e-6, 1e-6, run_bin, &a, &b, sh, op);
        }
        verify!("scale", 2e-6, 1e-6, run_scale, &a, sh, 3.25);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Broadcast binary (strided)
// -----------------------------------------------------------------------------

fn run_bcast<B: Backend>(
    dev: &B,
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    op: &str,
) -> Vec<f32> {
    let (x, y) = (t(dev, a, sa), t(dev, b, sb));
    let r = match op {
        "add" => x.broadcast_add(&y),
        "mul" => x.broadcast_mul(&y),
        "sub" => x.broadcast_sub(&y),
        _ => unreachable!(),
    };
    r.unwrap().to_vec().unwrap()
}

#[test]
fn broadcast_ops() -> Result<()> {
    let cases: &[(&[usize], &[usize])] = &[
        (&[4, 5], &[1, 5]),
        (&[4, 5], &[4, 1]),
        (&[3, 4, 5], &[1, 1, 5]),
        (&[3, 4, 5], &[1, 4, 1]),
        (&[2, 3, 4, 5], &[1, 3, 1, 5]),
        (&[7, 300], &[1, 300]),
    ];
    for (i, &(sa, sb)) in cases.iter().enumerate() {
        let a = rnd(i as u64 + 1, sa.iter().product(), -3.0, 3.0);
        let b = rnd(i as u64 + 77, sb.iter().product(), -3.0, 3.0);
        for op in ["add", "mul", "sub"] {
            verify!("broadcast", 2e-6, 1e-6, run_bcast, &a, sa, &b, sb, op);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Matmul: gemv (m==1), tiled gemm, batched, transposed rhs
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_matmul<B: Backend>(
    dev: &B,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    transpose_rhs: bool,
) -> Vec<f32> {
    let (sa, sb) = if batch == 1 {
        (vec![m, k], if transpose_rhs { vec![n, k] } else { vec![k, n] })
    } else {
        (vec![batch, m, k], if transpose_rhs { vec![batch, n, k] } else { vec![batch, k, n] })
    };
    let (x, y) = (t(dev, a, &sa), t(dev, b, &sb));
    let r = if transpose_rhs { x.matmul_t(&y) } else { x.matmul(&y) };
    r.unwrap().to_vec().unwrap()
}

#[test]
fn matmul_shapes() -> Result<()> {
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 32, 17, 1),
        (1, 256, 128, 1),
        (1, 4096, 4096, 1),
        (1, 4097, 64, 1),
        (4, 5, 6, 1),
        (16, 16, 16, 1),
        (17, 33, 19, 1),
        (64, 128, 96, 1),
        (3, 4, 5, 2),
        (8, 8, 8, 3),
        (2, 33, 40, 4),
    ];
    for (i, &(m, k, n, batch)) in cases.iter().enumerate() {
        let a = rnd(i as u64 + 1, batch * m * k, -1.0, 1.0);
        let b = rnd(i as u64 + 900, batch * k * n, -1.0, 1.0);
        let rtol = 1e-5 * (k as f32).sqrt() + 1e-5;
        verify!("matmul", rtol, 1e-4, run_matmul, &a, &b, m, k, n, batch, false);
        let bt = rnd(i as u64 + 1900, batch * n * k, -1.0, 1.0);
        verify!("matmul_t", rtol, 1e-4, run_matmul, &a, &bt, m, k, n, batch, true);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Softmax / RMSNorm / LayerNorm
// -----------------------------------------------------------------------------

fn run_softmax<B: Backend>(dev: &B, d: &[f32], sh: &[usize]) -> Vec<f32> {
    t(dev, d, sh).softmax().unwrap().to_vec().unwrap()
}
fn run_rmsnorm<B: Backend>(dev: &B, d: &[f32], w: &[f32], sh: &[usize], eps: f32) -> Vec<f32> {
    let ncols = *sh.last().unwrap();
    let wt = t(dev, w, &[ncols]);
    t(dev, d, sh).rms_norm(&wt, eps).unwrap().to_vec().unwrap()
}
fn run_layernorm<B: Backend>(
    dev: &B,
    d: &[f32],
    w: &[f32],
    b: &[f32],
    sh: &[usize],
    eps: f32,
) -> Vec<f32> {
    let ncols = *sh.last().unwrap();
    let (wt, bt) = (t(dev, w, &[ncols]), t(dev, b, &[ncols]));
    t(dev, d, sh).layer_norm(&wt, &bt, eps).unwrap().to_vec().unwrap()
}

#[test]
fn softmax_and_norms() -> Result<()> {
    let shapes: &[&[usize]] = &[&[6, 10], &[1, 1000], &[32, 257], &[4, 2048], &[3, 5, 129]];
    for (i, &sh) in shapes.iter().enumerate() {
        let n: usize = sh.iter().product();
        let ncols = *sh.last().unwrap();
        let d = rnd(i as u64 + 1, n, -5.0, 5.0);
        let w = rnd(i as u64 + 300, ncols, 0.2, 1.5);
        let b = rnd(i as u64 + 600, ncols, -0.5, 0.5);
        verify!("softmax", 1e-5, 1e-6, run_softmax, &d, sh);
        verify!("rms_norm", 2e-5, 1e-5, run_rmsnorm, &d, &w, sh, 1e-5);
        verify!("layer_norm", 2e-5, 1e-5, run_layernorm, &d, &w, &b, sh, 1e-5);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Reductions (sum / max / min / argmax / argmin) over every dim
// -----------------------------------------------------------------------------

fn run_reduce<B: Backend>(dev: &B, d: &[f32], sh: &[usize], dim: usize, op: &str) -> Vec<f32> {
    let x = t(dev, d, sh);
    let r = match op {
        "max" => x.max(dim),
        "min" => x.min(dim),
        "sum" => x.sum_keepdim(vec![dim]),
        _ => unreachable!(),
    };
    r.unwrap().to_vec().unwrap()
}
fn run_argreduce<B: Backend>(dev: &B, d: &[f32], sh: &[usize], dim: usize, max: bool) -> Vec<i64> {
    let x = t(dev, d, sh);
    let r = if max { x.argmax(dim) } else { x.argmin(dim) };
    r.unwrap().to_vec().unwrap()
}

#[test]
fn reductions() -> Result<()> {
    let shapes: &[&[usize]] = &[&[24], &[5, 7], &[2, 3, 4], &[3, 300], &[2, 3, 4, 5]];
    for (i, &sh) in shapes.iter().enumerate() {
        let n: usize = sh.iter().product();
        let d = rnd(i as u64 + 1, n, -10.0, 10.0);
        for dim in 0..sh.len() {
            for op in ["max", "min", "sum"] {
                let rtol = if op == "sum" { 1e-4 } else { 0.0 };
                verify!("reduce", rtol, 1e-4, run_reduce, &d, sh, dim, op);
            }
            let cpu_amax = run_argreduce(&CPU, &d, sh, dim, true);
            assert_eq!(cpu_amax, run_argreduce(&wg(), &d, sh, dim, true), "argmax dim {dim}");
            let cpu_amin = run_argreduce(&CPU, &d, sh, dim, false);
            assert_eq!(cpu_amin, run_argreduce(&wg(), &d, sh, dim, false), "argmin dim {dim}");
            #[cfg(feature = "cuda")]
            {
                assert_eq!(cpu_amax, run_argreduce(&cuda(), &d, sh, dim, true), "argmax cuda");
                assert_eq!(cpu_amin, run_argreduce(&cuda(), &d, sh, dim, false), "argmin cuda");
            }
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// RoPE (interleaved + non-interleaved)
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_rope<B: Backend>(
    dev: &B,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    dims: (usize, usize, usize, usize),
    max_pos: usize,
    pos: usize,
    interleaved: bool,
) -> Vec<f32> {
    let (b, h, tt, d) = dims;
    let xt = t(dev, x, &[b, h, tt, d]);
    let ct = t(dev, cos, &[max_pos, d / 2]);
    let st = t(dev, sin, &[max_pos, d / 2]);
    let r = if interleaved { xt.rope_i(&ct, &st, pos) } else { xt.rope(&ct, &st, pos) };
    r.unwrap().to_vec().unwrap()
}

#[test]
fn rope() -> Result<()> {
    let cases: &[(usize, usize, usize, usize)] = &[(1, 2, 3, 4), (2, 4, 5, 8), (1, 8, 1, 64)];
    for (i, &(b, h, tt, d)) in cases.iter().enumerate() {
        let max_pos = 32;
        let x = rnd(i as u64 + 1, b * h * tt * d, -2.0, 2.0);
        let cos: Vec<f32> = (0..max_pos * d / 2).map(|j| (j as f32 * 0.11).cos()).collect();
        let sin: Vec<f32> = (0..max_pos * d / 2).map(|j| (j as f32 * 0.11).sin()).collect();
        for pos in [0usize, 1, 7, 20] {
            let dims = (b, h, tt, d);
            verify!("rope", 1e-5, 1e-6, run_rope, &x, &cos, &sin, dims, max_pos, pos, false);
            verify!("rope_i", 1e-5, 1e-6, run_rope, &x, &cos, &sin, dims, max_pos, pos, true);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Layout: transpose (all dim pairs), cat + narrow (copy2d / copy_strided)
// -----------------------------------------------------------------------------

fn run_transpose<B: Backend>(dev: &B, d: &[f32], sh: &[usize], d1: usize, d2: usize) -> Vec<f32> {
    t(dev, d, sh).transpose(d1, d2).unwrap().contiguous().unwrap().to_vec().unwrap()
}

#[test]
fn transpose_all_pairs() -> Result<()> {
    let shapes: &[&[usize]] = &[&[3, 4], &[2, 3, 4], &[2, 3, 4, 5]];
    for (i, &sh) in shapes.iter().enumerate() {
        let n: usize = sh.iter().product();
        let d = rnd(i as u64 + 1, n, -3.0, 3.0);
        for d1 in 0..sh.len() {
            for d2 in (d1 + 1)..sh.len() {
                verify!("transpose", 0.0, 0.0, run_transpose, &d, sh, d1, d2);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cat_narrow<B: Backend>(
    dev: &B,
    a: &[f32],
    b: &[f32],
    sa: &[usize],
    sb: &[usize],
    dim: usize,
    lo: usize,
    hi: usize,
) -> Vec<f32> {
    let x = t(dev, a, sa);
    let y = t(dev, b, sb);
    let c = Tensor::cat(&[&x, &y], dim).unwrap();
    c.narrow(dim, lo..hi).unwrap().contiguous().unwrap().to_vec().unwrap()
}

#[test]
#[allow(clippy::type_complexity)]
fn cat_and_narrow() -> Result<()> {
    // (lhs shape, rhs shape, dim, narrow lo, narrow hi)
    let cases: &[(&[usize], &[usize], usize, usize, usize)] = &[
        (&[2, 3, 4], &[2, 2, 4], 1, 1, 4),
        (&[5, 6], &[5, 3], 1, 2, 7),
        (&[3, 4], &[2, 4], 0, 1, 4),
        (&[2, 3, 40], &[2, 3, 24], 2, 10, 50),
    ];
    for (i, &(sa, sb, dim, lo, hi)) in cases.iter().enumerate() {
        let a = rnd(i as u64 + 1, sa.iter().product(), -3.0, 3.0);
        let b = rnd(i as u64 + 44, sb.iter().product(), 5.0, 8.0);
        verify!("cat+narrow", 0.0, 0.0, run_cat_narrow, &a, &b, sa, sb, dim, lo, hi);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// index_select (f32 GPU path + f16/bf16 host fallback), incl. -1 -> zeros
// -----------------------------------------------------------------------------

fn run_index_select<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    sh: &[usize],
    ids: &[i64],
    dim: usize,
) -> Vec<f32> {
    let data: Vec<T> = d.iter().map(|&x| T::from_f32(x)).collect();
    let x = Tensor::<T, B>::from_vec(data, sh.to_vec(), dev).unwrap();
    let iv = Tensor::<i64, B>::from_vec(ids.to_vec(), vec![ids.len()], dev).unwrap();
    x.index_select(&iv, dim)
        .unwrap()
        .to_vec()
        .unwrap()
        .iter()
        .map(|&v| <T as WithDTypeF>::to_f32(v))
        .collect()
}

fn run_isel_f32<B: Backend>(dev: &B, d: &[f32], sh: &[usize], ids: &[i64], dim: usize) -> Vec<f32> {
    run_index_select::<f32, B>(dev, d, sh, ids, dim)
}

#[test]
fn index_select() -> Result<()> {
    let cases: &[(&[usize], &[i64], usize)] = &[
        (&[5, 3], &[0, 2, 4, 1, -1], 0),
        (&[5, 3], &[-1, 1, -1], 1),
        (&[4, 6, 2], &[3, 0, 3, 1], 0),
        (&[4, 6, 2], &[0, 5, -1, 2, 5], 1),
    ];
    for (i, &(sh, ids, dim)) in cases.iter().enumerate() {
        let n: usize = sh.iter().product();
        let d = rnd(i as u64 + 1, n, -3.0, 3.0);
        verify!("index_select f32", 0.0, 0.0, run_isel_f32, &d, sh, ids, dim);
        // f16 / bf16: host fallback, compared to the CPU backend (bit-identical).
        let c16 = run_index_select::<half::f16, _>(&CPU, &d, sh, ids, dim);
        let w16 = run_index_select::<half::f16, _>(&wg(), &d, sh, ids, dim);
        cmp("index_select f16", &c16, &w16, 0.0, 0.0);
        let cb = run_index_select::<half::bf16, _>(&CPU, &d, sh, ids, dim);
        let wb = run_index_select::<half::bf16, _>(&wg(), &d, sh, ids, dim);
        cmp("index_select bf16", &cb, &wb, 0.0, 0.0);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// scatter (scatter_set backend op)
// -----------------------------------------------------------------------------

fn run_scatter<B: Backend>(
    dev: &B,
    dst: &[f32],
    sd: &[usize],
    src: &[f32],
    ss: &[usize],
    ids: &[i64],
    dim: usize,
) -> Vec<f32> {
    let d = t(dev, dst, sd);
    let s = t(dev, src, ss);
    let iv = Tensor::<i64, B>::from_vec(ids.to_vec(), ss.to_vec(), dev).unwrap();
    d.scatter(&iv, &s, dim).unwrap().to_vec().unwrap()
}

#[test]
fn scatter() -> Result<()> {
    let dst = rnd(1, 12, -2.0, 2.0);
    let src = rnd(2, 6, 5.0, 9.0);
    let ids: &[i64] = &[0, 3, 2, 3, 1, 0];
    verify!("scatter", 0.0, 0.0, run_scatter, &dst, &[4, 3], &src, &[2, 3], ids, 0);
    Ok(())
}

// -----------------------------------------------------------------------------
// Conv1d / ConvTranspose1d (many configs)
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_conv1d<B: Backend>(
    dev: &B,
    src: &[f32],
    kern: &[f32],
    b: usize,
    ic: usize,
    oc: usize,
    len: usize,
    ks: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    groups: usize,
) -> Vec<f32> {
    let s = t(dev, src, &[b, ic, len]);
    let k = t(dev, kern, &[oc, ic / groups, ks]);
    s.conv1d(&k, None, stride, pad, dil, groups).unwrap().to_vec().unwrap()
}

#[test]
#[allow(clippy::type_complexity)]
fn conv1d() -> Result<()> {
    // (b, ic, oc, len, ks, stride, pad, dil, groups)
    let cases: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
        (1, 1, 1, 5, 3, 1, 0, 1, 1),
        (1, 1, 1, 5, 3, 1, 1, 1, 1),
        (1, 1, 1, 6, 3, 2, 0, 1, 1),
        (2, 3, 4, 7, 3, 1, 1, 1, 1),
        (1, 4, 4, 7, 3, 1, 1, 1, 2),
        (2, 6, 6, 9, 3, 1, 1, 1, 3),
        (2, 3, 8, 20, 3, 1, 2, 3, 1),
        (1, 2, 2, 25, 3, 1, 9, 9, 1),
        (2, 8, 16, 64, 5, 2, 2, 1, 1),
    ];
    for (i, &(b, ic, oc, len, ks, stride, pad, dil, groups)) in cases.iter().enumerate() {
        let src = rnd(i as u64 + 1, b * ic * len, -1.0, 1.0);
        let kern = rnd(i as u64 + 700, oc * (ic / groups) * ks, -1.0, 1.0);
        let rtol = 1e-5 * ((ic / groups * ks) as f32).sqrt() + 1e-5;
        verify!(
            "conv1d", rtol, 1e-4, run_conv1d, &src, &kern, b, ic, oc, len, ks, stride, pad, dil,
            groups
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<B: Backend>(
    dev: &B,
    src: &[f32],
    kern: &[f32],
    b: usize,
    ic: usize,
    oc: usize,
    len: usize,
    ks: usize,
    stride: usize,
) -> Vec<f32> {
    let s = t(dev, src, &[b, ic, len]);
    let k = t(dev, kern, &[ic, oc, ks]);
    s.conv_transpose1d(&k, None, stride, 0, 0, 1).unwrap().to_vec().unwrap()
}

#[test]
fn conv_transpose1d() -> Result<()> {
    // (b, ic, oc, len, ks, stride)
    let cases: &[(usize, usize, usize, usize, usize, usize)] =
        &[(1, 1, 1, 3, 3, 1), (1, 2, 3, 4, 3, 2), (2, 2, 2, 5, 2, 2), (2, 4, 3, 16, 4, 2)];
    for (i, &(b, ic, oc, len, ks, stride)) in cases.iter().enumerate() {
        let src = rnd(i as u64 + 1, b * ic * len, -1.0, 1.0);
        let kern = rnd(i as u64 + 800, ic * oc * ks, -1.0, 1.0);
        let rtol = 1e-5 * ((ic * ks) as f32).sqrt() + 1e-5;
        verify!(
            "conv_transpose1d",
            rtol,
            1e-4,
            run_conv_transpose1d,
            &src,
            &kern,
            b,
            ic,
            oc,
            len,
            ks,
            stride
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// dtype casts (all pairs), fill / zeros / full
// -----------------------------------------------------------------------------

#[test]
fn cast_all_pairs() -> Result<()> {
    let d = rnd(1, 128, -50.0, 50.0);
    macro_rules! roundtrip {
        ($ty:ty) => {{
            let c = t(&CPU, &d, &[128]).to::<$ty>()?.to_vec()?;
            let w = t(&wg(), &d, &[128]).to::<$ty>()?.to_vec()?;
            assert_eq!(c, w, "cast f32 -> {}", stringify!($ty));
        }};
    }
    roundtrip!(half::f16);
    roundtrip!(half::bf16);
    roundtrip!(i64);
    roundtrip!(u8);
    let ints: Vec<i64> = (0..64).map(|i| (i * 37 % 251) as i64 - 120).collect();
    let ci = Tensor::<i64, _>::from_vec(ints.clone(), vec![64], &CPU)?;
    let wi = Tensor::<i64, _>::from_vec(ints, vec![64], &wg())?;
    assert_eq!(ci.to::<f32>()?.to_vec()?, wi.to::<f32>()?.to_vec()?, "i64->f32");
    assert_eq!(ci.to::<u8>()?.to_vec()?, wi.to::<u8>()?.to_vec()?, "i64->u8");
    Ok(())
}

fn run_zeros<B: Backend>(dev: &B, n: usize) -> Vec<f32> {
    Tensor::<f32, B>::zeros(vec![n], dev).unwrap().to_vec().unwrap()
}
fn run_full<B: Backend>(dev: &B, v: f32, n: usize) -> Vec<f32> {
    Tensor::<f32, B>::full(v, vec![n], dev).unwrap().to_vec().unwrap()
}

#[test]
fn fill_zeros_full() -> Result<()> {
    verify!("zeros", 0.0, 0.0, run_zeros, 257);
    verify!("full", 0.0, 0.0, run_full, -3.5, 300);
    Ok(())
}
