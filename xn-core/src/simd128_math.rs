//! Four-lane `exp` and `erf` for the wasm SIMD target, and the ELU, GELU, SiLU and sigmoid
//! built on them.
//!
//! On wasm32 the scalar `f32::exp` and `libm::erff` are software routines costing several
//! nanoseconds per element, and the Mimi decoder applies ELU to every activation it
//! produces: about a tenth of a frame in Chrome. `exp` here is a range reduction and a
//! degree-6 polynomial (about 2 ulp); `erf` is Abramowitz-Stegun 7.1.26 (1.5e-7 absolute).

use crate::UnaryOp;
use core::arch::wasm32::*;

const LOG2E: f32 = 1.442_695_04;
const LN2_HI: f32 = 0.693_359_375;
const LN2_LO: f32 = -2.121_944_4e-4;
const P0: f32 = 1.987_569_15e-4;
const P1: f32 = 1.398_199_95e-3;
const P2: f32 = 8.333_451_9e-3;
const P3: f32 = 4.166_579_6e-2;
const P4: f32 = 1.666_666_5e-1;
const P5: f32 = 5.000_000_1e-1;
const EXP_LO: f32 = -87.336_54;
// Below `127.5 * ln 2`, so `round(x * LOG2E)` never reaches 128, whose exponent field would
// read as infinity. Inputs above it saturate near 1.65e38.
const EXP_HI: f32 = 88.0;
const ERF_P: f32 = 0.327_591_1;
const ERF_A1: f32 = 0.254_829_592;
const ERF_A2: f32 = -0.284_496_736;
const ERF_A3: f32 = 1.421_413_741;
const ERF_A4: f32 = -1.453_152_027;
const ERF_A5: f32 = 1.061_405_429;

#[inline(always)]
fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(f32x4_mul(a, b), c)
    }
}

/// `exp(x)` for four lanes, saturating outside `[EXP_LO, EXP_HI]`.
#[inline(always)]
pub fn exp_f32x4(x: v128) -> v128 {
    let x = f32x4_pmin(f32x4_pmax(x, f32x4_splat(EXP_LO)), f32x4_splat(EXP_HI));
    let n = f32x4_nearest(f32x4_mul(x, f32x4_splat(LOG2E)));
    let r = madd(n, f32x4_splat(-LN2_HI), x);
    let r = madd(n, f32x4_splat(-LN2_LO), r);
    let mut p = f32x4_splat(P0);
    p = madd(p, r, f32x4_splat(P1));
    p = madd(p, r, f32x4_splat(P2));
    p = madd(p, r, f32x4_splat(P3));
    p = madd(p, r, f32x4_splat(P4));
    p = madd(p, r, f32x4_splat(P5));
    let p = madd(p, f32x4_mul(r, r), r);
    let p = f32x4_add(p, f32x4_splat(1.0));
    // 2^n, built directly in the exponent field.
    let e = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(n), i32x4_splat(127)), 23);
    f32x4_mul(p, e)
}

/// `erf(x)` for four lanes.
#[inline(always)]
pub fn erf_f32x4(x: v128) -> v128 {
    let ax = f32x4_abs(x);
    let t = f32x4_div(f32x4_splat(1.0), madd(ax, f32x4_splat(ERF_P), f32x4_splat(1.0)));
    let mut p = f32x4_splat(ERF_A5);
    p = madd(p, t, f32x4_splat(ERF_A4));
    p = madd(p, t, f32x4_splat(ERF_A3));
    p = madd(p, t, f32x4_splat(ERF_A2));
    p = madd(p, t, f32x4_splat(ERF_A1));
    p = f32x4_mul(p, t);
    let e = exp_f32x4(f32x4_neg(f32x4_mul(ax, ax)));
    let y = f32x4_sub(f32x4_splat(1.0), f32x4_mul(p, e));
    v128_or(y, v128_and(x, f32x4_splat(-0.0)))
}

/// Applies `f` to `dst` four lanes at a time. A tail shorter than four lanes goes through
/// the same lanes via a padded buffer, so no element is computed differently.
fn map_inplace(dst: &mut [f32], f: impl Fn(v128) -> v128) {
    let (chunks, tail) = dst.as_chunks_mut::<4>();
    for c in chunks.iter_mut() {
        let v = f(unsafe { v128_load(c.as_ptr() as *const v128) });
        unsafe { v128_store(c.as_mut_ptr() as *mut v128, v) };
    }
    if !tail.is_empty() {
        let mut buf = [0f32; 4];
        buf[..tail.len()].copy_from_slice(tail);
        let v = f(unsafe { v128_load(buf.as_ptr() as *const v128) });
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, v) };
        tail.copy_from_slice(&buf[..tail.len()]);
    }
}

/// `x` if `x > 0`, else `alpha * (exp(x) - 1)`, in place.
pub fn elu_inplace(dst: &mut [f32], alpha: f32) {
    let (zero, one, alpha) = (f32x4_splat(0.0), f32x4_splat(1.0), f32x4_splat(alpha));
    map_inplace(dst, |x| {
        let neg = f32x4_mul(alpha, f32x4_sub(exp_f32x4(x), one));
        v128_bitselect(x, neg, f32x4_gt(x, zero))
    })
}

/// `0.5 * x * (1 + erf(x / sqrt 2))`, in place.
pub fn gelu_erf_inplace(dst: &mut [f32]) {
    let (half, one) = (f32x4_splat(0.5), f32x4_splat(1.0));
    let inv_sqrt2 = f32x4_splat(core::f32::consts::FRAC_1_SQRT_2);
    map_inplace(dst, |x| {
        let e = erf_f32x4(f32x4_mul(x, inv_sqrt2));
        f32x4_mul(f32x4_mul(x, half), f32x4_add(one, e))
    })
}

/// Lanes whose `exp` argument lies outside `[EXP_LO, EXP_HI]`, where [`exp_f32x4`] saturates.
/// NaN is in neither half, so it stays on the vector path and propagates.
#[inline(always)]
fn past_clamp(arg: v128) -> v128 {
    v128_or(f32x4_lt(arg, f32x4_splat(EXP_LO)), f32x4_gt(arg, f32x4_splat(EXP_HI)))
}

/// `f` four lanes at a time, except that a lane whose `exp` argument (`arg` of the input) is
/// past the clamp is recomputed by `scalar`, the CPU backend's own formula. The public ops
/// then underflow to zero and overflow to infinity exactly as the scalar path does, for one
/// compare per vector on inputs that stay in range.
fn map_exact_inplace(
    dst: &mut [f32],
    arg: impl Fn(v128) -> v128,
    f: impl Fn(v128) -> v128,
    scalar: impl Fn(f32) -> f32,
) {
    map_inplace(dst, |x| {
        let y = f(x);
        let edge = past_clamp(arg(x));
        if !v128_any_true(edge) {
            return y;
        }
        let (mut xs, mut ys, mut es) = ([0f32; 4], [0f32; 4], [0u32; 4]);
        unsafe {
            v128_store(xs.as_mut_ptr() as *mut v128, x);
            v128_store(ys.as_mut_ptr() as *mut v128, y);
            v128_store(es.as_mut_ptr() as *mut v128, edge);
        }
        for ((y, &x), &e) in ys.iter_mut().zip(&xs).zip(&es) {
            if e != 0 {
                *y = scalar(x);
            }
        }
        unsafe { v128_load(ys.as_ptr() as *const v128) }
    })
}

/// `exp`, in place. Past the polynomial's range it is the scalar `f32::exp`, so it
/// underflows to zero and overflows to infinity like every other backend.
pub fn exp_inplace(dst: &mut [f32]) {
    map_exact_inplace(dst, |x| x, exp_f32x4, f32::exp)
}

/// `x / (1 + exp(-x))`, in place, with the scalar formula where `-x` is past the clamp.
pub fn silu_inplace(dst: &mut [f32]) {
    let one = f32x4_splat(1.0);
    map_exact_inplace(
        dst,
        |x| f32x4_neg(x),
        |x| f32x4_div(x, f32x4_add(one, exp_f32x4(f32x4_neg(x)))),
        |x| x / (1.0 + (0.0 - x).exp()),
    )
}

/// `1 / (1 + exp(-x))`, in place, with the scalar formula where `-x` is past the clamp.
pub fn sigmoid_inplace(dst: &mut [f32]) {
    let one = f32x4_splat(1.0);
    map_exact_inplace(
        dst,
        |x| f32x4_neg(x),
        |x| f32x4_div(one, f32x4_add(one, exp_f32x4(f32x4_neg(x)))),
        |x| 1.0 / (1.0 + (0.0 - x).exp()),
    )
}

/// Runs `op` in place when this module has a kernel for it, and says whether it did. This is
/// the one list of supported ops.
pub fn unary_inplace(dst: &mut [f32], op: UnaryOp) -> bool {
    match op {
        UnaryOp::Elu { alpha } => elu_inplace(dst, alpha),
        UnaryOp::GeluErf => gelu_erf_inplace(dst),
        UnaryOp::Exp => exp_inplace(dst),
        UnaryOp::Silu => silu_inplace(dst),
        UnaryOp::Sigmoid => sigmoid_inplace(dst),
        _ => return false,
    }
    true
}

/// [`unary_inplace`] from `src` into `dst`, which must have the same length.
pub fn unary(dst: &mut [f32], src: &[f32], op: UnaryOp) -> bool {
    debug_assert_eq!(dst.len(), src.len());
    // On an empty slice every kernel is a no-op, so this asks whether there is one.
    unary_inplace(&mut [], op) && {
        dst.copy_from_slice(src);
        unary_inplace(dst, op)
    }
}

#[cfg(test)]
mod tests {
    //! Run under wasmtime: `cargo test -p xn --lib --target wasm32-wasip1 -- simd128` with
    //! `-C target-feature=+simd128`.
    use super::*;

    /// Both of `exp`'s tails, past the clamp on each side, the underflow range where the scalar
    /// result goes subnormal and then zero, the gap between `EXP_HI` and overflow, infinities
    /// and NaN.
    fn sweep() -> Vec<f32> {
        let mut v: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.05).collect();
        v.extend([0.0, -0.0, 1e-30, -1e-30, f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
        v.extend([-87.0, -87.5, -90.0, -100.0, -103.9, -110.0, -200.0]);
        v.extend([87.0, 88.0, 88.3, 88.7, 89.0, 200.0]);
        v.extend([f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        v
    }

    /// NaN for NaN; zeros (with their sign) and infinities exactly; relative error `tol`
    /// everywhere else, down to the subnormals, which get `tol * f32::MIN_POSITIVE`.
    fn assert_matches(got: f32, want: f32, tol: f32, what: &str) {
        if want.is_nan() {
            assert!(got.is_nan(), "{what}: got {got}, want NaN");
        } else if want == 0.0 || want.is_infinite() {
            assert_eq!(got.to_bits(), want.to_bits(), "{what}: got {got}, want {want}");
        } else {
            let err = (got - want).abs();
            let bound = tol * want.abs().max(f32::MIN_POSITIVE);
            assert!(err <= bound, "{what}: got {got}, want {want}, err {err}");
        }
    }

    /// `op` out of place and in place must agree bit for bit, and both match `reference`.
    fn check(name: &str, op: UnaryOp, reference: fn(f32) -> f32, tol: f32) {
        let src = sweep();
        for len in [src.len(), src.len() - 1, src.len() - 2, src.len() - 3, 1, 5] {
            let src = &src[..len];
            let mut dst = vec![0f32; len];
            assert!(unary(&mut dst, src, op));
            let mut inplace = src.to_vec();
            assert!(unary_inplace(&mut inplace, op));
            for (i, &x) in src.iter().enumerate() {
                assert_matches(dst[i], reference(x), tol, &format!("{name}({x})"));
                assert_eq!(
                    dst[i].to_bits(),
                    inplace[i].to_bits(),
                    "{name}: in place differs at {x}"
                );
            }
        }
    }

    #[test]
    fn exp_matches_the_scalar_reference() {
        check("exp", UnaryOp::Exp, f32::exp, 1e-6);
    }

    #[test]
    fn silu_matches_the_scalar_reference() {
        check("silu", UnaryOp::Silu, |x| x / (1.0 + (0.0 - x).exp()), 1e-6);
    }

    #[test]
    fn sigmoid_matches_the_scalar_reference() {
        check("sigmoid", UnaryOp::Sigmoid, |x| 1.0 / (1.0 + (0.0 - x).exp()), 1e-6);
    }

    /// ELU and GELU keep the polynomial past the clamp and use A-S 7.1.26 for `erf`, so they
    /// are held to an absolute `tol` near zero rather than the relative one of [`check`].
    #[test]
    fn elu_and_gelu_are_dispatched() {
        let elu_ref = |x: f32| if x > 0.0 { x } else { x.exp() - 1.0 };
        let gelu_ref = |x: f32| x * 0.5 * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2));
        let ops: [(UnaryOp, fn(f32) -> f32, f32); 2] =
            [(UnaryOp::Elu { alpha: 1.0 }, elu_ref, 2e-6), (UnaryOp::GeluErf, gelu_ref, 4e-7)];
        let src = sweep();
        for (op, reference, tol) in ops {
            let mut dst = vec![0f32; src.len()];
            assert!(unary(&mut dst, &src, op));
            let mut inplace = src.clone();
            assert!(unary_inplace(&mut inplace, op));
            for ((&x, &got), &ip) in src.iter().zip(&dst).zip(&inplace) {
                assert_eq!(got.to_bits(), ip.to_bits(), "{op:?}: in place differs at {x}");
                let want = reference(x);
                let close = (got - want).abs() <= tol * want.abs().max(1.0);
                let ok = (got.is_nan() && want.is_nan()) || got == want || close;
                assert!(ok, "{op:?}({x}): got {got}, want {want}");
            }
        }
    }

    /// Past the clamp every op is the scalar formula itself, so it matches to the bit: `exp`
    /// is zero or subnormal on the left and infinite on the right, and so on through SiLU and
    /// sigmoid. Mixed with in-range values, so only the edge lanes take the scalar path.
    #[test]
    fn past_the_clamp_is_the_scalar_path() {
        let edges = [-110.0f32, 1.0, -103.9, -100.0, 88.7, 89.0, 0.5, f32::INFINITY];
        let edges: Vec<f32> = edges.iter().flat_map(|&x| [x, -x]).collect();
        let exp_ref = |x: f32| x.exp();
        let silu_ref = |x: f32| x / (1.0 + (0.0 - x).exp());
        let sigmoid_ref = |x: f32| 1.0 / (1.0 + (0.0 - x).exp());
        type Op = (&'static str, UnaryOp, fn(f32) -> f32, fn(f32) -> f32);
        let ops: [Op; 3] = [
            ("exp", UnaryOp::Exp, exp_ref, |x| x),
            ("silu", UnaryOp::Silu, silu_ref, |x| -x),
            ("sigmoid", UnaryOp::Sigmoid, sigmoid_ref, |x| -x),
        ];
        for (name, op, reference, arg) in ops {
            let mut dst = vec![0f32; edges.len()];
            assert!(unary(&mut dst, &edges, op));
            for (&x, &got) in edges.iter().zip(&dst) {
                let a = arg(x);
                if !(EXP_LO..=EXP_HI).contains(&a) {
                    let want = reference(x);
                    assert!(
                        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                        "{name}({x}): got {got}, want {want}"
                    );
                }
            }
        }
    }
}
