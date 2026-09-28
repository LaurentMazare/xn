//! Four-lane `exp` and `erf` for the wasm SIMD target, and the ELU and GELU built on them.
//!
//! On wasm32 the scalar `f32::exp` and `libm::erff` are software routines costing several
//! nanoseconds per element, and the Mimi decoder applies ELU to every activation it
//! produces: about a tenth of a frame in Chrome. `exp` here is a range reduction and a
//! degree-6 polynomial (about 2 ulp); `erf` is Abramowitz-Stegun 7.1.26 (1.5e-7 absolute).

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

/// [`elu_inplace`] from `src` into `dst`.
pub fn elu(dst: &mut [f32], src: &[f32], alpha: f32) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    elu_inplace(&mut dst[..n], alpha)
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

/// [`gelu_erf_inplace`] from `src` into `dst`.
pub fn gelu_erf(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    gelu_erf_inplace(&mut dst[..n])
}

/// `exp` four lanes at a time, in place.
pub fn exp_inplace(dst: &mut [f32]) {
    map_inplace(dst, exp_f32x4)
}

pub fn exp(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    exp_inplace(&mut dst[..n])
}

/// `x / (1 + exp(-x))`, in place. The flow-LM MLP applies it on every step.
pub fn silu_inplace(dst: &mut [f32]) {
    let one = f32x4_splat(1.0);
    map_inplace(dst, |x| f32x4_div(x, f32x4_add(one, exp_f32x4(f32x4_neg(x)))))
}

pub fn silu(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    silu_inplace(&mut dst[..n])
}

/// `1 / (1 + exp(-x))`, in place.
pub fn sigmoid_inplace(dst: &mut [f32]) {
    let one = f32x4_splat(1.0);
    map_inplace(dst, |x| f32x4_div(one, f32x4_add(one, exp_f32x4(f32x4_neg(x)))))
}

pub fn sigmoid(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    sigmoid_inplace(&mut dst[..n])
}

#[cfg(test)]
mod tests {
    //! Run under wasmtime: `cargo test -p xn --lib --target wasm32-wasip1 -- simd128` with
    //! `-C target-feature=+simd128`.
    use super::*;

    /// Finite inputs across both of `exp`'s tails and past its clamp, with lengths that leave
    /// every size of padded tail.
    fn sweep() -> Vec<f32> {
        let mut v: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.05).collect();
        v.extend([0.0, -0.0, 1e-30, -1e-30, -87.0, -87.5, -90.0, -200.0, 87.0, 88.0, 88.5, 200.0]);
        v
    }

    fn assert_close(got: f32, want: f32, tol: f32, what: &str) {
        let err = (got - want).abs();
        assert!(err <= tol * want.abs().max(1.0), "{what}: got {got}, want {want}, err {err}");
    }

    /// `f` out of place and `f_inplace` must agree bit for bit, and both match `reference`.
    fn check(
        name: &str,
        f: fn(&mut [f32], &[f32]),
        f_inplace: fn(&mut [f32]),
        reference: fn(f32) -> f32,
        domain: impl Fn(f32) -> bool,
        tol: f32,
    ) {
        let src: Vec<f32> = sweep().into_iter().filter(|&x| domain(x)).collect();
        for len in [src.len(), src.len() - 1, src.len() - 2, src.len() - 3, 1, 5] {
            let src = &src[..len];
            let mut dst = vec![0f32; len];
            f(&mut dst, src);
            let mut inplace = src.to_vec();
            f_inplace(&mut inplace);
            for (i, &x) in src.iter().enumerate() {
                assert_close(dst[i], reference(x), tol, &format!("{name}({x})"));
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
        // Inside the clamp; past it the kernel saturates on purpose.
        check("exp", exp, exp_inplace, f32::exp, |x| (EXP_LO..=EXP_HI).contains(&x), 2e-6);
    }

    #[test]
    fn silu_matches_the_scalar_reference() {
        check("silu", silu, silu_inplace, |x| x / (1.0 + (-x).exp()), |_| true, 1e-6);
    }

    #[test]
    fn sigmoid_matches_the_scalar_reference() {
        check("sigmoid", sigmoid, sigmoid_inplace, |x| 1.0 / (1.0 + (-x).exp()), |_| true, 1e-6);
    }
}
