//! Four-lane `exp` and `erf`, and the ELU, GELU, SiLU and sigmoid built on them, for the
//! targets with 128-bit SIMD: wasm `simd128` and aarch64 NEON.
//!
//! The scalar `f32::exp` and `libm::erff` cost a call per element, and on wasm32 they are
//! software routines. `exp` here is a range reduction and a degree-6 polynomial (about 2 ulp);
//! `erf` is Abramowitz-Stegun 7.1.26 (1.5e-7 absolute). The kernels are written once against
//! [`F32x4`], which each target implements with its own instructions.

use crate::UnaryOp;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
type Lanes = core::arch::aarch64::float32x4_t;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
type Lanes = core::arch::wasm32::v128;
type Mask = <Lanes as F32x4>::Mask;

// Cephes `expf`, and Abramowitz-Stegun 7.1.26 for `erf`.
const LOG2E: f32 = core::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const P0: f32 = 1.987_569_1e-4;
const P1: f32 = 1.398_199_9e-3;
const P2: f32 = 8.333_452e-3;
const P3: f32 = 4.166_579_6e-2;
const P4: f32 = 1.666_666_5e-1;
const P5: f32 = 0.5;
const EXP_LO: f32 = -87.336_54;
// Below `127.5 * ln 2`, so `round(x * LOG2E)` never reaches 128, whose exponent field would
// read as infinity. Inputs above it saturate near 1.65e38.
const EXP_HI: f32 = 88.0;
const ERF_P: f32 = 0.327_591_1;
const ERF_A1: f32 = 0.254_829_6;
const ERF_A2: f32 = -0.284_496_72;
const ERF_A3: f32 = 1.421_413_8;
const ERF_A4: f32 = -1.453_152_1;
const ERF_A5: f32 = 1.061_405_4;

/// Four `f32` lanes and the operations the kernels here are built from. Each target maps them
/// to its own instructions, lane for lane: two targets that both fuse [`F32x4::madd`] compute
/// the same bits, except in the lanes handed to the platform's scalar `exp`.
pub(crate) trait F32x4: Copy {
    /// A lane mask, as a comparison returns it.
    type Mask: Copy;

    fn splat(x: f32) -> Self;
    /// # Safety
    /// `p` addresses four readable floats.
    unsafe fn load(p: *const f32) -> Self;
    /// # Safety
    /// `p` addresses four writable floats.
    unsafe fn store(self, p: *mut f32);
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn div(self, b: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    /// `a * b + c`, fused where the target can.
    fn madd(a: Self, b: Self, c: Self) -> Self;
    /// Clamped to `[lo, hi]`, with NaN left as NaN.
    fn clamp(self, lo: Self, hi: Self) -> Self;
    /// Rounded to the nearest integer, ties to even.
    fn round(self) -> Self;
    /// `2^n` for integral lanes `n` in `[-126, 127]`, built directly in the exponent field.
    fn exp2_int(self) -> Self;
    /// `self` with the sign bit of `sign` or-ed in.
    fn or_sign(self, sign: Self) -> Self;
    fn gt(self, b: Self) -> Self::Mask;
    fn lt(self, b: Self) -> Self::Mask;
    fn or_mask(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn any(m: Self::Mask) -> bool;
    fn mask_lanes(m: Self::Mask) -> [bool; 4];
    /// `a` where `m` is set, else `b`.
    fn select(m: Self::Mask, a: Self, b: Self) -> Self;
}

#[inline(always)]
fn splat(x: f32) -> Lanes {
    Lanes::splat(x)
}

/// `exp(x)` for four lanes, saturating outside `[EXP_LO, EXP_HI]`.
#[inline(always)]
fn exp(x: Lanes) -> Lanes {
    let x = x.clamp(splat(EXP_LO), splat(EXP_HI));
    let n = x.mul(splat(LOG2E)).round();
    let r = Lanes::madd(n, splat(-LN2_HI), x);
    let r = Lanes::madd(n, splat(-LN2_LO), r);
    let mut p = splat(P0);
    p = Lanes::madd(p, r, splat(P1));
    p = Lanes::madd(p, r, splat(P2));
    p = Lanes::madd(p, r, splat(P3));
    p = Lanes::madd(p, r, splat(P4));
    p = Lanes::madd(p, r, splat(P5));
    let p = Lanes::madd(p, r.mul(r), r);
    let p = p.add(splat(1.0));
    p.mul(n.exp2_int())
}

/// `erf(x)` for four lanes.
#[inline(always)]
fn erf(x: Lanes) -> Lanes {
    let one = splat(1.0);
    let ax = x.abs();
    let t = one.div(Lanes::madd(ax, splat(ERF_P), one));
    let mut p = splat(ERF_A5);
    p = Lanes::madd(p, t, splat(ERF_A4));
    p = Lanes::madd(p, t, splat(ERF_A3));
    p = Lanes::madd(p, t, splat(ERF_A2));
    p = Lanes::madd(p, t, splat(ERF_A1));
    p = p.mul(t);
    let e = exp(ax.mul(ax).neg());
    one.sub(p.mul(e)).or_sign(x)
}

/// `dst[i] = f(src[i])`, four lanes at a time. A tail shorter than four lanes goes through the
/// same lanes via a padded buffer, so no element is computed differently.
///
/// # Safety
/// `src` and `dst` address `len` floats each. They may be the same pointer, since each vector
/// is loaded before it is stored, but must not otherwise overlap.
#[inline(always)]
unsafe fn map_raw(dst: *mut f32, src: *const f32, len: usize, f: impl Fn(Lanes) -> Lanes) {
    let full = len / 4 * 4;
    let mut i = 0;
    while i < full {
        unsafe { f(Lanes::load(src.add(i))).store(dst.add(i)) };
        i += 4;
    }
    if full < len {
        let (mut buf, n) = ([0f32; 4], len - full);
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(full), buf.as_mut_ptr(), n);
            f(Lanes::load(buf.as_ptr())).store(buf.as_mut_ptr());
            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst.add(full), n);
        }
    }
}

/// Applies `f` to `dst` in place, or from `src` when given, which must have `dst`'s length.
#[inline(always)]
fn map(dst: &mut [f32], src: Option<&[f32]>, f: impl Fn(Lanes) -> Lanes) {
    let (len, d) = (dst.len(), dst.as_mut_ptr());
    let s = match src {
        Some(src) => {
            assert_eq!(src.len(), len);
            src.as_ptr()
        }
        None => d,
    };
    // SAFETY: both address `len` floats; a `&mut` and a `&` cannot overlap, so they are either
    // the same slice or disjoint.
    unsafe { map_raw(d, s, len, f) }
}

/// Lanes whose `exp` argument lies outside `[EXP_LO, EXP_HI]`, where [`exp`] saturates. NaN
/// is in neither half, so it stays on the vector path and propagates.
#[inline(always)]
fn past_clamp(arg: Lanes) -> Mask {
    Lanes::or_mask(arg.lt(splat(EXP_LO)), arg.gt(splat(EXP_HI)))
}

/// `y`, except that the lanes in `edge` are recomputed from `x` by `scalar`.
#[inline(always)]
fn exact(x: Lanes, y: Lanes, edge: Mask, scalar: impl Fn(f32) -> f32) -> Lanes {
    #[cold]
    fn fix(x: Lanes, y: Lanes, edge: Mask, scalar: impl Fn(f32) -> f32) -> Lanes {
        let (mut xs, mut ys) = ([0f32; 4], [0f32; 4]);
        // SAFETY: both arrays hold four floats.
        unsafe {
            x.store(xs.as_mut_ptr());
            y.store(ys.as_mut_ptr());
        }
        for ((y, &x), e) in ys.iter_mut().zip(&xs).zip(Lanes::mask_lanes(edge)) {
            if e {
                *y = scalar(x);
            }
        }
        // SAFETY: as above.
        unsafe { Lanes::load(ys.as_ptr()) }
    }
    if Lanes::any(edge) { fix(x, y, edge, scalar) } else { y }
}

/// `f`, except that a lane whose `exp` argument (`arg` of the input) is past the clamp is
/// recomputed by `scalar`, the CPU backend's own formula. The public ops then underflow to zero
/// and overflow to infinity exactly as the scalar path does, for one compare per vector on
/// inputs that stay in range.
#[inline(always)]
fn exact_op(
    arg: impl Fn(Lanes) -> Lanes,
    f: impl Fn(Lanes) -> Lanes,
    scalar: impl Fn(f32) -> f32,
) -> impl Fn(Lanes) -> Lanes {
    move |x| exact(x, f(x), past_clamp(arg(x)), &scalar)
}

/// `x` if `x > 0`, else `alpha * (exp(x) - 1)`.
#[inline(always)]
fn elu(alpha: f32) -> impl Fn(Lanes) -> Lanes {
    let (zero, one, alpha) = (splat(0.0), splat(1.0), splat(alpha));
    move |x| Lanes::select(x.gt(zero), x, alpha.mul(exp(x).sub(one)))
}

/// `0.5 * x * (1 + erf(x / sqrt 2))`.
#[inline(always)]
fn gelu_erf() -> impl Fn(Lanes) -> Lanes {
    let (half, one) = (splat(0.5), splat(1.0));
    let inv_sqrt2 = splat(core::f32::consts::FRAC_1_SQRT_2);
    move |x| x.mul(half).mul(one.add(erf(x.mul(inv_sqrt2))))
}

/// `exp`. Past the polynomial's range it is the scalar `f32::exp`, so it underflows to zero and
/// overflows to infinity like every other backend.
#[inline(always)]
fn exp_exact() -> impl Fn(Lanes) -> Lanes {
    exact_op(|x| x, exp, f32::exp)
}

/// `x / (1 + exp(-x))`, with the scalar formula where `-x` is past the clamp.
#[inline(always)]
fn silu() -> impl Fn(Lanes) -> Lanes {
    let one = splat(1.0);
    exact_op(|x| x.neg(), move |x| x.div(one.add(exp(x.neg()))), |x| x / (1.0 + (0.0 - x).exp()))
}

/// `1 / (1 + exp(-x))`, with the scalar formula where `-x` is past the clamp.
#[inline(always)]
fn sigmoid() -> impl Fn(Lanes) -> Lanes {
    let one = splat(1.0);
    exact_op(
        |x| x.neg(),
        move |x| one.div(one.add(exp(x.neg()))),
        |x| 1.0 / (1.0 + (0.0 - x).exp()),
    )
}

/// Runs `op` over `dst`, from `src` when given, if this module has a kernel for it, and says
/// whether it did. This is the one list of supported ops.
fn run(dst: &mut [f32], src: Option<&[f32]>, op: UnaryOp) -> bool {
    match op {
        UnaryOp::Elu { alpha } => map(dst, src, elu(alpha)),
        UnaryOp::GeluErf => map(dst, src, gelu_erf()),
        UnaryOp::Exp => map(dst, src, exp_exact()),
        UnaryOp::Silu => map(dst, src, silu()),
        UnaryOp::Sigmoid => map(dst, src, sigmoid()),
        _ => return false,
    }
    true
}

/// Runs `op` in place when this module has a kernel for it, and says whether it did.
pub fn unary_inplace(dst: &mut [f32], op: UnaryOp) -> bool {
    run(dst, None, op)
}

/// [`unary_inplace`] from `src` into `dst`, which must have the same length.
pub fn unary(dst: &mut [f32], src: &[f32], op: UnaryOp) -> bool {
    run(dst, Some(src), op)
}

#[cfg(test)]
mod tests {
    //! On wasm, run under wasmtime: `cargo test -p xn --lib --target wasm32-wasip1 -- simd`
    //! with `-C target-feature=+simd128,+relaxed-simd`.
    use super::*;

    fn exp_ref(x: f32) -> f32 {
        x.exp()
    }
    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (0.0 - x).exp())
    }
    fn sigmoid_ref(x: f32) -> f32 {
        1.0 / (1.0 + (0.0 - x).exp())
    }
    fn elu_ref(x: f32) -> f32 {
        if x > 0.0 { x } else { x.exp() - 1.0 }
    }
    fn gelu_ref(x: f32) -> f32 {
        x * 0.5 * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2))
    }

    /// Both of `exp`'s tails, past the clamp on each side, the underflow range where the scalar
    /// result goes subnormal and then zero, the gap between `EXP_HI` and overflow, zeros,
    /// subnormals, infinities and NaN.
    fn sweep() -> Vec<f32> {
        let mut v: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.05).collect();
        v.extend([0.0, -0.0, 1e-30, -1e-30, f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
        v.extend([1e-40, -1e-40, f32::from_bits(1), -f32::from_bits(1)]);
        v.extend([-87.0, -87.5, -88.0, -88.7, -89.0, -90.0, -100.0, -103.9, -110.0, -200.0]);
        v.extend([87.0, 88.0, 88.3, 88.7, 89.0, 200.0, f32::MAX, f32::MIN]);
        v.extend([f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN]);
        v
    }

    /// A dense sweep: every thousandth of a unit over `[-100, 100]`, and a stride through all
    /// bit patterns, which visits every exponent of both signs.
    fn dense() -> Vec<f32> {
        let mut v: Vec<f32> = (-100_000..=100_000).map(|i| i as f32 * 1e-3).collect();
        v.extend((0..=u32::MAX / 16_411).map(|i| f32::from_bits(i * 16_411)));
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
        check("exp", UnaryOp::Exp, exp_ref, 1e-6);
    }

    #[test]
    fn silu_matches_the_scalar_reference() {
        check("silu", UnaryOp::Silu, silu_ref, 1e-6);
    }

    #[test]
    fn sigmoid_matches_the_scalar_reference() {
        check("sigmoid", UnaryOp::Sigmoid, sigmoid_ref, 1e-6);
    }

    /// ELU and GELU keep the polynomial past the clamp and use A-S 7.1.26 for `erf`, so they
    /// are held to an absolute `tol` near zero rather than the relative one of [`check`].
    #[test]
    fn elu_and_gelu_are_dispatched() {
        type Op = (UnaryOp, fn(f32) -> f32, f32);
        let ops: [Op; 3] = [
            (UnaryOp::Elu { alpha: 1.0 }, elu_ref, 2e-6),
            (
                UnaryOp::Elu { alpha: 0.5 },
                |x| if x > 0.0 { x } else { 0.5 * (x.exp() - 1.0) },
                2e-6,
            ),
            (UnaryOp::GeluErf, gelu_ref, 4e-7),
        ];
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
        let edges = [-110.0f32, 1.0, -103.9, -100.0, 88.7, 89.0, 0.5, f32::INFINITY, 200.0];
        let edges: Vec<f32> = edges.iter().flat_map(|&x| [x, -x]).collect();
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

    /// The values the ops must get exactly right: signed zeros, subnormals, both sides of
    /// `exp`'s range, infinities and NaN.
    #[test]
    fn edge_values() {
        let tiny = f32::from_bits(1);
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        #[rustfmt::skip]
        let cases: &[(UnaryOp, f32, f32)] = &[
            (UnaryOp::Exp, 0.0, 1.0), (UnaryOp::Exp, -0.0, 1.0), (UnaryOp::Exp, tiny, 1.0),
            (UnaryOp::Exp, -tiny, 1.0), (UnaryOp::Exp, 89.0, inf), (UnaryOp::Exp, 200.0, inf),
            (UnaryOp::Exp, inf, inf), (UnaryOp::Exp, -110.0, 0.0), (UnaryOp::Exp, -200.0, 0.0),
            (UnaryOp::Exp, -inf, 0.0), (UnaryOp::Exp, -89.0, (-89f32).exp()),
            (UnaryOp::Exp, -88.0, (-88f32).exp()), (UnaryOp::Exp, nan, nan),
            (UnaryOp::Sigmoid, 0.0, 0.5), (UnaryOp::Sigmoid, -0.0, 0.5),
            (UnaryOp::Sigmoid, tiny, 0.5), (UnaryOp::Sigmoid, 200.0, 1.0),
            (UnaryOp::Sigmoid, inf, 1.0), (UnaryOp::Sigmoid, -200.0, 0.0),
            (UnaryOp::Sigmoid, -inf, 0.0), (UnaryOp::Sigmoid, nan, nan),
            (UnaryOp::Silu, 0.0, 0.0), (UnaryOp::Silu, -0.0, -0.0), (UnaryOp::Silu, 200.0, 200.0),
            (UnaryOp::Silu, inf, inf), (UnaryOp::Silu, -200.0, -0.0), (UnaryOp::Silu, -inf, nan),
            (UnaryOp::Silu, nan, nan),
            (UnaryOp::Elu { alpha: 1.0 }, 0.0, 0.0), (UnaryOp::Elu { alpha: 1.0 }, -0.0, 0.0),
            (UnaryOp::Elu { alpha: 1.0 }, tiny, tiny), (UnaryOp::Elu { alpha: 1.0 }, -tiny, 0.0),
            (UnaryOp::Elu { alpha: 1.0 }, 200.0, 200.0), (UnaryOp::Elu { alpha: 1.0 }, inf, inf),
            (UnaryOp::Elu { alpha: 1.0 }, -89.0, -1.0), (UnaryOp::Elu { alpha: 1.0 }, -200.0, -1.0),
            (UnaryOp::Elu { alpha: 1.0 }, -inf, -1.0), (UnaryOp::Elu { alpha: 1.0 }, nan, nan),
            (UnaryOp::GeluErf, 0.0, 0.0), (UnaryOp::GeluErf, -0.0, -0.0),
            (UnaryOp::GeluErf, 200.0, 200.0), (UnaryOp::GeluErf, inf, inf),
            (UnaryOp::GeluErf, -200.0, -0.0), (UnaryOp::GeluErf, -inf, nan),
            (UnaryOp::GeluErf, nan, nan),
        ];
        for &(op, x, want) in cases {
            // Each value in every lane position, among ordinary neighbours.
            for pos in 0..5 {
                let mut src = [0.25f32, -1.5, 3.0, -0.75, 0.5];
                src[pos] = x;
                let mut dst = [0f32; 5];
                assert!(unary(&mut dst, &src, op));
                let got = dst[pos];
                let ok = if want.is_nan() { got.is_nan() } else { got.to_bits() == want.to_bits() };
                assert!(ok, "{op:?}({x:e}) at lane {pos}: got {got:e}, want {want:e}");
            }
        }
        // Subnormal inputs to GELU come out as half of themselves, to within an ulp.
        for x in [tiny, -tiny, 1e-40, -1e-40, -f32::MIN_POSITIVE] {
            let mut dst = [0f32; 1];
            assert!(unary(&mut dst, &[x], UnaryOp::GeluErf));
            assert!((dst[0] - 0.5 * x).abs() <= tiny, "gelu({x:e}) = {:e}", dst[0]);
        }
    }

    /// Distance in units in the last place, counting across zero.
    fn ulps(a: f32, b: f32) -> u32 {
        let key = |x: f32| {
            let b = x.to_bits() as i32;
            if b < 0 { i32::MIN - b } else { b }
        };
        key(a).abs_diff(key(b))
    }

    /// Max ulp and max absolute error of every op over a dense sweep, against the scalar
    /// formula (libm's `expf` and `erff`) and against the same formula in `f64`. Ulp only for
    /// the ops whose error is relative; ELU's `exp(x) - 1` cancels near zero and A-S 7.1.26 is
    /// an absolute bound, so GELU and ELU get an absolute error, scaled by `max(1, |want|)` so
    /// that it reads as relative where results are large. The `f64` column leaves out results
    /// outside the normal `f32` range, exact or scalar, where the scalar formula is the
    /// reference and is matched to the bit. Printed with `--nocapture`.
    #[test]
    fn accuracy_against_libm() {
        let xs = dense();
        type Op = (&'static str, UnaryOp, fn(f32) -> f32, fn(f64) -> f64, Option<u32>, f32);
        let ops: [Op; 5] = [
            ("exp", UnaryOp::Exp, exp_ref, f64::exp, Some(2), 2.4e-7),
            (
                "sigmoid",
                UnaryOp::Sigmoid,
                sigmoid_ref,
                |x| 1.0 / (1.0 + (-x).exp()),
                Some(4),
                2.4e-7,
            ),
            ("silu", UnaryOp::Silu, silu_ref, |x| x / (1.0 + (-x).exp()), Some(4), 3e-7),
            (
                "elu",
                UnaryOp::Elu { alpha: 1.0 },
                elu_ref,
                |x| if x > 0.0 { x } else { x.exp_m1() },
                None,
                1.2e-7,
            ),
            (
                "gelu",
                UnaryOp::GeluErf,
                gelu_ref,
                |x| x * 0.5 * (1.0 + libm::erf(x * core::f64::consts::FRAC_1_SQRT_2)),
                None,
                4e-7,
            ),
        ];
        let mut report = String::new();
        for (name, op, reference, exact, max_ulp, max_abs) in ops {
            let mut got = vec![0f32; xs.len()];
            assert!(unary(&mut got, &xs, op));
            // (ulp, at, abs, at), against libm and against f64.
            let mut worst = [(0u32, 0f32, 0f32, 0f32); 2];
            for (&x, &g) in xs.iter().zip(&got) {
                let w = reference(x);
                if w.is_nan() {
                    assert!(g.is_nan(), "{name}({x:e}): got {g:e}, want NaN");
                    continue;
                }
                let e = exact(x as f64);
                let normal =
                    |v: f64| (f32::MIN_POSITIVE as f64..=f32::MAX as f64).contains(&v.abs());
                let wants = [Some(w), (normal(e) && normal(w as f64)).then_some(e as f32)];
                for (worst, w) in worst.iter_mut().zip(wants) {
                    let Some(w) = w else { continue };
                    let u = ulps(g, w);
                    if u > worst.0 {
                        (worst.0, worst.1) = (u, x);
                    }
                    let a = if g == w { 0.0 } else { (g - w).abs() / w.abs().max(1.0) };
                    if a > worst.2 {
                        (worst.2, worst.3) = (a, x);
                    }
                }
            }
            for ((ulp, ulp_at, abs, abs_at), vs) in worst.into_iter().zip(["libm", "f64"]) {
                let ulp = match max_ulp {
                    Some(_) => format!("{ulp:>2} ulp at {ulp_at:>10.3e}, "),
                    None => String::new(),
                };
                report += &format!("{name:>8} vs {vs:<4}: {ulp}abs {abs:.2e} at {abs_at:>10.3e}\n");
            }
            let (ulp, ulp_at, abs, abs_at) = worst[0];
            if let Some(max_ulp) = max_ulp {
                assert!(ulp <= max_ulp, "{name}: {ulp} ulp at {ulp_at:e}");
            }
            assert!(abs <= max_abs, "{name}: abs error {abs:e} at {abs_at:e}");
        }
        eprint!("{report}");
    }
}
