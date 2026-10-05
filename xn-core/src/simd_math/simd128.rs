//! [`F32x4`] on wasm `simd128`.

use super::F32x4;
use core::arch::wasm32::*;

/// `a * b + c`, fused when the engine fuses relaxed SIMD's multiply-add.
#[inline(always)]
pub(crate) fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(f32x4_mul(a, b), c)
    }
}

impl F32x4 for v128 {
    type Mask = v128;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        f32x4_splat(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { v128_store(p as *mut v128, self) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        f32x4_add(self, b)
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        f32x4_sub(self, b)
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        f32x4_mul(self, b)
    }
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        f32x4_div(self, b)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        f32x4_neg(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32x4_abs(self)
    }
    #[inline(always)]
    fn madd(a: Self, b: Self, c: Self) -> Self {
        madd(a, b, c)
    }
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f32x4_pmin(f32x4_pmax(self, lo), hi)
    }
    #[inline(always)]
    fn round(self) -> Self {
        f32x4_nearest(self)
    }
    #[inline(always)]
    fn exp2_int(self) -> Self {
        i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(self), i32x4_splat(127)), 23)
    }
    #[inline(always)]
    fn or_sign(self, sign: Self) -> Self {
        v128_or(self, v128_and(sign, f32x4_splat(-0.0)))
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self::Mask {
        f32x4_gt(self, b)
    }
    #[inline(always)]
    fn lt(self, b: Self) -> Self::Mask {
        f32x4_lt(self, b)
    }
    #[inline(always)]
    fn or_mask(a: Self::Mask, b: Self::Mask) -> Self::Mask {
        v128_or(a, b)
    }
    #[inline(always)]
    fn any(m: Self::Mask) -> bool {
        v128_any_true(m)
    }
    #[inline(always)]
    fn mask_lanes(m: Self::Mask) -> [bool; 4] {
        [
            u32x4_extract_lane::<0>(m) != 0,
            u32x4_extract_lane::<1>(m) != 0,
            u32x4_extract_lane::<2>(m) != 0,
            u32x4_extract_lane::<3>(m) != 0,
        ]
    }
    #[inline(always)]
    fn select(m: Self::Mask, a: Self, b: Self) -> Self {
        v128_bitselect(a, b, m)
    }
}
