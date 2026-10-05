//! [`F32x4`] on aarch64 NEON.
//!
//! The module is only compiled with `target_feature = "neon"`, which is what every intrinsic
//! below requires, so each call is sound.

use super::F32x4;
use core::arch::aarch64::*;

impl F32x4 for float32x4_t {
    type Mask = uint32x4_t;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        unsafe { vdupq_n_f32(x) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { vst1q_f32(p, self) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        unsafe { vaddq_f32(self, b) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        unsafe { vsubq_f32(self, b) }
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        unsafe { vmulq_f32(self, b) }
    }
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        unsafe { vdivq_f32(self, b) }
    }
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { vnegq_f32(self) }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { vabsq_f32(self) }
    }
    #[inline(always)]
    fn madd(a: Self, b: Self, c: Self) -> Self {
        unsafe { vfmaq_f32(c, a, b) }
    }
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        unsafe { vminq_f32(vmaxq_f32(self, lo), hi) }
    }
    #[inline(always)]
    fn round(self) -> Self {
        unsafe { vrndnq_f32(self) }
    }
    #[inline(always)]
    fn exp2_int(self) -> Self {
        unsafe {
            let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(self), vdupq_n_s32(127)));
            vreinterpretq_f32_s32(e)
        }
    }
    #[inline(always)]
    fn or_sign(self, sign: Self) -> Self {
        unsafe {
            let sign = vandq_u32(vreinterpretq_u32_f32(sign), vdupq_n_u32(0x8000_0000));
            vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(self), sign))
        }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self::Mask {
        unsafe { vcgtq_f32(self, b) }
    }
    #[inline(always)]
    fn lt(self, b: Self) -> Self::Mask {
        unsafe { vcltq_f32(self, b) }
    }
    #[inline(always)]
    fn or_mask(a: Self::Mask, b: Self::Mask) -> Self::Mask {
        unsafe { vorrq_u32(a, b) }
    }
    #[inline(always)]
    fn any(m: Self::Mask) -> bool {
        unsafe { vmaxvq_u32(m) != 0 }
    }
    #[inline(always)]
    fn mask_lanes(m: Self::Mask) -> [bool; 4] {
        unsafe {
            [
                vgetq_lane_u32::<0>(m) != 0,
                vgetq_lane_u32::<1>(m) != 0,
                vgetq_lane_u32::<2>(m) != 0,
                vgetq_lane_u32::<3>(m) != 0,
            ]
        }
    }
    #[inline(always)]
    fn select(m: Self::Mask, a: Self, b: Self) -> Self {
        unsafe { vbslq_f32(m, a, b) }
    }
}
