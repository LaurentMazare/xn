//! Arm KleidiAI's SME2 kernels for `q8_0` weights, behind the `kai` feature.
//!
//! Apple's M4 and M5, and Arm's Cortex-X925 and later, have an SME2 matrix unit that
//! multiplies a whole tile per instruction. Rust cannot reach it yet, so this binds
//! [KleidiAI]'s hand-written kernels for it, which `build.rs` compiles from the files vendored
//! in `third_party/kleidiai`. On a CPU without SME2, or with `XN_KAI=0`, nothing here is used
//! and `q8_0` weights keep the layouts in [`super::repack`].
//!
//! [KleidiAI]: https://github.com/ARM-software/kleidiai
//!
//! # Where it helps
//!
//! Once a matmul has more than a few rows the SME2 kernels are several times faster than the
//! NEON ones, so prompt prefill gets faster. A one-row matmul, which is what batch-one decode
//! runs, is bound by memory bandwidth whatever the kernel, so decode does not.
//!
//! # What is stored
//!
//! KleidiAI has no int8 kernel with per-block scales. Its int8 format is symmetric with one
//! f32 scale per output row, so a `q8_0` weight is requantized when it is loaded: each row is
//! dequantized and rescaled to `max|w| / 127`. That loses about a bit in the blocks of a row
//! that are much smaller than its largest one. llama.cpp's KleidiAI backend does the same.
//! Activations are quantized per row by KleidiAI's own packing routine.
//!
//! Because that is lossy, [`raw_data`](super::QuantizedType::raw_data) quantizes the stored
//! weights to `q8_0` again rather than returning the bytes that were loaded.
//!
//! # Threads
//!
//! A matmul runs on the calling thread. The SME2 unit is shared by the cores of a cluster, so
//! splitting one matmul over several threads was measured to gain little or nothing.

use super::GgmlDType;
use super::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use crate::Result;
use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::OnceLock;

// -------------------------------------------------------------------------------------------
// CPU support
// -------------------------------------------------------------------------------------------

/// Whether this CPU has SME2. `is_aarch64_feature_detected!("sme2")` is not stable yet, so
/// this asks the OS the same way the standard library does.
#[cfg(target_vendor = "apple")]
fn has_sme2() -> bool {
    use std::ffi::{c_char, c_int};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }
    let mut value: c_int = 0;
    let mut len = std::mem::size_of::<c_int>();
    // SAFETY: `value` and `len` are valid for writes, and `len` is the size of `value`.
    let rc = unsafe {
        sysctlbyname(
            c"hw.optional.arm.FEAT_SME2".as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && value != 0
}

/// Whether this CPU has SME2, from the kernel's hardware capability bits.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn has_sme2() -> bool {
    use std::ffi::c_ulong;
    const AT_HWCAP2: c_ulong = 26;
    const HWCAP2_SME2: c_ulong = 1 << 37;
    unsafe extern "C" {
        fn getauxval(kind: c_ulong) -> c_ulong;
    }
    // SAFETY: `getauxval` has no preconditions.
    unsafe { getauxval(AT_HWCAP2) & HWCAP2_SME2 != 0 }
}

fn sme2() -> bool {
    static SME2: OnceLock<bool> = OnceLock::new();
    *SME2.get_or_init(has_sme2)
}

/// Whether `q8_0` weights take this layout: the CPU has SME2 and `XN_KAI` is not `0`. Read
/// once, because weights are packed for it as they are loaded.
pub fn active() -> bool {
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| sme2() && std::env::var("XN_KAI").as_deref() != Ok("0"))
}

// -------------------------------------------------------------------------------------------
// Bindings
// -------------------------------------------------------------------------------------------

/// `kai_rhs_pack_qsi8cx_params`.
#[repr(C)]
struct RhsPackParams {
    lhs_zero_point: i32,
    scale_multiplier: f32,
}

unsafe extern "C" {
    fn kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
        m_idx_start: usize,
        lhs: *const f32,
        lhs_stride: usize,
        lhs_packed: *mut c_void,
    );
    fn kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs: *const i8,
        bias: *const f32,
        scale: *const f32,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const RhsPackParams,
    );
}

type RunFn = unsafe extern "C" fn(
    usize,
    usize,
    usize,
    *const c_void,
    *const c_void,
    *mut f32,
    usize,
    usize,
    f32,
    f32,
);

/// A matmul kernel's tile constants, which its packed operands are laid out for, and its
/// entry point.
struct Kernel {
    mr: usize,
    nr: usize,
    kr: usize,
    sr: usize,
    run: RunFn,
}

/// Declares one kernel's getters and entry point, and a constructor that reads the getters.
macro_rules! kernel {
    ($ctor:ident: $mr:ident, $nr:ident, $kr:ident, $sr:ident, $run:ident) => {
        unsafe extern "C" {
            fn $mr() -> usize;
            fn $nr() -> usize;
            fn $kr() -> usize;
            fn $sr() -> usize;
            fn $run(
                m: usize,
                n: usize,
                k: usize,
                lhs_packed: *const c_void,
                rhs_packed: *const c_void,
                dst: *mut f32,
                dst_stride_row: usize,
                dst_stride_col: usize,
                clamp_min: f32,
                clamp_max: f32,
            );
        }

        /// # Safety
        /// The CPU must have SME2: the getters read the vector length with an SME instruction.
        unsafe fn $ctor() -> Kernel {
            unsafe { Kernel { mr: $mr(), nr: $nr(), kr: $kr(), sr: $sr(), run: $run } }
        }
    };
}

kernel!(gemm_kernel:
    kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa);
kernel!(gemv_kernel:
    kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot);

/// The gemm, for several rows, and the gemv, for one. They read the same packed weights.
fn kernels() -> &'static (Kernel, Kernel) {
    static KERNELS: OnceLock<(Kernel, Kernel)> = OnceLock::new();
    KERNELS.get_or_init(|| {
        assert!(sme2(), "kai: the SME2 kernels need an SME2 CPU");
        // SAFETY: the CPU has SME2, checked just above.
        let (gemm, gemv) = unsafe { (gemm_kernel(), gemv_kernel()) };
        assert_eq!(
            (gemm.nr, gemm.kr, gemm.sr),
            (gemv.nr, gemv.kr, gemv.sr),
            "kai: the gemm and the gemv disagree on the weight layout"
        );
        (gemm, gemv)
    })
}

// -------------------------------------------------------------------------------------------
// Storage
// -------------------------------------------------------------------------------------------

/// A `[n, k]` `q8_0` weight, requantized to one int8 scale per row and packed for the SME2
/// kernels.
pub struct Q8_0Kai {
    packed: Vec<u8>,
    n: usize,
    k: usize,
}

impl Q8_0Kai {
    /// Pack `n` rows of `k / 32` `q8_0` blocks. Fails on a CPU without SME2, on blocks that are
    /// not a `[n, k]` weight, and on a row with a weight that is not finite.
    pub fn from_q8_0(src: &[BlockQ8_0], n: usize, k: usize) -> Result<Self> {
        if !sme2() {
            crate::bail!("kai: this CPU has no SME2")
        }
        if n == 0 || k == 0 || !k.is_multiple_of(QK8_0) || src.len() != n * (k / QK8_0) {
            crate::bail!("kai: {} q8_0 blocks are not a [{n}, {k}] weight", src.len())
        }
        let mut qs = vec![0i8; n * k];
        let mut scales = vec![0f32; n];
        let mut row = vec![0f32; k];
        for (r, (q, scale)) in qs.chunks_mut(k).zip(scales.iter_mut()).enumerate() {
            BlockQ8_0::to_float(&src[r * k / QK8_0..(r + 1) * k / QK8_0], &mut row)?;
            let mut max = 0f32;
            for w in &row {
                if !w.is_finite() {
                    crate::bail!("kai: row {r} has a weight that is not finite")
                }
                max = max.max(w.abs());
            }
            *scale = max / 127.0;
            let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
            for (q, w) in q.iter_mut().zip(&row) {
                *q = (w * inv).round() as i8;
            }
        }

        let (gemm, _) = kernels();
        // `lhs_zero_point = 1` stores plain row sums; the kernel scales them by each
        // activation row's zero point when it runs.
        let params = RhsPackParams { lhs_zero_point: 1, scale_multiplier: 1.0 };
        // SAFETY: `qs` holds `n * k` values and `scales` `n`, and `packed` is sized by the
        // packing routine's own query.
        let packed = unsafe {
            let size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                n, k, gemm.nr, gemm.kr, gemm.sr,
            );
            let mut packed = vec![0u8; size];
            kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                1,
                n,
                k,
                gemm.nr,
                gemm.kr,
                gemm.sr,
                qs.as_ptr(),
                std::ptr::null(),
                scales.as_ptr(),
                packed.as_mut_ptr().cast(),
                0,
                &params,
            );
            packed
        };
        Ok(Self { packed, n, k })
    }

    /// Read row `r`'s int8 values out of the packed layout, and return its scale.
    ///
    /// The layout is `kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon`'s: groups of `nr` rows, each the
    /// rows' values interleaved `kr` at a time and padded to `k_padded`, then `nr` row sums
    /// (i32), `nr` scales (f32) and `nr` biases (f32).
    fn row(&self, r: usize, out: &mut [i8]) -> f32 {
        let (gemm, _) = kernels();
        let (nr, kr) = (gemm.nr, gemm.kr);
        // SAFETY: a size query.
        let stride = unsafe {
            kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(self.k, nr, kr, gemm.sr)
        };
        let k_padded = stride / nr - 12;
        debug_assert!(gemm.sr == 1 && k_padded >= self.k && stride == nr * (k_padded + 12));
        let (base, i) = ((r / nr) * stride, r % nr);
        for (b, chunk) in out.chunks_mut(kr).enumerate() {
            let at = base + (b * nr + i) * kr;
            let len = chunk.len();
            for (o, &v) in chunk.iter_mut().zip(&self.packed[at..at + len]) {
                *o = v as i8;
            }
        }
        let at = base + nr * (k_padded + 4) + i * 4;
        f32::from_ne_bytes(self.packed[at..at + 4].try_into().unwrap())
    }

    /// The weights the kernels multiply by, as f32.
    fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.n * self.k];
        let mut q = vec![0i8; self.k];
        for (r, row) in out.chunks_mut(self.k).enumerate() {
            let scale = self.row(r, &mut q);
            for (o, &v) in row.iter_mut().zip(&q) {
                *o = v as f32 * scale;
            }
        }
        out
    }

    /// `dst = lhs x self^T` for `m` activation rows, in one kernel call.
    #[tracing::instrument(name = "q-matmul-kai", skip_all, fields(m = m, n = self.n, k = self.k))]
    fn matmul(&self, m: usize, lhs: &[f32], dst: &mut [f32]) {
        let (gemm, gemv) = kernels();
        let kern = if m == 1 { gemv } else { gemm };
        let (n, k) = (self.n, self.k);
        let f32_size = std::mem::size_of::<f32>();
        // SAFETY: `lhs` holds `m * k` floats and `dst` at least `m * n`, which `matmul_t`
        // checked; `lhs_packed` is sized by the packing routine's own query; and the one call
        // covers the whole output.
        unsafe {
            let size =
                kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, kern.mr, kern.kr, kern.sr);
            // Zeroed because the kernel also computes the padding rows of a partial last tile,
            // and only drops them on the store, so they have to hold finite values.
            let mut lhs_packed = vec![0u8; size];
            kai_run_lhs_quant_pack_qai8dxp_f32(
                m,
                k,
                kern.mr,
                kern.kr,
                kern.sr,
                0,
                lhs.as_ptr(),
                k * f32_size,
                lhs_packed.as_mut_ptr().cast(),
            );
            (kern.run)(
                m,
                n,
                k,
                lhs_packed.as_ptr().cast(),
                self.packed.as_ptr().cast(),
                dst.as_mut_ptr(),
                n * f32_size,
                f32_size,
                f32::NEG_INFINITY,
                f32::INFINITY,
            );
        }
    }
}

impl super::QuantizedType for Q8_0Kai {
    fn dtype(&self) -> GgmlDType {
        // The packing is a storage detail; callers still see a q8_0 tensor.
        GgmlDType::Q8_0
    }

    fn block_size(&self) -> usize {
        QK8_0
    }

    /// Bytes held in memory, which is the packed layout.
    fn size(&self) -> usize {
        self.packed.len()
    }

    /// Bytes of the tensor as `q8_0`, which is what `raw_data` returns and what the
    /// GGUF writer sizes the tensor by. Not the memory held: see `size`.
    fn storage_size_in_bytes(&self) -> usize {
        self.n * self.k / QK8_0 * std::mem::size_of::<BlockQ8_0>()
    }

    /// The packed layout, valid for `size` bytes only. Unlike the other storages, this is not
    /// `storage_size_in_bytes` long, and for `k > 192` it is shorter, so reading that many bytes
    /// from here reads past the end. `raw_data` gives the `q8_0` bytes.
    fn as_ptr(&self) -> *const u8 {
        self.packed.as_ptr()
    }

    fn raw_data(&self) -> Result<Cow<'_, [u8]>> {
        let mut blocks = vec![BlockQ8_0::zeros(); self.n * self.k / QK8_0];
        BlockQ8_0::from_float(&self.to_f32(), &mut blocks)?;
        // SAFETY: `BlockQ8_0` is plain data, and the slice covers exactly `blocks`.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                std::mem::size_of_val(blocks.as_slice()),
            )
        };
        Ok(Cow::Owned(bytes.to_vec()))
    }

    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>> {
        if elem_count != self.n * self.k {
            crate::bail!("kai: asked for {elem_count} values of a [{}, {}] weight", self.n, self.k)
        }
        Ok(self.to_f32())
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        if xs.len() != self.n * self.k {
            crate::bail!("kai: {} values for a [{}, {}] weight", xs.len(), self.n, self.k)
        }
        let mut blocks = vec![BlockQ8_0::zeros(); xs.len() / QK8_0];
        BlockQ8_0::from_float(xs, &mut blocks)?;
        *self = Self::from_q8_0(&blocks, self.n, self.k)?;
        Ok(())
    }

    fn matmul_t(
        &self,
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
    ) -> Result<()> {
        if (n, k) != (self.n, self.k) {
            crate::bail!("kai: matmul_t with n={n}, k={k} on a [{}, {}] weight", self.n, self.k)
        }
        if lhs.len() != m * k || dst.len() < m * n {
            crate::bail!(
                "kai: matmul_t with {} lhs and {} dst values for m={m}",
                lhs.len(),
                dst.len()
            )
        }
        if m > 0 {
            self.matmul(m, lhs, dst);
        }
        Ok(())
    }
}

/// The storage for a freshly read or quantized `q8_0` weight when this path is active, or
/// `None` to leave the choice to [`super::repack`]. A weight that cannot be packed keeps the
/// other layouts, which answer the same matmuls.
pub fn q8_0_storage(src: &[BlockQ8_0], dims: &[usize]) -> Option<super::QStorage> {
    let &[n, k] = dims else { return None };
    if !active() {
        return None;
    }
    match Q8_0Kai::from_q8_0(src, n, k) {
        Ok(weight) => Some(super::QStorage::Cpu(Box::new(weight))),
        Err(e) => {
            tracing::warn!("{e}, keeping the plain layout");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::QuantizedType;

    /// Uniform in `[-1, 1)` from a xorshift stream. An arithmetic sequence modulo a prime
    /// makes the int8 rounding errors correlate across `k`, which overstates the error.
    fn uniform(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn blocks(n: usize, k: usize) -> Vec<BlockQ8_0> {
        let mut blocks = vec![BlockQ8_0::zeros(); n * k / QK8_0];
        BlockQ8_0::from_float(&uniform(0x9E37_79B9 + (n * k) as u64, n * k), &mut blocks).unwrap();
        blocks
    }

    fn dequantize(blocks: &[BlockQ8_0]) -> Vec<f32> {
        let mut out = vec![0f32; blocks.len() * QK8_0];
        BlockQ8_0::to_float(blocks, &mut out).unwrap();
        out
    }

    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
        let den: f32 = want.iter().map(|w| w.powi(2)).sum();
        (num / den.max(1e-20)).sqrt()
    }

    #[test]
    fn active_needs_sme2() {
        assert!(!active() || sme2());
    }

    /// Against an f32 matmul over the weights the kernels actually hold, so the only error
    /// left is the activations' int8 rounding. Covers the gemv, partial tiles, and an `n`
    /// below the tile width.
    #[test]
    fn matches_the_f32_reference() {
        if !sme2() {
            return;
        }
        for (m, k, n) in [(1, 512, 512), (1, 64, 12), (4, 128, 16), (7, 96, 20), (33, 256, 68)] {
            let weight = Q8_0Kai::from_q8_0(&blocks(n, k), n, k).unwrap();
            let held = weight.dequantize(n * k).unwrap();
            let lhs = uniform(0xD1B5_4A32 + m as u64, m * k);
            let want: Vec<f32> = (0..m * n)
                .map(|i| (0..k).map(|l| lhs[i / n * k + l] * held[i % n * k + l]).sum())
                .collect();
            let mut got = vec![0f32; m * n];
            weight.matmul_t((m, k, n), &lhs, &mut got).unwrap();
            let err = rel_l2(&got, &want);
            assert!(err < 5e-3, "m={m} k={k} n={n}: rel l2 {err}");
        }
    }

    /// Each stored value is on its row's grid and within half a step of the `q8_0` value.
    #[test]
    fn stores_the_per_row_requantization() {
        if !sme2() {
            return;
        }
        let (n, k) = (12, 96);
        let src = blocks(n, k);
        let held = Q8_0Kai::from_q8_0(&src, n, k).unwrap().dequantize(n * k).unwrap();
        for (r, (held, want)) in held.chunks(k).zip(dequantize(&src).chunks(k)).enumerate() {
            let step = want.iter().fold(0f32, |m, w| m.max(w.abs())) / 127.0;
            for (h, w) in held.iter().zip(want) {
                assert!((h - w).abs() <= step / 2.0 + 1e-6, "row {r}: {h} vs {w}");
                assert!((h / step - (h / step).round()).abs() < 1e-2, "row {r}: {h} off the grid");
            }
        }
    }

    #[test]
    fn raw_data_is_q8_0_of_the_stored_weights() {
        if !sme2() {
            return;
        }
        let (n, k) = (8, 64);
        let weight = Q8_0Kai::from_q8_0(&blocks(n, k), n, k).unwrap();
        let bytes = weight.raw_data().unwrap();
        assert_eq!(bytes.len(), weight.storage_size_in_bytes());
        // SAFETY: `raw_data` returns whole `BlockQ8_0`s.
        let back = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const BlockQ8_0,
                bytes.len() / std::mem::size_of::<BlockQ8_0>(),
            )
        };
        assert!(rel_l2(&dequantize(back), &weight.dequantize(n * k).unwrap()) < 1e-2);
    }

    #[test]
    fn from_float_replaces_the_weights() {
        if !sme2() {
            return;
        }
        let (n, k) = (4, 64);
        let mut weight = Q8_0Kai::from_q8_0(&blocks(n, k), n, k).unwrap();
        let other: Vec<f32> = uniform(7, n * k).iter().map(|v| v * 0.5).collect();
        weight.from_float(&other).unwrap();
        assert!(rel_l2(&weight.dequantize(n * k).unwrap(), &other) < 1e-2);
    }

    /// A weight that cannot be packed is refused, so the hook keeps the plain layout for it.
    #[test]
    fn refuses_what_it_cannot_pack() {
        if !sme2() {
            return;
        }
        let (n, k) = (4, 64);
        assert!(Q8_0Kai::from_q8_0(&blocks(n, k), n, k + QK8_0).is_err(), "wrong block count");
        let mut nan = blocks(n, k);
        nan[3].d = half::f16::NAN;
        assert!(Q8_0Kai::from_q8_0(&nan, n, k).is_err(), "NaN weight");
    }

    #[test]
    fn the_hook_follows_the_gate() {
        let (n, k) = (8, 64);
        assert_eq!(q8_0_storage(&blocks(n, k), &[n, k]).is_some(), active());
        assert!(q8_0_storage(&blocks(n, k), &[n * k]).is_none(), "only 2-d weights");
    }
}
