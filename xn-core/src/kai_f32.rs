//! Arm KleidiAI's SME2 kernels for f32 matmuls, behind the `kai` feature.
//!
//! The arithmetic is f32 multiply-adds like the NEON path's, so the results differ from it
//! only by the order of the sums. Both operands are packed into KleidiAI's layout on every
//! call, into buffers kept per thread. Used when [`crate::quantized::kai::active`] holds and
//! `accelerate` is off, which already runs f32 matmuls on the same unit.
//!
//! A one-row matmul stays on the NEON path: it reads each weight once, so packing the weight
//! first would cost as much as the matmul itself.

use std::cell::RefCell;
use std::ffi::c_void;

unsafe extern "C" {
    fn kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
        m: usize,
        n: usize,
        k: usize,
        lhs_packed: *const c_void,
        rhs_packed: *const c_void,
        dst: *mut c_void,
        dst_stride_row: usize,
        dst_stride_col: usize,
        clamp_min: f32,
        clamp_max: f32,
    );
    fn kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_lhs_pack_f32p2vlx1_f32_sme(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
        m_idx_start: usize,
        lhs: *const c_void,
        lhs_stride_row: usize,
        lhs_packed: *mut c_void,
    );
    fn kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        n: usize,
        k: usize,
    ) -> usize;
    fn kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs_stride_row: usize,
        rhs: *const c_void,
        bias: *const c_void,
        scale: *const c_void,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const c_void,
    );
    fn kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
        n: usize,
        k: usize,
    ) -> usize;
    fn kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs_stride: usize,
        rhs: *const c_void,
        bias: *const c_void,
        scale: *const c_void,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const c_void,
    );
}

/// How the rhs is laid out: `k` rows of `n`, or `n` rows of `k`, each with a row stride.
#[derive(Clone, Copy)]
enum Rhs {
    KxN(usize),
    NxK(usize),
}

#[derive(Default)]
struct Scratch {
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    bias: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

fn grow(buf: &mut Vec<f32>, bytes: usize) -> *mut c_void {
    let len = bytes.div_ceil(4);
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    buf.as_mut_ptr().cast()
}

/// `dst[b] = lhs[b] @ rhs[b]` with the same stride conventions as `gemm_`, element strides
/// given as `(column, row)`. Returns `false`, having done nothing, for a layout or size the
/// kernels do not take.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    (m, n, k): (usize, usize, usize),
    batch: usize,
    (lhs_b_stride, rhs_b_stride): (usize, usize),
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> bool {
    if m < 2 || n == 0 || k == 0 || batch == 0 || !crate::quantized::kai::active() {
        return false;
    }
    if dst_cs != 1 || lhs_cs != 1 {
        return false;
    }
    let layout = if rhs_cs == 1 {
        Rhs::KxN(rhs_rs)
    } else if rhs_rs == 1 {
        Rhs::NxK(rhs_cs)
    } else {
        return false;
    };
    // Every element the kernels read or write has to be inside the slices.
    let lhs_end = (batch - 1) * lhs_b_stride + (m - 1) * lhs_rs + k;
    let rhs_end = (batch - 1) * rhs_b_stride
        + match layout {
            Rhs::KxN(s) => (k - 1) * s + n,
            Rhs::NxK(s) => (n - 1) * s + k,
        };
    let dst_end = (batch - 1) * m * n + (m - 1) * dst_rs + n;
    if lhs.len() < lhs_end || rhs.len() < rhs_end || dst.len() < dst_end || dst_rs < n {
        return false;
    }

    SCRATCH.with_borrow_mut(|s| {
        // SAFETY: the getters have no preconditions; the packers write at most the sizes their
        // own queries return, into buffers grown to that size; and every operand offset was
        // checked against its slice above.
        unsafe {
            let mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let f = std::mem::size_of::<f32>();
            // The packed rhs carries a bias per column, which is zero here.
            if s.bias.len() < n {
                s.bias.resize(n, 0.0);
            }
            let bias = s.bias.as_ptr().cast();
            let lhs_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
            let lhs_packed = grow(&mut s.lhs, lhs_size);
            let rhs_size = match layout {
                Rhs::KxN(_) => {
                    kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k)
                }
                Rhs::NxK(_) => {
                    kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k)
                }
            };
            let rhs_packed = grow(&mut s.rhs, rhs_size);
            let pack_rhs = |rhs: *const f32| match layout {
                Rhs::KxN(st) => kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                    1,
                    n,
                    k,
                    nr,
                    kr,
                    sr,
                    st * f,
                    rhs.cast(),
                    bias,
                    std::ptr::null(),
                    rhs_packed,
                    0,
                    std::ptr::null(),
                ),
                Rhs::NxK(st) => kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
                    1,
                    n,
                    k,
                    nr,
                    kr,
                    sr,
                    st * f,
                    rhs.cast(),
                    bias,
                    std::ptr::null(),
                    rhs_packed,
                    0,
                    std::ptr::null(),
                ),
            };
            for b in 0..batch {
                if b == 0 || rhs_b_stride != 0 {
                    pack_rhs(rhs.as_ptr().add(b * rhs_b_stride));
                }
                kai_run_lhs_pack_f32p2vlx1_f32_sme(
                    m,
                    k,
                    mr,
                    kr,
                    sr,
                    0,
                    lhs.as_ptr().add(b * lhs_b_stride).cast(),
                    lhs_rs * f,
                    lhs_packed,
                );
                kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                    m,
                    n,
                    k,
                    lhs_packed,
                    rhs_packed,
                    dst.as_mut_ptr().add(b * m * n).cast(),
                    dst_rs * f,
                    f,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                );
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// `lhs` row-major with row stride `lhs_rs`; `rhs[(p, j)]` at `p * rs.1 + j * rs.0`.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        lhs: &[f32],
        rhs: &[f32],
        (m, n, k): (usize, usize, usize),
        batch: usize,
        (lb, rb): (usize, usize),
        dst_rs: usize,
        lhs_rs: usize,
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Vec<f32> {
        let mut dst = vec![0f32; (batch - 1) * m * n + (m - 1) * dst_rs + n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    dst[b * m * n + i * dst_rs + j] = (0..k)
                        .map(|p| {
                            let r = rhs[b * rb + p * rhs_rs + j * rhs_cs] as f64;
                            lhs[b * lb + i * lhs_rs + p] as f64 * r
                        })
                        .sum::<f64>() as f32;
                }
            }
        }
        dst
    }

    /// Both rhs layouts, partial tiles in `m` and `n`, padded lhs and dst rows, and a batch
    /// with a shared and with a per-batch rhs.
    #[test]
    fn matches_the_reference() {
        let cases = [
            // (m, n, k, batch, rhs per batch, kxn, lhs pad, dst pad)
            (2, 3, 1, 1, false, true, 0, 0),
            (16, 2048, 512, 1, false, false, 0, 0),
            (17, 70, 33, 1, false, true, 5, 3),
            (96, 128, 64, 1, false, false, 0, 0),
            (480, 64, 384, 1, false, true, 0, 0),
            (16, 266, 64, 3, true, true, 0, 0),
            (16, 64, 266, 3, true, false, 0, 0),
            (5, 40, 24, 4, false, false, 2, 0),
        ];
        for (m, n, k, batch, per_batch, kxn, lhs_pad, dst_pad) in cases {
            let lhs_rs = k + lhs_pad;
            let dst_rs = n + dst_pad;
            let (rhs_cs, rhs_rs) = if kxn { (1, n) } else { (k, 1) };
            let rb = if per_batch { n * k } else { 0 };
            let lb = m * lhs_rs;
            let lhs = uniform(0xA076_1D64 + (m * n) as u64, batch * lb);
            let rhs =
                uniform(0xE703_7ED1 + (n * k) as u64, n * k * if per_batch { batch } else { 1 });
            let want =
                reference(&lhs, &rhs, (m, n, k), batch, (lb, rb), dst_rs, lhs_rs, (rhs_cs, rhs_rs));
            let mut got = vec![f32::NAN; want.len()];
            let ran = gemm(
                &mut got,
                &lhs,
                &rhs,
                (m, n, k),
                batch,
                (lb, rb),
                (1, dst_rs),
                (1, lhs_rs),
                (rhs_cs, rhs_rs),
            );
            if !crate::quantized::kai::active() {
                assert!(!ran);
                continue;
            }
            assert!(ran, "m={m} n={n} k={k}");
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                let (b, r) = (idx / (m * n), idx % (m * n));
                if r / dst_rs >= m || r % dst_rs >= n {
                    continue;
                }
                let tol = 1e-5 * (k as f32).sqrt().max(1.0);
                assert!((g - w).abs() <= tol, "m={m} n={n} k={k} b={b} at {idx}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn declines_what_it_does_not_take() {
        let (lhs, rhs, mut dst) = (vec![1f32; 64], vec![1f32; 64], vec![0f32; 64]);
        let run =
            |dst: &mut [f32], m, d, l, r| gemm(dst, &lhs, &rhs, (m, 4, 4), 1, (0, 0), d, l, r);
        // One row, a strided lhs column, an rhs strided both ways, a strided dst column.
        assert!(!run(&mut dst, 1, (1, 4), (1, 4), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (2, 8), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (1, 4), (2, 8)));
        assert!(!run(&mut dst, 4, (2, 8), (1, 4), (1, 4)));
        // A dst too short for the rows.
        assert!(!run(&mut dst[..10], 4, (1, 4), (1, 4), (1, 4)));
        assert!(!dst.iter().any(|&x| x != 0.0));
    }
}
