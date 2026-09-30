//! Direct convolution and gemm kernels for the wasm SIMD target.
//!
//! The generic path turns a `conv1d` into im2col, a packed gemm and a transpose. These
//! kernels read the operands where they are, with a 4x16 register tile and a narrow tile for
//! the rows and columns left over.
//!
//! * [`conv1d`] vectorizes over time: with stride 1 a tap's input samples are contiguous, the
//!   weight is a broadcast, and the output row is written in place.
//! * [`gemm_dot`] is `x @ W^T` on a row-major weight.

use crate::simd128_math::madd;
use core::arch::wasm32::*;
use std::array::from_fn;

/// `R` output channels by `4 * V` time steps of a stride-1 convolution.
///
/// # Safety
/// `x` addresses `ci` rows of `l_in` floats starting at the tile's first input sample,
/// each `w[r]` addresses `ci * k` floats, and each `out[r]` addresses `4 * V` floats.
#[inline(always)]
unsafe fn conv_tile<const R: usize, const V: usize>(
    x: *const f32,
    l_in: usize,
    ci: usize,
    k: usize,
    dilation: usize,
    w: [*const f32; R],
    out: [*mut f32; R],
) {
    unsafe {
        let mut acc = [[f32x4_splat(0.0); V]; R];
        for c in 0..ci {
            let xrow = x.add(c * l_in);
            for t in 0..k {
                let xp = xrow.add(t * dilation);
                let mut xv = [f32x4_splat(0.0); V];
                for v in 0..V {
                    xv[v] = v128_load(xp.add(4 * v) as *const v128);
                }
                for r in 0..R {
                    let s = v128_load32_splat(w[r].add(c * k + t) as *const u32);
                    for v in 0..V {
                        acc[r][v] = madd(s, xv[v], acc[r][v]);
                    }
                }
            }
        }
        for r in 0..R {
            for v in 0..V {
                v128_store(out[r].add(4 * v) as *mut v128, acc[r][v]);
            }
        }
    }
}

/// `dst[b, co, l] = sum_{ci, t} w[co, ci, t] * src[b, ci, l + t * dilation]`; stride 1, no
/// padding, one group. `src` is `[batch, ci, l_in]`, `w` is `[co, ci, k]`, `dst` is
/// `[batch, co, l_out]`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    batch: usize,
    ci: usize,
    co: usize,
    l_in: usize,
    l_out: usize,
    k: usize,
    dilation: usize,
) {
    assert!(src.len() >= batch * ci * l_in);
    assert!(w.len() >= co * ci * k);
    assert!(dst.len() >= batch * co * l_out);
    assert!(l_out == 0 || (l_out - 1) + (k - 1) * dilation < l_in);
    let (d, s, wt) = (dst.as_mut_ptr() as usize, src.as_ptr() as usize, w.as_ptr() as usize);
    // Output channels `[c_lo, c_hi)` of every batch element.
    let job = move |c_lo: usize, c_hi: usize| {
        let (dst, src, w) = (d as *mut f32, s as *const f32, wt as *const f32);
        // SAFETY: the shapes asserted above bound every access, and units write disjoint channels.
        unsafe {
            for b in 0..batch {
                let x = src.add(b * ci * l_in);
                let y = dst.add(b * co * l_out);
                // Time outermost so a 16-sample window of every input channel stays in L1 while
                // the output channels sweep over it.
                for l0 in (0..l_out).step_by(16) {
                    let width = (l_out - l0).min(16);
                    let xt = x.add(l0);
                    let wp = |c: usize| w.add(c * ci * k);
                    let op = |c: usize| y.add(c * l_out + l0);
                    for c0 in (c_lo..c_hi).step_by(4) {
                        let rows = (c_hi - c0).min(4);
                        if rows == 4 && width == 16 {
                            let (ws, os) = (from_fn(|r| wp(c0 + r)), from_fn(|r| op(c0 + r)));
                            conv_tile::<4, 4>(xt, l_in, ci, k, dilation, ws, os);
                            continue;
                        }
                        for c in c0..c0 + rows {
                            let mut l = 0;
                            while l + 4 <= width {
                                let (ws, os) = ([wp(c)], [op(c).add(l)]);
                                conv_tile::<1, 1>(xt.add(l), l_in, ci, k, dilation, ws, os);
                                l += 4;
                            }
                            for l in l..width {
                                let mut s = 0f32;
                                for cc in 0..ci {
                                    for t in 0..k {
                                        s += *wp(c).add(cc * k + t)
                                            * *xt.add(cc * l_in + l + t * dilation);
                                    }
                                }
                                *op(c).add(l) = s;
                            }
                        }
                    }
                }
            }
        }
    };
    crate::threadpool::par_units_by(batch * co * l_out * ci * k, co, 4, job);
}

/// `R` lhs rows against `C` rhs rows, `k` deep; writes `out[r][0..C]`.
///
/// # Safety
/// Every pointer addresses `k` floats; each `out[r]` addresses `C` floats.
#[inline(always)]
unsafe fn dot_tile<const R: usize, const C: usize>(
    lhs: [*const f32; R],
    rhs: [*const f32; C],
    k: usize,
    out: [*mut f32; R],
) {
    unsafe {
        let mut acc = [[f32x4_splat(0.0); C]; R];
        let k4 = k / 4 * 4;
        for kk in (0..k4).step_by(4) {
            let mut a = [f32x4_splat(0.0); R];
            for r in 0..R {
                a[r] = v128_load(lhs[r].add(kk) as *const v128);
            }
            for c in 0..C {
                let b = v128_load(rhs[c].add(kk) as *const v128);
                for r in 0..R {
                    acc[r][c] = madd(a[r], b, acc[r][c]);
                }
            }
        }
        for r in 0..R {
            for c in 0..C {
                let v = acc[r][c];
                let mut s = f32x4_extract_lane::<0>(v)
                    + f32x4_extract_lane::<1>(v)
                    + f32x4_extract_lane::<2>(v)
                    + f32x4_extract_lane::<3>(v);
                for t in k4..k {
                    s += *lhs[r].add(t) * *rhs[c].add(t);
                }
                *out[r].add(c) = s;
            }
        }
    }
}

/// `out[i, j] = sum_kk lhs[i * lhs_rs + kk] * rhs[j * rhs_cs + kk]`, `out[i, j]` at
/// `out[i * out_rs + j]`: both operands contiguous along `k`, as `x @ W^T` on a row-major
/// weight is.
#[allow(clippy::too_many_arguments)]
pub fn gemm_dot(
    out: &mut [f32],
    out_rs: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(out.len() >= (m - 1) * out_rs + n);
    assert!(lhs.len() >= (m - 1) * lhs_rs + k);
    assert!(rhs.len() >= (n - 1) * rhs_cs + k);
    let (o, l, r) = (out.as_mut_ptr() as usize, lhs.as_ptr() as usize, rhs.as_ptr() as usize);
    // Output columns `[j_lo, j_hi)`.
    let job = move |j_lo: usize, j_hi: usize| {
        let (out, lhs, rhs) = (o as *mut f32, l as *const f32, r as *const f32);
        // SAFETY: the bounds asserted above, and units write disjoint columns.
        unsafe {
            let lp = |i: usize| lhs.add(i * lhs_rs);
            let rp = |j: usize| rhs.add(j * rhs_cs);
            let op = |i: usize, j: usize| out.add(i * out_rs + j);
            for i0 in (0..m).step_by(4) {
                let rows = (m - i0).min(4);
                let mut j = j_lo;
                if rows == 4 {
                    while j + 4 <= j_hi {
                        let (ls, rs) = (from_fn(|t| lp(i0 + t)), from_fn(|c| rp(j + c)));
                        dot_tile::<4, 4>(ls, rs, k, from_fn(|t| op(i0 + t, j)));
                        j += 4;
                    }
                }
                // Leftover rows and columns: one row against eight columns, then single ones.
                for i in i0..i0 + rows {
                    let mut j = j;
                    while j + 8 <= j_hi {
                        dot_tile::<1, 8>([lp(i)], from_fn(|c| rp(j + c)), k, [op(i, j)]);
                        j += 8;
                    }
                    for j in j..j_hi {
                        dot_tile::<1, 1>([lp(i)], [rp(j)], k, [op(i, j)]);
                    }
                }
            }
        }
    };
    crate::threadpool::par_units_by(m * n * k, n, 32, job);
}

#[cfg(test)]
mod tests {
    //! Run under wasmtime: `cargo test -p xn --lib --target wasm32-wasip1 -- simd128` with
    //! `-C target-feature=+simd128`. Shapes are chosen to hit every tile and tail branch.
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{what}[{i}]: got {g}, want {w}");
        }
    }

    #[test]
    fn conv1d_matches_the_scalar_reference() {
        for batch in [1, 2] {
            for (ci, co) in [(1, 1), (3, 5), (5, 4), (2, 17)] {
                for (k, dilation) in [(1, 1), (3, 1), (3, 2), (7, 1)] {
                    for l_out in [1, 5, 16, 17, 33, 48] {
                        let l_in = l_out + (k - 1) * dilation;
                        let x = data(batch * ci * l_in, 1);
                        let w = data(co * ci * k, 2);
                        let mut got = vec![0f32; batch * co * l_out];
                        conv1d(&mut got, &x, &w, batch, ci, co, l_in, l_out, k, dilation);
                        let mut want = vec![0f32; batch * co * l_out];
                        for b in 0..batch {
                            for o in 0..co {
                                for l in 0..l_out {
                                    let mut acc = 0f32;
                                    for c in 0..ci {
                                        for t in 0..k {
                                            acc += w[o * ci * k + c * k + t]
                                                * x[b * ci * l_in + c * l_in + l + t * dilation];
                                        }
                                    }
                                    want[b * co * l_out + o * l_out + l] = acc;
                                }
                            }
                        }
                        assert_close(
                            &got,
                            &want,
                            &format!("conv {batch}x{ci}->{co} k{k} d{dilation} L{l_out}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gemm_dot_matches_the_scalar_reference() {
        // `k == 0` writes zeros; `k % 4 != 0` exercises the scalar tail of each dot product.
        for (m, n, k) in [
            (3, 5, 0),
            (1, 1, 1),
            (1, 8, 4),
            (1, 9, 5),
            (3, 3, 3),
            (4, 4, 4),
            (5, 7, 6),
            (4, 17, 33),
            (16, 12, 7),
            (9, 40, 16),
        ] {
            let lhs = data(m * k, 5);
            let rhs = data(n * k, 6);
            let mut got = vec![0f32; m * n];
            gemm_dot(&mut got, n, &lhs, k, &rhs, k, m, n, k);
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[i * n + j] = (0..k).map(|t| lhs[i * k + t] * rhs[j * k + t]).sum();
                }
            }
            assert_close(&got, &want, &format!("dot {m}x{n}x{k}"));
        }
    }

    /// A batched input against one 2-D weight, through the backend: the weight's batch
    /// stride is 0, which the kernels only see from `gemm_`.
    #[test]
    fn batched_matmul_with_a_broadcast_weight() {
        use crate::{CPU, CpuDevice, Tensor};
        let (batch, m, n, k) = (3, 5, 20, 9);
        let (xv, wv) = (data(batch * m * k, 7), data(n * k, 8));
        let x = Tensor::<f32, CpuDevice>::from_vec(xv.clone(), (batch, m, k), &CPU).unwrap();
        let w = Tensor::<f32, CpuDevice>::from_vec(wv.clone(), (n, k), &CPU).unwrap();
        let got = x.matmul_t(&w).unwrap().to_vec().unwrap();
        let mut want = vec![0f32; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let x_row = &xv[(b * m + i) * k..(b * m + i + 1) * k];
                    let w_row = &wv[j * k..(j + 1) * k];
                    want[(b * m + i) * n + j] = x_row.iter().zip(w_row).map(|(a, c)| a * c).sum();
                }
            }
        }
        assert_close(&got, &want, "batched matmul_t");
    }
}
