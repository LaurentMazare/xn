//! Direct convolution and gemm kernels for the wasm SIMD target.
//!
//! The generic path turns a `conv1d` into im2col, a packed gemm and a transpose. These
//! kernels read the operands where they are, with a 4x16 register tile and a narrow tile for
//! the rows and columns left over.
//!
//! * [`conv1d`] vectorizes over time: with stride 1 a tap's input samples are contiguous, the
//!   weight is a broadcast, and the output row is written in place.
//! * [`gemm_bcast_lhs`] is `x @ W` on a row-major weight.
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

/// `R` rows by `4 * V` columns of a broadcast-lhs product.
///
/// # Safety
/// Each `lhs[r]` addresses `k` floats, `rhs` addresses `k` rows of at least `4 * V` floats
/// `ld` apart, and each `out[r]` addresses `4 * V` floats.
#[inline(always)]
unsafe fn bcast_tile<const R: usize, const V: usize>(
    lhs: [*const f32; R],
    rhs: *const f32,
    ld: usize,
    k: usize,
    out: [*mut f32; R],
) {
    unsafe {
        let mut acc = [[f32x4_splat(0.0); V]; R];
        for kk in 0..k {
            let bp = rhs.add(kk * ld);
            let mut bv = [f32x4_splat(0.0); V];
            for v in 0..V {
                bv[v] = v128_load(bp.add(4 * v) as *const v128);
            }
            for r in 0..R {
                let s = v128_load32_splat(lhs[r].add(kk) as *const u32);
                for v in 0..V {
                    acc[r][v] = madd(s, bv[v], acc[r][v]);
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

/// `out[i, j] = sum_kk lhs[i * lhs_rs + kk] * rhs[kk * rhs_rs + j]`, `out[i, j]` at
/// `out[i * out_rs + j]`. The left operand is only ever broadcast; the right one is read
/// sixteen contiguous floats at a time.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bcast_lhs(
    out: &mut [f32],
    out_rs: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_rs: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(rhs.len() >= (k.max(1) - 1) * rhs_rs + n);
    assert!(out.len() >= (m - 1) * out_rs + n);
    assert!(lhs.len() >= (m - 1) * lhs_rs + k);
    let (o, l, r) = (out.as_mut_ptr() as usize, lhs.as_ptr() as usize, rhs.as_ptr() as usize);
    // Output columns `[j_lo, j_hi)`.
    let job = move |j_lo: usize, j_hi: usize| {
        let (out, lhs, rhs) = (o as *mut f32, l as *const f32, r as *const f32);
        // SAFETY: the bounds asserted above, and units write disjoint columns.
        unsafe {
            for j0 in (j_lo..j_hi).step_by(16) {
                let width = (j_hi - j0).min(16);
                let lp = |i: usize| lhs.add(i * lhs_rs);
                let op = |i: usize| out.add(i * out_rs + j0);
                let rp = rhs.add(j0);
                for i0 in (0..m).step_by(4) {
                    let rows = (m - i0).min(4);
                    if rows == 4 && width == 16 {
                        let (ls, os) = (from_fn(|t| lp(i0 + t)), from_fn(|t| op(i0 + t)));
                        bcast_tile::<4, 4>(ls, rp, rhs_rs, k, os);
                        continue;
                    }
                    for i in i0..i0 + rows {
                        let mut j = 0;
                        while j + 4 <= width {
                            bcast_tile::<1, 1>([lp(i)], rp.add(j), rhs_rs, k, [op(i).add(j)]);
                            j += 4;
                        }
                        for j in j..width {
                            let mut s = 0f32;
                            for kk in 0..k {
                                s += *lp(i).add(kk) * *rp.add(kk * rhs_rs + j);
                            }
                            *op(i).add(j) = s;
                        }
                    }
                }
            }
        }
    };
    crate::threadpool::par_units_by(m * n * k, n, 16, job);
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
