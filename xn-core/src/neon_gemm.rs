//! A direct `x @ W^T` kernel for aarch64.
//!
//! xn-gemm packs its operands into panels before it multiplies, which pays off when both sides
//! have many rows. A linear layer applied to a few positions multiplies a handful of rows by a
//! weight it reads once, and there the packing and blocking cost about as much as the multiply.
//! This kernel reads the operands where they are: both are contiguous along `k`, so every output
//! is a dot product. A tile of `R` rows by `C` columns keeps one vector accumulator per output
//! and reduces across lanes once, at the end.
//!
//! The weight's columns are the outer loop and the rows of `x` the inner one, so each weight row
//! is read from memory once and then reused from L1 for every row of `x`.

use core::arch::aarch64::*;
use std::array::from_fn;

/// `k` is taken in pieces of this many, and the rows in blocks whose pieces fit in
/// `BLOCK_BYTES`: the block stays in L1 while the weight's columns sweep over it, and each piece
/// of a weight row is read from memory once for the whole block.
const KC: usize = 512;
const BLOCK_BYTES: usize = 32 * 1024;

/// Below this many multiply-adds a matmul runs on the calling thread: handing it to the pool
/// costs more than it saves.
const PAR_WORK: usize = 1 << 16;

/// `R` lhs rows against `C` rhs rows, `k` deep; writes `out[r][0..C]`, or adds to it when
/// `add`.
///
/// # Safety
/// Every `lhs` and `rhs` pointer addresses `k` floats; each `out[r]` addresses `C` floats.
#[inline(always)]
unsafe fn dot_tile<const R: usize, const C: usize>(
    lhs: [*const f32; R],
    rhs: [*const f32; C],
    k: usize,
    out: [*mut f32; R],
    add: bool,
) {
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); C]; R];
        let k4 = k / 4 * 4;
        let mut t = 0;
        while t < k4 {
            let a: [float32x4_t; R] = from_fn(|r| vld1q_f32(lhs[r].add(t)));
            for c in 0..C {
                let b = vld1q_f32(rhs[c].add(t));
                for r in 0..R {
                    acc[r][c] = vfmaq_f32(acc[r][c], a[r], b);
                }
            }
            t += 4;
        }
        let tail = |r: usize, c: usize| -> f32 {
            let mut s = 0f32;
            for t in k4..k {
                s += *lhs[r].add(t) * *rhs[c].add(t);
            }
            s
        };
        for r in 0..R {
            let mut c = 0;
            // Four columns at a time: two rounds of pairwise adds leave their four sums in one
            // vector, in column order.
            while c + 4 <= C {
                let lo = vpaddq_f32(acc[r][c], acc[r][c + 1]);
                let hi = vpaddq_f32(acc[r][c + 2], acc[r][c + 3]);
                let mut s = vpaddq_f32(lo, hi);
                if k4 < k {
                    let t: [f32; 4] = from_fn(|cc| tail(r, c + cc));
                    s = vaddq_f32(s, vld1q_f32(t.as_ptr()));
                }
                if add {
                    s = vaddq_f32(s, vld1q_f32(out[r].add(c)));
                }
                vst1q_f32(out[r].add(c), s);
                c += 4;
            }
            for (c, &v) in acc[r].iter().enumerate().skip(c) {
                let s = vaddvq_f32(v) + tail(r, c);
                *out[r].add(c) = if add { *out[r].add(c) + s } else { s };
            }
        }
    }
}

/// Rows `i0..i0 + R` against columns `j..j + w`, `w <= 8`, written or added to as `add` says.
///
/// # Safety
/// The rows and columns are in bounds for the pointers' strides, as [`gemm_dot`] asserts.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn row_tile<const R: usize>(
    out: *mut f32,
    out_rs: usize,
    lhs: *const f32,
    lhs_rs: usize,
    rhs: *const f32,
    rhs_cs: usize,
    k: usize,
    i0: usize,
    j: usize,
    w: usize,
    add: bool,
) {
    unsafe {
        let lp: [*const f32; R] = from_fn(|r| lhs.add((i0 + r) * lhs_rs));
        let op = |c: usize| -> [*mut f32; R] { from_fn(|r| out.add((i0 + r) * out_rs + j + c)) };
        let rp = |c: usize| rhs.add((j + c) * rhs_cs);
        // One or two rows give too few independent accumulators across four columns to keep
        // the multiply-add pipes busy, so they take eight at once.
        if R <= 2 && w == 8 {
            dot_tile::<R, 8>(lp, from_fn(rp), k, op(0), add);
            return;
        }
        let mut c = 0;
        while c + 4 <= w {
            dot_tile::<R, 4>(lp, from_fn(|cc| rp(c + cc)), k, op(c), add);
            c += 4;
        }
        for c in c..w {
            dot_tile::<R, 1>(lp, [rp(c)], k, op(c), add);
        }
    }
}

/// How much of `k` one piece takes, for `m` rows: all of it when every row fits in the budget,
/// as one or a few rows do, so that each weight row is one long sequential read; otherwise `KC`.
///
/// Where the pieces fall decides the order of each dot product's sum, so it depends on the
/// whole matmul and not on the part a thread gets: any split gives the same bits.
fn piece(m: usize, k: usize) -> usize {
    if m * k * 4 <= BLOCK_BYTES { k.max(1) } else { KC.min(k) }
}

/// Columns `[j_lo, j_hi)` of [`gemm_dot`] for `m` rows, `k` in pieces of `kc`, on the calling
/// thread.
///
/// # Safety
/// The bounds [`gemm_dot`] asserts, with `j_hi <= n`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_dot_cols(
    out: *mut f32,
    out_rs: usize,
    lhs: *const f32,
    lhs_rs: usize,
    rhs: *const f32,
    rhs_cs: usize,
    m: usize,
    k: usize,
    kc: usize,
    j_lo: usize,
    j_hi: usize,
) {
    let block = (BLOCK_BYTES / (4 * kc)).max(4) / 4 * 4;
    for i_lo in (0..m).step_by(block) {
        let i_hi = (i_lo + block).min(m);
        // At least one piece, so that `k == 0` still writes its zeros.
        for p in 0..k.div_ceil(kc).max(1) {
            let (k0, add) = (p * kc, p > 0);
            let kb = kc.min(k - k0);
            // SAFETY: `k0 + kb <= k`.
            let (l, r) = unsafe { (lhs.add(k0), rhs.add(k0)) };
            for j in (j_lo..j_hi).step_by(8) {
                let w = (j_hi - j).min(8);
                for i0 in (i_lo..i_hi).step_by(4) {
                    let o = out;
                    // SAFETY: rows `i0..i0 + 4` (or to `m`) and columns `j..j + w` are in bounds.
                    unsafe {
                        match (i_hi - i0).min(4) {
                            4 => row_tile::<4>(o, out_rs, l, lhs_rs, r, rhs_cs, kb, i0, j, w, add),
                            3 => row_tile::<3>(o, out_rs, l, lhs_rs, r, rhs_cs, kb, i0, j, w, add),
                            2 => row_tile::<2>(o, out_rs, l, lhs_rs, r, rhs_cs, kb, i0, j, w, add),
                            _ => row_tile::<1>(o, out_rs, l, lhs_rs, r, rhs_cs, kb, i0, j, w, add),
                        }
                    }
                }
            }
        }
    }
}

/// `out[i, j] = sum_kk lhs[i * lhs_rs + kk] * rhs[j * rhs_cs + kk]`, `out[i, j]` at
/// `out[i * out_rs + j]`: both operands contiguous along `k`, as `x @ W^T` on a row-major
/// weight is. Large enough matmuls split over the thread pool.
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
    assert!(out_rs >= n && out.len() >= (m - 1) * out_rs + n);
    assert!(lhs.len() >= (m - 1) * lhs_rs + k);
    assert!(rhs.len() >= (n - 1) * rhs_cs + k);
    let (o, l, r) = (out.as_mut_ptr() as usize, lhs.as_ptr() as usize, rhs.as_ptr() as usize);
    let kc = piece(m, k);
    // Rows `[i_lo, i_hi)` by columns `[j_lo, j_hi)`.
    let job = move |i_lo: usize, i_hi: usize, j_lo: usize, j_hi: usize| {
        // SAFETY: the bounds asserted above, and units write disjoint parts of `out`.
        unsafe {
            let out = (o as *mut f32).add(i_lo * out_rs);
            let (lhs, rhs) = ((l as *const f32).add(i_lo * lhs_rs), r as *const f32);
            gemm_dot_cols(out, out_rs, lhs, lhs_rs, rhs, rhs_cs, i_hi - i_lo, k, kc, j_lo, j_hi)
        }
    };
    // Split the longer side in pieces of 16: the columns of a wide weight, or the rows when the
    // weight is narrow and there are more of them.
    let work = m * n * k;
    if work < PAR_WORK {
        job(0, m, 0, n);
    } else if n >= m {
        crate::threadpool::par_units_by(work, n, 16, |j_lo, j_hi| job(0, m, j_lo, j_hi));
    } else {
        crate::threadpool::par_units_by(work, m, 16, |i_lo, i_hi| job(i_lo, i_hi, 0, n));
    }
}

/// The batch of a batched [`gemm_dot`]: `n` entries, entry `b` reading `lhs[b * self.lhs..]`
/// and `rhs[b * self.rhs..]` and writing the `m * n` floats of output from `b * m * n` on.
#[derive(Clone, Copy)]
pub struct Batch {
    pub n: usize,
    pub lhs: usize,
    pub rhs: usize,
}

/// [`gemm_dot`] over a batch, such as one product per attention head. Entries too small to be
/// split themselves go to the pool whole, so a batch of small products still uses the threads;
/// each entry is computed the same way either way, so the bits do not depend on the split.
#[allow(clippy::too_many_arguments)]
pub fn gemm_dot_batched(
    out: &mut [f32],
    out_rs: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    m: usize,
    n: usize,
    k: usize,
    batch: Batch,
) {
    let entry = m * n * k;
    let one = |out: &mut [f32], b: usize| {
        let (lhs, rhs) = (&lhs[b * batch.lhs..], &rhs[b * batch.rhs..]);
        gemm_dot(out, out_rs, lhs, lhs_rs, rhs, rhs_cs, m, n, k)
    };
    if batch.n <= 1 || entry >= PAR_WORK || entry * batch.n < PAR_WORK {
        for b in 0..batch.n {
            one(&mut out[b * m * n..(b + 1) * m * n], b);
        }
        return;
    }
    assert!(out.len() >= batch.n * m * n);
    let o = out.as_mut_ptr() as usize;
    crate::threadpool::par_units_by(entry * batch.n, batch.n, 1, |b_lo, b_hi| {
        for b in b_lo..b_hi {
            // SAFETY: inside `out`, as asserted above, and entries write disjoint blocks.
            let out =
                unsafe { std::slice::from_raw_parts_mut((o as *mut f32).add(b * m * n), m * n) };
            one(out, b);
        }
    });
}

#[cfg(test)]
mod tests {
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

    /// `gemm_dot` with padded rows on every side, against a scalar reference; the padding of
    /// `out` must come back untouched.
    fn check(m: usize, n: usize, k: usize, pad: usize) {
        let (out_rs, lhs_rs, rhs_cs) = (n + pad, k + pad, k + 2 * pad);
        let lhs = data(m * lhs_rs, 5);
        let rhs = data(n * rhs_cs, 6);
        let mut got = vec![f32::NAN; m * out_rs];
        gemm_dot(&mut got, out_rs, &lhs, lhs_rs, &rhs, rhs_cs, m, n, k);
        let mut want = vec![f32::NAN; m * out_rs];
        for i in 0..m {
            for j in 0..n {
                want[i * out_rs + j] =
                    (0..k).map(|t| lhs[i * lhs_rs + t] * rhs[j * rhs_cs + t]).sum();
            }
        }
        let what = format!("dot {m}x{n}x{k} pad {pad}");
        for i in 0..m {
            let row = i * out_rs;
            assert_close(&got[row..row + n], &want[row..row + n], &what);
            assert!(got[row + n..row + out_rs].iter().all(|v| v.is_nan()), "{what}: padding");
        }
    }

    #[test]
    fn gemm_dot_matches_the_scalar_reference() {
        // Every row count from 1 to 9 reaches each row tile and the leftovers after full ones;
        // column counts around 4 and 8 reach the column tails; `k == 0` writes zeros and
        // `k % 4 != 0` the scalar tail of each dot product.
        for m in 1..=9 {
            for n in [1, 3, 4, 5, 8, 9, 13, 17] {
                for k in [0, 1, 3, 4, 7, 16, 33] {
                    check(m, n, k, 0);
                }
            }
        }
    }

    #[test]
    fn gemm_dot_with_strided_rows() {
        for (m, n, k) in [(1, 9, 5), (4, 8, 16), (5, 7, 6), (7, 20, 33), (16, 12, 7)] {
            for pad in [1, 3, 8] {
                check(m, n, k, pad);
            }
        }
    }

    /// The direct kernel against xn-gemm, without and with its packed weight cache, and with
    /// the `accelerate` feature against Accelerate's sgemm: one thread, then the pool's width
    /// (`RAYON_NUM_THREADS`) as the backend splits each. Interleaved, best of `BENCH_ROUNDS`;
    /// `BENCH_FROM` skips that many shapes.
    /// `cargo test --release -p xn --lib -- --ignored --nocapture neon_gemm::tests::bench`.
    #[test]
    #[ignore]
    fn bench_against_the_generic_gemm() {
        use std::time::Instant;
        let from = std::env::var("BENCH_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let shapes = [
            // A linear layer applied to a few positions.
            (16, 1536, 512),
            (16, 512, 512),
            (16, 2048, 512),
            (16, 512, 2048),
            // One head of attention scores.
            (16, 266, 64),
            // Convolutions through im2col.
            (16, 512, 3584),
            (96, 128, 768),
            (96, 256, 128),
            (480, 64, 384),
            (480, 128, 64),
            (1920, 64, 32),
            (1920, 32, 192),
            (1920, 1, 192),
            // A linear layer applied to one position.
            (1, 512, 512),
            (1, 1536, 512),
            (1, 512, 256),
            (1, 1024, 512),
            (1, 32, 512),
            (1, 512, 32),
            (1, 512, 768),
            (1, 768, 32),
            (1, 1, 768),
            (1, 2304, 768),
            (1, 3072, 768),
            (1, 768, 3072),
            (1, 768, 768),
            // A prompt's worth of positions.
            (25, 2304, 768),
            (25, 768, 3072),
            (125, 2304, 768),
            (125, 3072, 768),
            (125, 768, 3072),
            (125, 768, 768),
            (125, 125, 64),
            (25, 150, 64),
            (32, 768, 16),
            (64, 768, 768),
            (256, 768, 768),
            (512, 2304, 768),
            (1024, 768, 768),
            (2048, 512, 512),
        ];
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
            }
            // QOS_CLASS_USER_INTERACTIVE, so that the thread prefers a performance core.
            unsafe { pthread_set_qos_class_self_np(0x21, 0) };
        }
        let rounds = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let nth = crate::threadpool::size();
        let names = ["direct", "gemm", "gemm+c", "accel", "direct-mt", "gemm-mt", "gemm+c-mt"];
        let shown: Vec<usize> = (0..names.len())
            .filter(|&v| (v != 3 || cfg!(feature = "accelerate")) && (v < 4 || nth > 1))
            .collect();
        print!("\n{:>14}", "m x n x k");
        shown.iter().for_each(|&v| print!(" {:>9}", names[v]));
        println!("   us per call, best of {rounds}, -mt on {nth} threads");
        for (m, n, k) in shapes.into_iter().skip(from) {
            let (lhs, rhs) = (data(m * k, 1), data(n * k, 2));
            let mut out = vec![0f32; m * n];
            let (l, r) = (lhs.as_ptr() as usize, rhs.as_ptr() as usize);
            // Columns `n0..n1` through xn-gemm, as the backend calls it.
            let generic_cols = move |o: usize, n0: usize, n1: usize| unsafe {
                let (o, l, r) =
                    ((o as *mut f32).add(n0), l as *const f32, (r as *const f32).add(n0 * k));
                let (ni, ki) = (n as isize, k as isize);
                let p = gemm::Parallelism::None;
                gemm::gemm(
                    m,
                    n1 - n0,
                    k,
                    o,
                    1,
                    ni,
                    false,
                    l,
                    1,
                    ki,
                    r,
                    ki,
                    1,
                    0f32,
                    1f32,
                    false,
                    false,
                    false,
                    p,
                )
            };
            // The backend's column stripes over the pool.
            let generic_mt = move |o: usize| {
                if n >= nth * 4 && m * n * k >= 1 << 14 {
                    let per = n.div_ceil(nth);
                    crate::threadpool::par_units(nth, |s| {
                        let (n0, n1) = ((s * per).min(n), ((s + 1) * per).min(n));
                        if n0 < n1 {
                            generic_cols(o, n0, n1)
                        }
                    });
                } else {
                    generic_cols(o, 0, n)
                }
            };
            let cached = |f: &dyn Fn()| {
                gemm::packed_cache::set_enabled(true);
                f();
                gemm::packed_cache::set_enabled(false);
            };
            let (lp, rp) = (l as *const f32, r as *const f32);
            let run = |v: usize, o: &mut [f32]| {
                let op = o.as_mut_ptr() as usize;
                match v {
                    0 => unsafe {
                        gemm_dot_cols(o.as_mut_ptr(), n, lp, k, rp, k, m, k, piece(m, k), 0, n)
                    },
                    1 => generic_cols(op, 0, n),
                    2 => cached(&|| generic_cols(op, 0, n)),
                    #[cfg(feature = "accelerate")]
                    3 => unsafe {
                        let (ni, mi, ki) = (n as i32, m as i32, k as i32);
                        crate::accelerate::sgemm(
                            b'T',
                            b'N',
                            ni,
                            mi,
                            ki,
                            1.,
                            rp,
                            ki,
                            lp,
                            ki,
                            0.,
                            o.as_mut_ptr(),
                            ni,
                        )
                    },
                    4 => gemm_dot(o, n, &lhs, k, &rhs, k, m, n, k),
                    5 => generic_mt(op),
                    6 => cached(&|| generic_mt(op)),
                    _ => {}
                }
            };
            // Calls per sample: about a millisecond's worth.
            let t = Instant::now();
            for _ in 0..10 {
                run(0, &mut out);
            }
            let iters = ((1e-3 / (t.elapsed().as_secs_f64() / 10.0)) as usize).clamp(5, 100_000);
            let mut best = [f64::INFINITY; 7];
            for _ in 0..rounds {
                for &v in &shown {
                    run(v, &mut out);
                    let t = Instant::now();
                    for _ in 0..iters {
                        run(v, &mut out);
                    }
                    best[v] = best[v].min(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
                }
            }
            print!("{:>14}", format!("{m}x{n}x{k}"));
            shown.iter().for_each(|&v| print!(" {:9.2}", best[v]));
            println!();
        }
    }

    /// A split changes which thread computes an output, never the order of its sum: the pool's
    /// result, and one row or eight columns at a time, are the same bits as one call.
    #[test]
    fn gemm_dot_gives_the_same_bits_on_any_split() {
        for (m, n, k) in [(40, 8, 1000), (16, 520, 2048), (3, 700, 600)] {
            let (lhs, rhs) = (data(m * k, 7), data(n * k, 8));
            let (l, r, kc) = (lhs.as_ptr(), rhs.as_ptr(), piece(m, k));
            let mut whole = vec![0f32; m * n];
            unsafe { gemm_dot_cols(whole.as_mut_ptr(), n, l, k, r, k, m, k, kc, 0, n) };
            let mut pool = vec![0f32; m * n];
            gemm_dot(&mut pool, n, &lhs, k, &rhs, k, m, n, k);
            let mut parts = vec![0f32; m * n];
            for i in 0..m {
                for j in (0..n).step_by(8) {
                    let o = parts[i * n..].as_mut_ptr();
                    let j_hi = (j + 8).min(n);
                    unsafe { gemm_dot_cols(o, n, l.add(i * k), k, r, k, 1, k, kc, j, j_hi) };
                }
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&pool), bits(&whole), "{m}x{n}x{k}: pool");
            assert_eq!(bits(&parts), bits(&whole), "{m}x{n}x{k}: parts");
        }
    }

    /// A batch of products each too small to split, like one attention head per entry, goes to
    /// the pool whole and gives the same bits as running the entries one by one; a shared
    /// weight (stride 0) too.
    #[test]
    fn gemm_dot_batched_matches_the_entries_one_by_one() {
        for (batch, m, n, k, rhs_bs) in
            [(8, 16, 37, 64, 37 * 64), (12, 25, 9, 64, 0), (3, 2, 5, 7, 35)]
        {
            let (lhs, rhs) = (data(batch * m * k, 21), data(batch * n * k + 3, 22));
            let mut got = vec![0f32; batch * m * n];
            let b = Batch { n: batch, lhs: m * k, rhs: rhs_bs };
            gemm_dot_batched(&mut got, n, &lhs, k, &rhs, k, m, n, k, b);
            let mut want = vec![0f32; batch * m * n];
            for e in 0..batch {
                let out = &mut want[e * m * n..(e + 1) * m * n];
                gemm_dot(out, n, &lhs[e * m * k..], k, &rhs[e * rhs_bs..], k, m, n, k);
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{batch} x {m}x{n}x{k}");
        }
    }

    #[test]
    fn gemm_dot_across_row_blocks_and_threads() {
        // Long rows make the row blocks short, so several of them run, and `k` is taken in
        // pieces, the last one with a scalar tail. The larger shapes are past the threshold
        // where the work splits over the pool, by columns, or by rows for the last one.
        for (m, n, k) in [
            (70, 9, 4100),
            (20, 13, 1027),
            (130, 40, 1500),
            (16, 300, 64),
            (1, 2049, 64),
            (700, 5, 30),
        ] {
            check(m, n, k, 3);
        }
    }
}
