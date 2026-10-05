//! Direct convolution kernel for aarch64 NEON.
//!
//! The generic path turns a `conv1d` into im2col, a packed gemm and a transpose. [`conv1d`]
//! reads the operands where they are instead, with a 4x16 register tile and narrower tiles for
//! the rows and columns left over, as the wasm kernel in `simd128_conv` does.
//!
//! It vectorizes over time: with stride 1 a tap's input samples are contiguous and the output
//! row is written in place. The reduction runs over `(input channel, tap)` flattened, along
//! which each output channel's weights are contiguous: one load brings four of them and
//! `vfmaq_laneq_f32` applies each lane to the input samples of its step.

use core::arch::aarch64::*;
use std::array::from_fn;

/// One reduction step of a tile: input samples at `xp` against lane `L` of each row's weights.
///
/// # Safety
/// `xp` addresses `4 * V` floats.
#[inline(always)]
unsafe fn step<const L: i32, const R: usize, const V: usize>(
    acc: &mut [[float32x4_t; V]; R],
    xp: *const f32,
    w: &[float32x4_t; R],
) {
    unsafe {
        let xv: [float32x4_t; V] = from_fn(|v| vld1q_f32(xp.add(4 * v)));
        for r in 0..R {
            for v in 0..V {
                acc[r][v] = vfmaq_laneq_f32::<L>(acc[r][v], xv[v], w[r]);
            }
        }
    }
}

/// `R` output channels by `4 * V` time steps of a stride-1 convolution.
///
/// # Safety
/// `x + offs[j]` addresses `4 * V` floats for every reduction step `j`, each `w[r]` addresses
/// `offs.len()` floats, and each `out[r]` addresses `4 * V` floats.
#[inline(always)]
unsafe fn conv_tile<const R: usize, const V: usize>(
    x: *const f32,
    offs: &[u32],
    w: [*const f32; R],
    out: [*mut f32; R],
) {
    unsafe {
        let n = offs.len();
        let at = |j: usize| x.add(*offs.get_unchecked(j) as usize);
        let mut acc = [[vdupq_n_f32(0.0); V]; R];
        let mut j = 0;
        while j + 4 <= n {
            let wv: [float32x4_t; R] = from_fn(|r| vld1q_f32(w[r].add(j)));
            step::<0, R, V>(&mut acc, at(j), &wv);
            step::<1, R, V>(&mut acc, at(j + 1), &wv);
            step::<2, R, V>(&mut acc, at(j + 2), &wv);
            step::<3, R, V>(&mut acc, at(j + 3), &wv);
            j += 4;
        }
        for j in j..n {
            let wv: [float32x4_t; R] = from_fn(|r| vld1q_dup_f32(w[r].add(j)));
            step::<0, R, V>(&mut acc, at(j), &wv);
        }
        for (o, a) in out.iter().zip(&acc) {
            for (v, &a) in a.iter().enumerate() {
                vst1q_f32(o.add(4 * v), a);
            }
        }
    }
}

/// `R` output channels over a window of `width <= 16` time steps: one tile of whole groups of
/// four steps, then the last few steps one at a time. Row `r` of the weights starts at
/// `w + r * w_rs` and of the output at `out + r * out_rs`.
///
/// # Safety
/// As [`conv_tile`], for `width` time steps.
#[inline(always)]
unsafe fn rows<const R: usize>(
    x: *const f32,
    offs: &[u32],
    (w, w_rs): (*const f32, usize),
    (out, out_rs): (*mut f32, usize),
    width: usize,
) {
    unsafe {
        let (w, out) = (from_fn(|r| w.add(r * w_rs)), from_fn(|r| out.add(r * out_rs)));
        let quads = width / 4;
        match quads {
            4 => conv_tile::<R, 4>(x, offs, w, out),
            3 => conv_tile::<R, 3>(x, offs, w, out),
            2 => conv_tile::<R, 2>(x, offs, w, out),
            1 => conv_tile::<R, 1>(x, offs, w, out),
            _ => {}
        }
        for l in 4 * quads..width {
            for r in 0..R {
                let mut s = 0f32;
                for (j, &o) in offs.iter().enumerate() {
                    s += *w[r].add(j) * *x.add(o as usize + l);
                }
                *out[r].add(l) = s;
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
    assert!(ci * l_in <= u32::MAX as usize);
    // Where reduction step `c * k + t` reads, relative to the input window's first sample.
    let offs: Vec<u32> =
        (0..ci).flat_map(|c| (0..k).map(move |t| (c * l_in + t * dilation) as u32)).collect();
    let offs = offs.as_slice();
    let (d, s, wt) = (dst.as_mut_ptr() as usize, src.as_ptr() as usize, w.as_ptr() as usize);
    // Output channels `[c_lo, c_hi)` over time steps `[l_lo, l_hi)` of every batch element,
    // `l_lo` a multiple of 16.
    let run = move |(c_lo, c_hi): (usize, usize), (l_lo, l_hi): (usize, usize)| {
        let (dst, src, w) = (d as *mut f32, s as *const f32, wt as *const f32);
        // SAFETY: the shapes asserted above bound every access, and units write disjoint outputs.
        unsafe {
            for b in 0..batch {
                let x = src.add(b * ci * l_in);
                let y = dst.add(b * co * l_out);
                // Time outermost so a 16-sample window of every input channel stays in L1 while
                // the output channels sweep over it.
                for l0 in (l_lo..l_hi).step_by(16) {
                    let width = (l_hi - l0).min(16);
                    let xt = x.add(l0);
                    for c0 in (c_lo..c_hi).step_by(4) {
                        let wr = (w.add(c0 * ci * k), ci * k);
                        let or = (y.add(c0 * l_out + l0), l_out);
                        match c_hi - c0 {
                            1 => rows::<1>(xt, offs, wr, or, width),
                            2 => rows::<2>(xt, offs, wr, or, width),
                            3 => rows::<3>(xt, offs, wr, or, width),
                            _ => rows::<4>(xt, offs, wr, or, width),
                        }
                    }
                }
            }
        }
    };
    // Units of four output channels, or of 16-sample windows when there are too few channels
    // to go round the threads.
    let work = batch * co * l_out * ci * k;
    let windows = l_out.div_ceil(16);
    if co.div_ceil(4) >= crate::threadpool::threads_for(work) || windows == 1 {
        crate::threadpool::par_units_by(work, co, 4, |c_lo, c_hi| run((c_lo, c_hi), (0, l_out)));
    } else {
        crate::threadpool::par_units_by(work, windows, 1, |w_lo, w_hi| {
            run((0, co), (16 * w_lo, (16 * w_hi).min(l_out)))
        });
    }
}

/// Whether a stride-1, unpadded, single-group convolution should go to [`conv1d`] rather than
/// to im2col and a gemm.
///
/// Below four output steps every output is a scalar tail, and a gemm wins. With Accelerate the
/// gemm runs on the matrix unit, which beats NEON once each im2col column feeds more than a few
/// output channels; with four or fewer, copying the columns costs more than the matrix unit
/// saves.
pub fn takes(ci: usize, co: usize, l_in: usize, l_out: usize) -> bool {
    #[cfg(test)]
    match tests::FORCE.get() {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let fits = ci * l_in <= u32::MAX as usize;
    fits && l_out >= 4 && (cfg!(not(feature = "accelerate")) || co <= 4)
}

#[cfg(test)]
mod tests {
    //! Shapes are chosen to hit every tile and tail branch.
    use super::*;
    use crate::{CPU, CpuDevice, Tensor};

    thread_local! {
        /// Overrides [`takes`] on this thread: 1 takes the direct kernel, 2 the im2col path.
        pub(super) static FORCE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    }

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

    /// `conv1d` with any stride, padding and groups, one multiply-add at a time.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        w: &[f32],
        (batch, ci, co, l_in): (usize, usize, usize, usize),
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Vec<f32> {
        let l_out = (l_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let (cig, cog) = (ci / groups, co / groups);
        let mut y = vec![0f32; batch * co * l_out];
        for b in 0..batch {
            for o in 0..co {
                let g = o / cog;
                for l in 0..l_out {
                    let mut acc = 0f32;
                    for c in 0..cig {
                        for t in 0..k {
                            let pos = (l * stride + t * dilation) as isize - padding as isize;
                            if pos < 0 || pos as usize >= l_in {
                                continue;
                            }
                            let xi = (b * ci + g * cig + c) * l_in + pos as usize;
                            acc += w[(o * cig + c) * k + t] * x[xi];
                        }
                    }
                    y[(b * co + o) * l_out + l] = acc;
                }
            }
        }
        y
    }

    #[test]
    fn conv1d_matches_the_scalar_reference() {
        // Output channel counts leave 1, 2 and 3 rows over; `ci * k` leaves 1 to 3 reduction
        // steps over; lengths leave every number of whole and partial four-step groups over.
        for batch in [1, 2] {
            for (ci, co) in [(1, 1), (3, 5), (5, 4), (2, 17), (6, 7), (4, 6)] {
                for (k, dilation) in [(1, 1), (3, 1), (3, 2), (3, 4), (7, 1)] {
                    for l_out in [1, 3, 4, 5, 8, 12, 15, 16, 17, 33, 48, 61] {
                        let l_in = l_out + (k - 1) * dilation;
                        let x = data(batch * ci * l_in, 1);
                        let w = data(co * ci * k, 2);
                        let mut got = vec![0f32; batch * co * l_out];
                        conv1d(&mut got, &x, &w, batch, ci, co, l_in, l_out, k, dilation);
                        let want = reference(&x, &w, (batch, ci, co, l_in), k, 1, 0, dilation, 1);
                        let what = format!("conv {batch}x{ci}->{co} k{k} d{dilation} L{l_out}");
                        assert_close(&got, &want, &what);
                    }
                }
            }
        }
    }

    /// An input longer than the convolution needs: the kernel reads only the window it covers.
    #[test]
    fn conv1d_reads_rows_longer_than_the_window() {
        let (batch, ci, co, k, l_out) = (2, 3, 5, 3, 20);
        let l_in = l_out + k - 1;
        let x = data(batch * ci * (l_in + 7), 3);
        let w = data(co * ci * k, 4);
        let mut got = vec![0f32; batch * co * l_out];
        conv1d(&mut got, &x, &w, batch, ci, co, l_in + 7, l_out, k, 1);
        let want = reference(&x, &w, (batch, ci, co, l_in + 7), k, 1, 0, 1, 1);
        // The reference covers the whole row; keep its first `l_out` outputs per channel.
        let want: Vec<f32> =
            want.chunks(l_out + 7).flat_map(|row| row[..l_out].iter().copied()).collect();
        assert_close(&got, &want, "long rows");
    }

    /// Convolutions through the backend, the ones the direct kernel can take and the ones it
    /// must leave to im2col: a stride, padding, groups, and a single output step.
    #[test]
    fn backend_conv1d_matches_the_reference() -> crate::Result<()> {
        // (batch, ci, co, l_in, k, stride, padding, dilation, groups)
        for (batch, ci, co, l_in, k, stride, padding, dilation, groups) in [
            (1, 4, 8, 22, 7, 1, 0, 1, 1),
            (2, 6, 5, 40, 3, 1, 0, 2, 1),
            (2, 5, 3, 17, 1, 1, 0, 1, 1),
            (1, 4, 8, 22, 3, 2, 0, 1, 1),
            (2, 4, 8, 22, 3, 1, 1, 1, 1),
            (2, 4, 8, 22, 3, 1, 2, 2, 1),
            (1, 4, 8, 22, 3, 1, 0, 1, 2),
            (2, 6, 6, 22, 4, 2, 1, 1, 3),
            (1, 3, 2, 7, 7, 1, 0, 1, 1),
        ] {
            let x = data(batch * ci * l_in, 5);
            let w = data(co * (ci / groups) * k, 6);
            let xt = Tensor::<f32, CpuDevice>::from_vec(x.clone(), (batch, ci, l_in), &CPU)?;
            let wt = Tensor::<f32, CpuDevice>::from_vec(w.clone(), (co, ci / groups, k), &CPU)?;
            let want =
                reference(&x, &w, (batch, ci, co, l_in), k, stride, padding, dilation, groups);
            // Whatever `takes` decides, each shape is checked with the direct kernel offered and
            // refused; the guards keep the shapes it cannot compute away from it either way.
            for force in [1, 2] {
                FORCE.set(force);
                let got = xt.conv1d(&wt, None, stride, padding, dilation, groups)?.to_vec()?;
                FORCE.set(0);
                let what = format!(
                    "{batch}x{ci}->{co} L{l_in} k{k} s{stride} p{padding} d{dilation} g{groups}"
                );
                assert_close(&got, &want, &what);
            }
        }
        Ok(())
    }

    #[test]
    fn takes_leaves_short_outputs_to_the_gemm() {
        assert!(!takes(32, 512, 1, 1));
        assert!(!takes(32, 512, 3, 3));
        assert!(takes(64, 1, 1922, 1920));
        assert!(takes(64, 4, 1922, 1920));
        assert_eq!(takes(64, 8, 1922, 1920), cfg!(not(feature = "accelerate")));
        assert_eq!(takes(64, 32, 1922, 1920), cfg!(not(feature = "accelerate")));
        assert_eq!(takes(512, 512, 22, 16), cfg!(not(feature = "accelerate")));
    }

    /// Times every convolution of a streaming decoder frame, direct against im2col and a gemm,
    /// through the backend. Run it alone, at a fixed thread count:
    /// `RAYON_NUM_THREADS=1 cargo test -p xn --release --lib conv1d_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn conv1d_bench() {
        use crate::backend::Backend;
        use std::time::Instant;
        // (ci, co, k, l_out)
        let shapes = [
            (32, 512, 1, 1),
            (32, 512, 1, 2),
            (32, 512, 1, 4),
            (32, 512, 1, 8),
            (32, 512, 1, 16),
            (512, 512, 7, 4),
            (512, 512, 7, 8),
            (512, 512, 7, 16),
            (256, 128, 3, 96),
            (128, 256, 1, 96),
            (128, 64, 3, 480),
            (64, 128, 1, 480),
            (64, 32, 3, 1920),
            (32, 64, 1, 1920),
            (64, 1, 3, 1920),
            (64, 2, 3, 1920),
            (64, 4, 3, 1920),
            (64, 8, 3, 1920),
            (64, 16, 3, 1920),
        ];
        println!("threads {}", crate::threadpool::size());
        println!("microseconds, best and median of 40; ratio is im2col over direct, best of each");
        println!("{:>16} {:>7} {:>15} {:>15} {:>6}", "shape", "MMAC", "direct", "im2col", "ratio");
        for (ci, co, k, l_out) in shapes {
            let l_in = l_out + k - 1;
            let (x, w) = (data(ci * l_in, 7), data(co * ci * k, 8));
            let mut y = vec![0f32; co * l_out];
            let mut run = |path: u8| {
                FORCE.set(path);
                CpuDevice::conv1d(&mut y, &x, &w, 1, ci, co, l_in, l_out, k, 1, 0, 1, 1).unwrap();
            };
            let macs = ci * co * k * l_out;
            // About 2 ms a sample.
            let reps = (2_000_000_000 / (macs * 10).max(1)).clamp(1, 2000);
            let mut times = [vec![], vec![]];
            for round in 0..41 {
                for i in 0..2 {
                    let path = if round % 2 == 0 { i } else { 1 - i };
                    let t = Instant::now();
                    for _ in 0..reps {
                        run(path as u8 + 1);
                    }
                    if round > 0 {
                        times[path].push(t.elapsed().as_secs_f64() * 1e6 / reps as f64);
                    }
                }
            }
            let best = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[0], v[v.len() / 2])
            };
            let (d, d_med) = best(&mut times[0]);
            let (i, i_med) = best(&mut times[1]);
            println!(
                "{:>16} {:>7.2} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>6.2}",
                format!("{ci}->{co} k{k} L{l_out}"),
                macs as f64 / 1e6,
                d,
                d_med,
                i,
                i_med,
                i / d
            );
        }
        FORCE.set(0);
    }
}
