use crate::{BinaryOp, Result, UnaryOp, WithDType, WithDTypeF};
use rayon::prelude::*;

const USE_IM2COL_CONV1D: bool = true;
const USE_COL2IM_CONV1D_TR: bool = true;

fn copy_strided_2d<T: WithDType>(
    dst: &mut [T],
    src: &[T],
    src_offset: usize,
    d0: usize,
    d1: usize,
    s0: usize,
) {
    let mut src_idx = src_offset;
    let mut dst_off = 0;
    for _ in 0..d0 {
        dst[dst_off..dst_off + d1].copy_from_slice(&src[src_idx..src_idx + d1]);
        src_idx += s0;
        dst_off += d1;
    }
}

fn copy_strided_3d<T: WithDType>(
    dst: &mut [T],
    src: &[T],
    src_offset: usize,
    dims: [usize; 3],
    strides: [usize; 2],
) {
    let [d0, d1, d2] = dims;
    let [s0, s1] = strides;
    let mut dst_off = 0;
    for i0 in 0..d0 {
        let base = src_offset + i0 * s0;
        for i1 in 0..d1 {
            let src_idx = base + i1 * s1;
            dst[dst_off..dst_off + d2].copy_from_slice(&src[src_idx..src_idx + d2]);
            dst_off += d2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_<T: WithDType>(
    dst: &mut [T],
    (lhs, lhs_o): (&[T], usize),
    (rhs, rhs_o): (&[T], usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> Result<()> {
    let lhs = &lhs[lhs_o..];
    let rhs = &rhs[rhs_o..];
    for b_idx in 0..lhs_b {
        let dst = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
        let lhs = &lhs[b_idx * lhs_b_stride..];
        let rhs = &rhs[b_idx * rhs_b_stride..];
        unsafe {
            gemm::gemm(
                /* m: usize = */ m,
                /* n: usize = */ n,
                /* k: usize = */ k,
                /* dst: *mut T = */ dst.as_mut_ptr(),
                /* dst_cs: isize = */ dst_cs as isize,
                /* dst_rs: isize = */ dst_rs as isize,
                /* read_dst: bool = */ false,
                /* lhs: *const T = */ lhs.as_ptr(),
                /* lhs_cs: isize = */ lhs_cs as isize,
                /* lhs_rs: isize = */ lhs_rs as isize,
                /* rhs: *const T = */ rhs.as_ptr(),
                /* rhs_cs: isize = */ rhs_cs as isize,
                /* rhs_rs: isize = */ rhs_rs as isize,
                /* alpha: T = */ T::zero(),
                /* beta: T = */ T::one(),
                /* conj_dst: bool = */ false,
                /* conj_lhs: bool = */ false,
                /* conj_rhs: bool = */ false,
                gemm::Parallelism::Rayon(get_num_threads()),
            )
        }
    }
    Ok(())
}

impl crate::Backend for crate::CpuDevice {
    type Storage<T: WithDType> = Vec<T>;

    fn name(&self) -> String {
        "cpu".to_string()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, _: &Self) -> Result<Self::Storage<T>> {
        Ok(vec![T::zero(); len])
    }

    fn from_vec<T: WithDType>(v: Vec<T>, _: &Self) -> Result<Self::Storage<T>> {
        Ok(v)
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        Ok(std::borrow::Cow::Borrowed(&src[..len]))
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d += s),
            BinaryOp::Sub => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d -= s),
            BinaryOp::Mul => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d *= s),
            BinaryOp::Div => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d /= s),
            BinaryOp::Maximum => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| {
                if s > *d {
                    *d = s
                }
            }),
            BinaryOp::Minimum => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| {
                if s < *d {
                    *d = s
                }
            }),
        }
        Ok(())
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        match op {
            UnaryOp::Cos => apply_inplace_unary(&mut dst[..len], |v| *v = v.cos()),
            UnaryOp::Sin => apply_inplace_unary(&mut dst[..len], |v| *v = v.sin()),
            UnaryOp::Sqr => apply_inplace_unary(&mut dst[..len], |v| *v = *v * *v),
            UnaryOp::Sqrt => apply_inplace_unary(&mut dst[..len], |v| *v = v.sqrt()),
            UnaryOp::Rsqrt => apply_inplace_unary(&mut dst[..len], |v| *v = T::one() / v.sqrt()),
            UnaryOp::Abs => apply_inplace_unary(&mut dst[..len], |v| *v = v.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_inplace_unary(&mut dst[..len], |v| {
                    let x = v.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    *v = T::from_f32(x * 0.5 * (1.0 + erf_val));
                })
            }
            UnaryOp::Elu { alpha } => apply_inplace_unary(&mut dst[..len], |v| {
                let x = v.to_f32();
                *v = T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) });
            }),
            UnaryOp::Relu => apply_inplace_unary(&mut dst[..len], |v| {
                if *v < T::zero() {
                    *v = T::zero()
                }
            }),
            UnaryOp::Silu => apply_inplace_unary(&mut dst[..len], |v| {
                *v = *v / (T::one() + (T::zero() - *v).exp())
            }),
            UnaryOp::Tanh => apply_inplace_unary(&mut dst[..len], |v| *v = v.tanh()),
            UnaryOp::Sigmoid => apply_inplace_unary(&mut dst[..len], |v| {
                *v = T::one() / (T::one() + (T::zero() - *v).exp())
            }),
        }
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        match op {
            UnaryOp::Cos => apply_unary(&mut dst[..len], &src[..len], |s| s.cos()),
            UnaryOp::Sin => apply_unary(&mut dst[..len], &src[..len], |s| s.sin()),
            UnaryOp::Sqr => apply_unary(&mut dst[..len], &src[..len], |s| s * s),
            UnaryOp::Sqrt => apply_unary(&mut dst[..len], &src[..len], |s| s.sqrt()),
            UnaryOp::Rsqrt => apply_unary(&mut dst[..len], &src[..len], |s| T::one() / s.sqrt()),
            UnaryOp::Abs => apply_unary(&mut dst[..len], &src[..len], |s| s.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_unary(&mut dst[..len], &src[..len], |s| {
                    let x = s.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    T::from_f32(x * 0.5 * (1.0 + erf_val))
                })
            }
            UnaryOp::Elu { alpha } => apply_unary(&mut dst[..len], &src[..len], |s| {
                let x = s.to_f32();
                T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }),
            UnaryOp::Relu => {
                let zero = T::zero();
                apply_unary(&mut dst[..len], &src[..len], |s| if s < zero { zero } else { s })
            }
            UnaryOp::Silu => apply_unary(&mut dst[..len], &src[..len], |s| {
                s / (T::one() + (T::zero() - s).exp())
            }),
            UnaryOp::Tanh => apply_unary(&mut dst[..len], &src[..len], |s| s.tanh()),
            UnaryOp::Sigmoid => apply_unary(&mut dst[..len], &src[..len], |s| {
                T::one() / (T::one() + (T::zero() - s).exp())
            }),
        }
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a + b),
            BinaryOp::Sub => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a - b),
            BinaryOp::Mul => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a * b),
            BinaryOp::Div => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a / b),
            BinaryOp::Maximum => {
                apply_binary(
                    &mut dst[..len],
                    &lhs[..len],
                    &rhs[..len],
                    |a, b| {
                        if a > b { a } else { b }
                    },
                )
            }
            BinaryOp::Minimum => {
                apply_binary(
                    &mut dst[..len],
                    &lhs[..len],
                    &rhs[..len],
                    |a, b| {
                        if a < b { a } else { b }
                    },
                )
            }
        }
        Ok(())
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        let zero = T::zero();
        let one = T::one();
        if add == zero && scale == one {
            Self::copy(dst, src, len)
        } else if add == zero {
            apply_unary(&mut dst[..len], &src[..len], |s| s * scale);
            Ok(())
        } else if scale == one {
            apply_unary(&mut dst[..len], &src[..len], |s| s + add);
            Ok(())
        } else {
            apply_unary(&mut dst[..len], &src[..len], |s| s * scale + add);
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        for i1 in 0..d1 {
            let dst_idx = i1 * dst_s + dst_o;
            let src_idx = i1 * src_s + src_o;
            let dst = &mut dst[dst_idx..dst_idx + d2];
            let src = &src[src_idx..src_idx + d2];
            dst.copy_from_slice(src)
        }
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            dst.copy_from_slice(src);
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_i = dims[..dim1].iter().product::<usize>();
            let d_j = dims[dim1 + 1..dim2].iter().product::<usize>();
            let d_k = dims[(dim2 + 1)..].iter().product::<usize>();
            let d1 = dims[dim1];
            let d2 = dims[dim2];
            // Inefficient, we should blit the data where possible.
            // i: pre
            for i in 0..d_i {
                for a1 in 0..d1 {
                    // j: mid
                    for j in 0..d_j {
                        for a2 in 0..d2 {
                            let src_idx = i * d1 * d_j * d2 * d_k
                                + a1 * d_j * d2 * d_k
                                + j * d2 * d_k
                                + a2 * d_k;
                            let dst_idx = i * d2 * d_j * d1 * d_k
                                + a2 * d_j * d1 * d_k
                                + j * d1 * d_k
                                + a1 * d_k;
                            dst[dst_idx..dst_idx + d_k]
                                .copy_from_slice(&src[src_idx..src_idx + d_k]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        dst[..l].copy_from_slice(&src[..l]);
        Ok(())
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, v: T, l: usize) -> Result<()> {
        dst[..l].fill(v);
        Ok(())
    }

    fn rope<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if dst.len() != b * h * t * d {
            crate::bail!("rope unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
            for i_t in 0..t {
                for i_d in 0..d / 2 {
                    let i1 = i_t * d + i_d;
                    let i2 = i1 + d / 2;
                    let i_cs = i_t * (d / 2) + i_d;
                    let (src_i1, src_i2) = (src[i1], src[i2]);
                    dst[i1] = src_i1 * cos[i_cs] - src_i2 * sin[i_cs];
                    dst[i2] = src_i1 * sin[i_cs] + src_i2 * cos[i_cs];
                }
            }
        });

        Ok(())
    }

    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if dst.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let (s_i, s_ip) = (src[i], src[i + 1]);
                dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
            }
        });
        Ok(())
    }

    #[cfg(feature = "accelerate")]
    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        let lhs = &lhs[lhs_o..];
        let rhs = &rhs[rhs_o..];
        let (lda, transa) = if (rhs_cs == 1 || n == 1) && (rhs_rs == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_cs == k && rhs_rs == 1 {
            (k as i32, b'T')
        } else {
            return gemm_(
                dst,
                (lhs, lhs_o),
                (rhs, rhs_o),
                m,
                n,
                k,
                lhs_b,
                lhs_b_stride,
                rhs_b_stride,
                (dst_cs, dst_rs),
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_cs == 1 || k == 1) && (lhs_rs == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_cs == m && lhs_rs == 1 {
            (m as i32, b'T')
        } else {
            return gemm_(
                dst,
                (lhs, lhs_o),
                (rhs, rhs_o),
                m,
                n,
                k,
                lhs_b,
                lhs_b_stride,
                rhs_b_stride,
                (dst_cs, dst_rs),
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
        };

        match T::DTYPE {
            crate::DType::F32 => {
                for b_idx in 0..lhs_b {
                    let dst_p = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
                    let lhs_p = &lhs[b_idx * lhs_b_stride..];
                    let rhs_p = &rhs[b_idx * rhs_b_stride..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        crate::accelerate::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            _ => {
                return gemm_(
                    dst,
                    (lhs, lhs_o),
                    (rhs, rhs_o),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    (dst_cs, dst_rs),
                    (lhs_cs, lhs_rs),
                    (rhs_cs, rhs_rs),
                );
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "accelerate"))]
    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        gemm_(
            dst,
            (lhs, lhs_o),
            (rhs, rhs_o),
            m,
            n,
            k,
            lhs_b,
            lhs_b_stride,
            rhs_b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
        )
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let rank = dims.len();
        let total: usize = dims.iter().product();
        if rank == 1 && src_strides[0] == 1 {
            dst[..total].copy_from_slice(&src[src_offset..src_offset + total]);
            return Ok(());
        }
        if rank == 2 && src_strides[1] == 1 {
            copy_strided_2d(dst, src, src_offset, dims[0], dims[1], src_strides[0]);
            return Ok(());
        }
        if rank == 3 && src_strides[2] == 1 {
            copy_strided_3d(
                dst,
                src,
                src_offset,
                [dims[0], dims[1], dims[2]],
                [src_strides[0], src_strides[1]],
            );
            return Ok(());
        }
        let mut index = vec![0usize; rank];
        for dst_elem in dst.iter_mut().take(total) {
            let mut src_idx = src_offset;
            for d in 0..rank {
                src_idx += index[d] * src_strides[d];
            }
            *dst_elem = src[src_idx];
            for d in (0..rank).rev() {
                index[d] += 1;
                if index[d] < dims[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        Ok(())
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = src_dims[..dim].iter().product();
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];

        for left in 0..left_size {
            for i in 0..src_dim_size {
                for right in 0..right_size {
                    let src_flat = left * src_dim_size * right_size + i * right_size + right;
                    let idx = ids[src_flat] as usize;
                    let dst_flat = left * dst_dim_size * right_size + idx * right_size + right;
                    dst[dst_flat] = src[src_flat];
                }
            }
        }
        Ok(())
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = dims[..dim].iter().product();
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = dims[dim];

        for left in 0..left_size {
            for (i, &idx) in ids.iter().enumerate() {
                let idx = idx as usize;
                let src_offset = left * src_dim_size * right_size + idx * right_size;
                let dst_offset = left * ids.len() * right_size + i * right_size;
                dst[dst_offset..dst_offset + right_size]
                    .copy_from_slice(&src[src_offset..src_offset + right_size]);
            }
        }
        Ok(())
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                // Query at position offset + idx1 can attend to keys at positions 0..=offset+idx1
                // So mask positions where idx2 > offset + idx1
                for idx2 in (offset + idx1 + 1)..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    dst[idx] = T::neg_infinity()
                }
            }
        }
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let mut max = T::neg_infinity();
            for &v in src.iter() {
                max = T::max(v, max)
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s - max).exp();
            }
            let sum_exp = dst.iter().map(|v| <T as WithDTypeF>::to_f32(*v)).sum::<f32>();
            for d in dst.iter_mut() {
                *d = T::from_f32(d.to_f32() / sum_exp)
            }
        });
        Ok(())
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v.to_f32() * v.to_f32()).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = T::from_f32((*s).to_f32() / m * (*alpha).to_f32())
            }
        });
        Ok(())
    }

    fn layer_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        let weight = &weight[..dim_m1];
        let bias = &bias[..dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            // Compute mean
            let sum: f32 = src.iter().map(|&v| v.to_f32()).sum();
            let mean = sum / dim_m1 as f32;

            // Compute variance (always uses mean)
            let var: f32 = src
                .iter()
                .map(|&v| {
                    let diff = v.to_f32() - mean;
                    diff * diff
                })
                .sum::<f32>()
                / dim_m1 as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize and apply weight/bias
            for i in 0..dim_m1 {
                let centered = if remove_mean { src[i].to_f32() - mean } else { src[i].to_f32() };
                let normalized = centered * inv_std;
                dst[i] = T::from_f32(normalized * weight[i].to_f32() + bias[i].to_f32());
            }
        });
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_val = T::neg_infinity();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    max_val = T::max(max_val, src[src_idx]);
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = max_val;
            }
        }
        Ok(())
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_val = T::infinity();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    min_val = T::min(min_val, src[src_idx]);
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = min_val;
            }
        }
        Ok(())
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_val = T::infinity();
                let mut min_idx: usize = 0;
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    if src[src_idx].to_f32() < min_val.to_f32() {
                        min_val = src[src_idx];
                        min_idx = d;
                    }
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = min_idx as i64;
            }
        }
        Ok(())
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = T::zero();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum += src[src_idx];
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = sum;
            }
        }
        Ok(())
    }

    fn broadcast_binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a + b
                })
            }
            BinaryOp::Sub => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a - b
                })
            }
            BinaryOp::Mul => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a * b
                })
            }
            BinaryOp::Div => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a / b
                })
            }
            BinaryOp::Maximum => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    if a > b { a } else { b }
                })
            }
            BinaryOp::Minimum => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    if a < b { a } else { b }
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        if USE_IM2COL_CONV1D && groups == 1 {
            // IM2COL approach: transform conv1d into matrix multiplication
            // 1. Im2Col: transform input [B, C, L] -> [B, L_out, C * K]
            // 2. Matmul: [B, L_out, C*K] x [C*K, out_channels] -> [B, L_out, out_channels]
            // 3. Transpose result to [B, out_channels, L_out]

            let k = in_channels * kernel_size;

            // Step 1: Im2Col transformation
            let col = im2col1d(
                src,
                batch,
                in_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
            );

            // Step 2: Matrix multiplication
            // col is [B, L_out, K] where K = in_channels * kernel_size
            // kernel is [out_channels, in_channels, kernel_size] = [out_channels, K]
            // We want [B, L_out, out_channels]
            let mut result = vec![T::zero(); batch * out_length * out_channels];

            Self::gemm(
                &mut result,
                (&col, 0),
                (kernel, 0),
                /* m */ out_length,
                /* n */ out_channels,
                /* k */ k,
                batch,
                out_length * k,
                out_length * out_channels,
                (1, out_channels),
                (1, k),
                (k, 1),
            )?;

            // Step 3: Transpose from [B, L_out, out_channels] to [B, out_channels, L_out]
            for b_idx in 0..batch {
                for l_idx in 0..out_length {
                    for c_idx in 0..out_channels {
                        let src_idx =
                            b_idx * out_length * out_channels + l_idx * out_channels + c_idx;
                        let dst_idx =
                            b_idx * out_channels * out_length + c_idx * out_length + l_idx;
                        dst[dst_idx] = result[src_idx];
                    }
                }
            }

            Ok(())
        } else {
            // Fallback: original implementation for grouped convolutions
            conv1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        // COL2IM approach can be used when:
        // - groups == 1
        // - padding == 0
        // - output_padding == 0
        let can_use_col2im = groups == 1 && padding == 0 && output_padding == 0;

        if USE_COL2IM_CONV1D_TR && can_use_col2im {
            // COL2IM approach: matmul + col2im transformation
            // 1. Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
            // 2. Matmul: [B, L_in, C_in] @ [C_in, C_out * K] -> [B, L_in, C_out * K]
            // 3. Col2Im: [B, L_in, C_out, K] -> [B, C_out, L_out]

            // Step 1: Transpose input to [B, L_in, C_in]
            let mut src_transposed = vec![T::zero(); batch * length * in_channels];
            for b in 0..batch {
                for l in 0..length {
                    for c in 0..in_channels {
                        let src_idx = b * in_channels * length + c * length + l;
                        let dst_idx = b * length * in_channels + l * in_channels + c;
                        src_transposed[dst_idx] = src[src_idx];
                    }
                }
            }

            // Step 2: Matrix multiplication
            // src_transposed: [B, L_in, C_in]
            // kernel: [C_in, C_out, K] stored row-major, treat as [C_in, C_out * K]
            // result: [B, L_in, C_out * K]
            let n = out_channels * kernel_size;
            let mut col = vec![T::zero(); batch * length * n];

            Self::gemm(
                &mut col,
                (&src_transposed, 0),
                (kernel, 0),
                /* m */ length,
                /* n */ n,
                /* k */ in_channels,
                batch,
                length * in_channels,
                in_channels * n,
                (1, n),
                (1, in_channels),
                (1, n),
            )?;

            // Step 3: Col2Im transformation
            // col is [B, L_in, C_out * K] = [B, L_in, C_out, K]
            // output is [B, C_out, L_out]
            col2im1d(dst, &col, batch, length, out_channels, kernel_size, stride);

            Ok(())
        } else {
            // Fallback: original implementation for grouped convolutions or with padding
            conv_transpose1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
            )
        }
    }
}

/// Apply a binary operation in-place: dst[i] = op(dst[i], src[i])
#[inline(always)]
fn apply_bin_assign<T: Copy, F>(dst: &mut [T], src: &[T], f: F)
where
    F: Fn(&mut T, T),
{
    for (d, s) in dst.iter_mut().zip(src) {
        f(d, *s);
    }
}

/// Apply a unary operation in-place: dst[i] = op(dst[i])
#[inline(always)]
fn apply_inplace_unary<T: Copy, F>(dst: &mut [T], f: F)
where
    F: Fn(&mut T),
{
    for d in dst.iter_mut() {
        f(d);
    }
}

/// Apply a unary operation: dst[i] = op(src[i])
#[inline(always)]
fn apply_unary<T: Copy, F>(dst: &mut [T], src: &[T], f: F)
where
    F: Fn(T) -> T,
{
    for (d, s) in dst.iter_mut().zip(src) {
        *d = f(*s);
    }
}

/// Apply a binary operation: dst[i] = op(lhs[i], rhs[i])
#[inline(always)]
fn apply_binary<T: Copy, F>(dst: &mut [T], lhs: &[T], rhs: &[T], f: F)
where
    F: Fn(T, T) -> T,
{
    for ((d, l), r) in dst.iter_mut().zip(lhs).zip(rhs) {
        *d = f(*l, *r);
    }
}

/// Im2Col transformation for 1D convolution.
/// Transforms input from [B, C, L] to [B, L_out, C * K] for matrix multiplication.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all)]
fn im2col1d<T: WithDTypeF>(
    src: &[T],
    batch: usize,
    in_channels: usize,
    length: usize,
    l_out: usize,
    l_k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Vec<T> {
    let k = in_channels * l_k;
    let mut dst = vec![T::zero(); batch * l_out * k];

    for b_idx in 0..batch {
        let src_b_offset = b_idx * in_channels * length;
        let dst_b_offset = b_idx * l_out * k;

        for l_idx in 0..l_out {
            let dst_l_offset = dst_b_offset + l_idx * k;

            for c_idx in 0..in_channels {
                let src_c_offset = src_b_offset + c_idx * length;
                let dst_c_offset = dst_l_offset + c_idx * l_k;

                for (l_k_idx, dst) in dst[dst_c_offset..dst_c_offset + l_k].iter_mut().enumerate() {
                    let src_l = l_idx * stride + l_k_idx * dilation;

                    // Handle padding
                    if src_l < padding || src_l >= length + padding {
                        // Zero padding - already initialized to zero
                        continue;
                    }
                    let src_l = src_l - padding;
                    let src_idx = src_c_offset + src_l;
                    *dst = src[src_idx];
                }
            }
        }
    }

    dst
}

/// Direct conv1d implementation (fallback for grouped convolutions).
#[allow(clippy::too_many_arguments)]
fn conv1d_direct<T: WithDTypeF>(
    dst: &mut [T],
    src: &[T],
    kernel: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> crate::Result<()> {
    let in_c_per_group = in_channels / groups;

    // Initialize output to zero
    dst.iter_mut().for_each(|v| *v = T::zero());

    // Reorder input from [B, C, L] to [B, L, C] for better memory access in the inner loop
    let mut src_reordered = vec![T::zero(); batch * length * in_channels];
    for b in 0..batch {
        for l in 0..length {
            for c in 0..in_channels {
                let src_idx = b * in_channels * length + c * length + l;
                let dst_idx = b * length * in_channels + l * in_channels + c;
                src_reordered[dst_idx] = src[src_idx];
            }
        }
    }

    // Process each kernel offset
    for k_offset in 0..kernel_size {
        // Parallelize over output channels
        (0..out_channels).into_par_iter().for_each(|out_c| {
            let g = out_c / (out_channels / groups);
            let in_c_start = g * in_c_per_group;

            // Gather kernel weights for this output channel and kernel offset
            // kernel layout: [out_channels, in_c_per_group, kernel_size]
            let k_cont: Vec<T> = (0..in_c_per_group)
                .map(|ic| {
                    let k_idx = out_c * in_c_per_group * kernel_size + ic * kernel_size + k_offset;
                    kernel[k_idx]
                })
                .collect();

            for b in 0..batch {
                let dst_base = b * out_channels * out_length + out_c * out_length;

                for ol in 0..out_length {
                    let src_l = ol * stride + k_offset * dilation;

                    // Check padding bounds
                    if src_l < padding || src_l >= padding + length {
                        continue;
                    }
                    let src_l = src_l - padding;

                    // Compute dot product over input channels
                    let src_base = b * length * in_channels + src_l * in_channels + in_c_start;
                    let mut d = T::zero();
                    for ic in 0..in_c_per_group {
                        d += src_reordered[src_base + ic] * k_cont[ic];
                    }

                    // Accumulate into output
                    // Safety: each out_c is processed by a different thread, so no races
                    let dst_idx = dst_base + ol;
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr += d;
                    }
                }
            }
        });
    }
    Ok(())
}

/// Col2Im transformation for 1D transposed convolution.
/// Transforms col from [B, L_in, C_out, K] to output [B, C_out, L_out].
/// Following the reference implementation closely.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all)]
fn col2im1d<T: WithDTypeF>(
    dst: &mut [T],
    col: &[T],
    b_size: usize,
    l_in: usize,
    c_out: usize,
    k_size: usize,
    stride: usize,
) {
    let l_out = (l_in - 1) * stride + k_size;

    dst.fill(T::zero());

    // Strides for destination [B, C_out, L_out]
    let (dst_s0, dst_s1) = (c_out * l_out, l_out);

    // Strides for source [B, L_in, C_out, K] stored as [B, L_in, C_out * K]
    let (src_s0, src_s1, src_s2) = (c_out * k_size * l_in, c_out * k_size, k_size);

    for l_in_i in 0..l_in {
        for k_i in 0..k_size {
            let l_out_i = l_in_i * stride + k_i;
            for b_i in 0..b_size {
                for c_i in 0..c_out {
                    let dst_idx = b_i * dst_s0 + c_i * dst_s1 + l_out_i;
                    let src_idx = b_i * src_s0 + l_in_i * src_s1 + c_i * src_s2 + k_i;
                    dst[dst_idx] += col[src_idx];
                }
            }
        }
    }
}

/// Direct conv_transpose1d implementation (fallback for grouped convolutions or with padding).
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_direct<T: WithDTypeF>(
    dst: &mut [T],
    src: &[T],
    kernel: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    groups: usize,
) -> crate::Result<()> {
    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    dst.fill(T::zero());

    // Reorder input from [B, C, L] to [B, L, C] for contiguous memory access
    let mut src_reordered = vec![T::zero(); batch * length * in_channels];
    for b in 0..batch {
        for l in 0..length {
            for c in 0..in_channels {
                let src_idx = b * in_channels * length + c * length + l;
                let dst_idx = b * length * in_channels + l * in_channels + c;
                src_reordered[dst_idx] = src[src_idx];
            }
        }
    }

    // Process each kernel offset
    for k_offset in 0..kernel_size {
        // Parallelize over output channels
        (0..out_channels).into_par_iter().for_each(|out_c| {
            let g = out_c / out_c_per_group;
            let oc_in_group = out_c % out_c_per_group;
            let in_c_start = g * in_c_per_group;

            // Gather kernel weights for this output channel and kernel offset
            // Kernel layout: [in_channels, out_channels/groups, kernel_size]
            let k_cont: Vec<T> = (0..in_c_per_group)
                .map(|ic| {
                    let in_c = in_c_start + ic;
                    let k_idx =
                        in_c * out_c_per_group * kernel_size + oc_in_group * kernel_size + k_offset;
                    kernel[k_idx]
                })
                .collect();

            for b in 0..batch {
                for il in 0..length {
                    let out_pos_raw = il * stride + k_offset;

                    // Check padding bounds
                    if out_pos_raw < padding || out_pos_raw >= out_length + padding {
                        continue;
                    }
                    let out_pos = out_pos_raw - padding;

                    // Compute dot product over input channels
                    let src_base = b * length * in_channels + il * in_channels + in_c_start;
                    let mut d = T::zero();
                    for ic in 0..in_c_per_group {
                        d += src_reordered[src_base + ic] * k_cont[ic];
                    }

                    // Accumulate into output
                    // Safety: each out_c is processed by a different thread, so no races
                    let dst_idx = b * out_channels * out_length + out_c * out_length + out_pos;
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr += d;
                    }
                }
            }
        });
    }
    Ok(())
}

/// Helper function for broadcast binary operations.
#[inline(always)]
fn broadcast_binary_op<T: WithDType>(
    dst: &mut [T],
    lhs: &[T],
    rhs: &[T],
    dst_shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    op: impl Fn(T, T) -> T,
) -> Result<()> {
    let lhs_no_zero = lhs_strides.iter().all(|&s| s > 0);
    let rhs_no_zero = rhs_strides.iter().all(|&s| s > 0);

    if lhs_no_zero && rhs_no_zero {
        apply_binary(dst, lhs, rhs, &op);
        return Ok(());
    }
    if lhs_no_zero && rhs_strides == [0, 1] {
        for idx0 in 0..dst_shape[0] {
            for (idx1, rhs) in rhs.iter().enumerate().take(dst_shape[1]) {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let lhs_idx = idx0 * lhs_strides[0] + idx1;
                dst[dst_idx] = op(lhs[lhs_idx], *rhs);
            }
        }
        return Ok(());
    }
    if lhs_no_zero && rhs_strides == [1, 0] {
        for (idx0, rhs) in rhs.iter().enumerate().take(dst_shape[0]) {
            for idx1 in 0..dst_shape[1] {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let lhs_idx = idx0 * lhs_strides[0] + idx1;
                dst[dst_idx] = op(lhs[lhs_idx], *rhs);
            }
        }
        return Ok(());
    }
    if rhs_no_zero && lhs_strides == [0, 1] {
        for idx0 in 0..dst_shape[0] {
            for (idx1, lhs) in lhs.iter().enumerate().take(dst_shape[1]) {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let rhs_idx = idx0 * rhs_strides[0] + idx1;
                dst[dst_idx] = op(*lhs, rhs[rhs_idx]);
            }
        }
        return Ok(());
    }
    if rhs_no_zero && lhs_strides == [1, 0] {
        for (idx0, lhs) in lhs.iter().enumerate().take(dst_shape[0]) {
            for idx1 in 0..dst_shape[1] {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let rhs_idx = idx0 * rhs_strides[0] + idx1;
                dst[dst_idx] = op(*lhs, rhs[rhs_idx]);
            }
        }
        return Ok(());
    }

    let total_elems: usize = dst_shape.iter().product();
    let rank = dst_shape.len();

    for (dst_idx, dst) in dst.iter_mut().enumerate().take(total_elems) {
        // Convert linear index to multi-dimensional indices
        let mut remaining = dst_idx;
        let mut lhs_idx = 0usize;
        let mut rhs_idx = 0usize;

        for d in 0..rank {
            let stride: usize = dst_shape[d + 1..].iter().product::<usize>().max(1);
            let coord = remaining / stride;
            remaining %= stride;

            lhs_idx += coord * lhs_strides[d];
            rhs_idx += coord * rhs_strides[d];
        }

        *dst = op(lhs[lhs_idx], rhs[rhs_idx]);
    }

    Ok(())
}

pub(crate) fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}
