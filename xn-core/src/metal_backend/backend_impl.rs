// Included from `mod.rs`. Implements the `Backend` trait for the Metal
// `Device`, plus host fallbacks for non-float data-movement ops.

/// Convert a float-typed scalar value to `f32`. Only called on the GPU path,
/// where `float_suffix` has already restricted `T` to {f32, f16, bf16}.
fn scalar_to_f32<T: WithDType>(v: T) -> f32 {
    match T::DTYPE {
        DType::F32 => unsafe { *(&v as *const T as *const f32) },
        DType::F16 => unsafe { (*(&v as *const T as *const half::f16)).to_f32() },
        DType::BF16 => unsafe { (*(&v as *const T as *const half::bf16)).to_f32() },
        d => unreachable!("scalar_to_f32 on non-float dtype {d:?}"),
    }
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("Metal ({})", self.device_name)
    }

    fn synchronize(&self) -> Result<()> {
        // Submit any pending batch and wait for it.
        self.flush("synchronize")
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let (buffer, ptr, class) = dev.alloc_buffer(len * T::BYTE_SIZE)?;
        Ok(Storage {
            buffer,
            ptr,
            len,
            class,
            gpu_used: std::sync::atomic::AtomicBool::new(false),
            device: dev.clone(),
            _t: PhantomData,
        })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let len = v.len();
        let storage = unsafe { Self::alloc_uninit::<T>(len, dev)? };
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, storage.ptr, len * T::BYTE_SIZE);
        }
        Ok(storage)
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        // Values whose bytes are all identical (zeros for every dtype) fill
        // on the GPU as a blit, keeping the batch pipeline intact. Anything
        // else drains the pipeline and fills from the host.
        let elem_bytes =
            unsafe { std::slice::from_raw_parts(&elem as *const T as *const u8, T::BYTE_SIZE) };
        if elem_bytes.iter().all(|&b| b == elem_bytes[0]) {
            return dst.device.record_fill(dst.buf(), elem_bytes[0], len * T::BYTE_SIZE);
        }
        if dst.is_gpu_used() {
            dst.device.flush("host fill")?;
        }
        dst.as_mut_slice()[..len].fill(elem);
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        if dst.is_gpu_used() {
            dst.device.flush("rand_uniform")?;
        }
        let range = up - lo;
        for v in dst.as_mut_slice()[..len].iter_mut() {
            *v = rand::random::<f32>() * range + lo;
        }
        Ok(())
    }

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()> {
        use rand_distr::Distribution;
        let distr = match rand_distr::Normal::<f32>::new(mean, std) {
            Ok(d) => d,
            Err(e) => crate::bail!("failed to create normal distribution for randn: {e}"),
        };
        if dst.is_gpu_used() {
            dst.device.flush("randn")?;
        }
        let mut rng = rand::rng();
        for v in dst.as_mut_slice()[..len].iter_mut() {
            *v = distr.sample(&mut rng);
        }
        Ok(())
    }

    fn copy<T: WithDType>(dst: &mut Self::Storage<T>, src: &Self::Storage<T>, len: usize) -> Result<()> {
        // Recorded as a GPU blit so it stays in the batch.
        dst.device.record_copy(dst.buf(), src.buf(), len * T::BYTE_SIZE)
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        // Same-dtype conversion is a plain buffer copy; everything else runs
        // as a GPU cast kernel. Nothing drains the pipeline. The float->int
        // kernels follow the host `as` semantics (truncate, saturate,
        // NaN -> 0).
        if T::DTYPE == U::DTYPE {
            return src.device.record_copy(dst.buf(), src.buf(), len * T::BYTE_SIZE);
        }
        let (s, d) = (any_suffix::<T>(), any_suffix::<U>());
        let push = Pc::new().usize(len);
        src.device.dispatch(
            &format!("cast_{s}_{d}"),
            &[src.buf(), dst.buf()],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        if src.is_gpu_used() {
            src.device.flush("readback")?;
        }
        Ok(std::borrow::Cow::Owned(src.as_slice()[..len].to_vec()))
    }

    fn inplace_unary<T: WithDTypeF>(dst: &mut Self::Storage<T>, len: usize, op: UnaryOp) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "inplace_unary")?;
        let (code, alpha) = unary_op_code(op);
        let push = Pc::new().usize(len).u32(code).f32(alpha);
        dst.device.dispatch(
            &format!("unary_{dt}"),
            &[dst.buf(), dst.buf()],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "unary")?;
        let (code, alpha) = unary_op_code(op);
        let push = Pc::new().usize(len).u32(code).f32(alpha);
        dst.device.dispatch(
            &format!("unary_{dt}"),
            &[src.buf(), dst.buf()],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let dt = any_suffix::<T>();
        let push = Pc::new().usize(len).u32(binary_op_code(op));
        dst.device.dispatch(
            &format!("binary_{dt}"),
            &[dst.buf(), s.buf(), dst.buf()],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let dt = any_suffix::<T>();
        let push = Pc::new().usize(len).u32(binary_op_code(op));
        dst.device.dispatch(
            &format!("binary_{dt}"),
            &[lhs.buf(), rhs.buf(), dst.buf()],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        if add == T::zero() && scale == T::one() {
            return Self::copy(dst, src, len);
        }
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).f32(scalar_to_f32(scale)).f32(scalar_to_f32(add));
            dst.device.dispatch(
                &format!("scale_add_{dt}"),
                &[src.buf(), dst.buf()],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("host scale_add")?;
            let s = src.as_slice()[..len].to_vec();
            for (d, sv) in dst.as_mut_slice()[..len].iter_mut().zip(s) {
                *d = sv * scale + add;
            }
            Ok(())
        }
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            return Self::copy(dst, src, numel);
        }
        let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
        let d_i: usize = dims[..dim1].iter().product();
        let d_j: usize = dims[dim1 + 1..dim2].iter().product();
        let d_k: usize = dims[(dim2 + 1)..].iter().product();
        let d1 = dims[dim1];
        let d2 = dims[dim2];
        let dt = any_suffix::<T>();
        let push = Pc::new().usize(numel).usize(d1).usize(d2).usize(d_i).usize(d_j).usize(d_k);
        dst.device.dispatch(
            &format!("transpose_{dt}"),
            &[src.buf(), dst.buf()],
            &push,
            div_ceil(numel, WORKGROUP_SIZE),
        )
    }

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
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dt = any_suffix::<T>();
        let push =
            Pc::new().usize(d1).usize(d2).usize(src_s).usize(dst_s).usize(src_o).usize(dst_o);
        dst.device.dispatch(
            &format!("copy2d_{dt}"),
            &[src.buf(), dst.buf()],
            &push,
            div_ceil(d1 * d2, WORKGROUP_SIZE),
        )
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
        unbatched_rope: bool,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "rope")?;
        let bh = b * h;
        let td = t * d;
        let cs_stride_b = if unbatched_rope { t * d / 2 } else { 0 };
        let off = pos * d / 2;
        let push = Pc::new()
            .usize(bh)
            .usize(td)
            .usize(d)
            .usize(h)
            .usize(cs_stride_b)
            .usize(off)
            .usize(off);
        dst.device.dispatch(
            &format!("rope_{dt}"),
            &[cos.buf(), sin.buf(), src.buf(), dst.buf()],
            &push,
            div_ceil(bh * td / 2, WORKGROUP_SIZE),
        )
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
        unbatched_rope: bool,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "rope_i")?;
        let bh = b * h;
        let td = t * d;
        let cs_stride_b = if unbatched_rope { t * d / 2 } else { 0 };
        let off = pos * d / 2;
        let push = Pc::new().usize(bh).usize(td).usize(h).usize(cs_stride_b).usize(off).usize(off);
        dst.device.dispatch(
            &format!("rope_i_{dt}"),
            &[cos.buf(), sin.buf(), src.buf(), dst.buf()],
            &push,
            div_ceil(bh * td / 2, WORKGROUP_SIZE),
        )
    }

    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: (&Self::Storage<T>, usize),
        rhs: (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        dst_strides: (usize, usize),
        lhs_strides: (usize, usize),
        rhs_strides: (usize, usize),
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "gemm")?;
        let (dst_cs, dst_rs) = dst_strides;
        let (lhs_cs, lhs_rs) = lhs_strides;
        let (rhs_cs, rhs_rs) = rhs_strides;
        // MLX GEMV path for m == 1: `gemv` when each output reads a
        // contiguous matrix row (matmul_t against row-major weights, the
        // decode hot path), `gemv_t` when the matrix is contiguous along its
        // output columns (plain matmul). The block-shape heuristic mirrors
        // MLX's `matmul.cpp` and only picks instantiated specializations.
        // Ineligible or unavailable cases fall through to the steel GEMM /
        // local gemv below.
        if m == 1 && dst.device.use_mlx_gemv && dst_rs == n && dst_cs == 1 && lhs_cs == 1 {
            let vec_byte_off = lhs.1 * T::BYTE_SIZE;
            let mat_byte_off = rhs.1 * T::BYTE_SIZE;
            // (kernel name, matrix ld, outputs per threadgroup, group (y, z))
            let sel = if rhs_rs == 1 {
                let bm = if n >= 4096 { 8u64 } else { 4 };
                let tm = if n < 4 { 1u64 } else { 4 };
                let name = format!("gemv_{dt}_bm{bm}_bn1_sm1_sn32_tm{tm}_tn4");
                Some((name, rhs_cs, bm * tm, (1u64, bm)))
            } else if rhs_cs == 1 {
                let (sm, sn) = if k >= 8192 && n >= 2048 { (4u64, 8u64) } else { (8, 4) };
                let bn = if n >= 2048 {
                    16u64
                } else if n >= 512 {
                    4
                } else {
                    2
                };
                let tn = if n < 4 { 1u64 } else { 4 };
                let name = format!("gemv_t_{dt}_bm1_bn{bn}_sm{sm}_sn{sn}_tm4_tn{tn}");
                Some((name, rhs_rs, bn * sn * tn, (bn, 1u64)))
            } else {
                None
            };
            if vec_byte_off % 16 == 0
                && mat_byte_off % 16 == 0
                && let Some((name, ld, n_out_per_tgp, (gy, gz))) = sel
                && let Some(pipeline) = dst.device.get_mlx_gemv_pipeline(&name)
            {
                let n_tgp = (n as u64).div_ceil(n_out_per_tgp);
                return dst.device.dispatch_mlx_gemv(
                    &pipeline,
                    &name,
                    (rhs.0.buf(), mat_byte_off as u64),
                    (lhs.0.buf(), vec_byte_off as u64),
                    dst.buf(),
                    (k as i32, n as i32, ld as i32),
                    (lhs_b_stride as u64, rhs_b_stride as u64),
                    (n_tgp, 1, lhs_b as u64),
                    (32, gy, gz),
                );
            }
        }
        // MLX steel GEMM path (simdgroup-matrix kernels), much faster than the
        // plain tiled kernel. It needs a contiguous row-major dst and a unit
        // minor stride on each input (the nn/nt/tn/tt layouts, with a free
        // leading dimension). Buffer byte offsets are kept 16-byte aligned for
        // the vectorized block loaders. Anything else falls through to the
        // generic kernels.
        if (m > 1 || rhs_rs != 1) && dst_rs == n && dst_cs == 1 {
            // (leading dimension, transposed) when the layout is compatible.
            let a_layout = if lhs_cs == 1 {
                Some((lhs_rs, "n"))
            } else if lhs_rs == 1 {
                Some((lhs_cs, "t"))
            } else {
                None
            };
            let b_layout = if rhs_cs == 1 {
                Some((rhs_rs, "n"))
            } else if rhs_rs == 1 {
                Some((rhs_cs, "t"))
            } else {
                None
            };
            let a_byte_off = lhs.1 * T::BYTE_SIZE;
            let b_byte_off = rhs.1 * T::BYTE_SIZE;
            if let (Some((lda, ta)), Some((ldb, tb))) = (a_layout, b_layout)
                && a_byte_off % 16 == 0
                && b_byte_off % 16 == 0
            {
                let aligned =
                    (m.is_multiple_of(MLX_BM), n.is_multiple_of(MLX_BN), k.is_multiple_of(MLX_BK));
                let trans = format!("{ta}{tb}");
                if let Some(pipeline) =
                    dst.device.get_mlx_gemm_pipeline(&trans, dt, aligned, lhs_b > 1)
                {
                    let (tiles_n, tiles_m) = (n.div_ceil(MLX_BN), m.div_ceil(MLX_BM));
                    let params = MlxGemmParams {
                        m: m as i32,
                        n: n as i32,
                        k: k as i32,
                        lda: lda as i32,
                        ldb: ldb as i32,
                        ldd: n as i32,
                        tiles_n: tiles_n as i32,
                        tiles_m: tiles_m as i32,
                        batch_stride_a: lhs_b_stride as isize,
                        batch_stride_b: rhs_b_stride as isize,
                        batch_stride_d: (m * n) as isize,
                        swizzle_log: 0,
                        gemm_k_iterations_aligned: (k / MLX_BK) as i32,
                        batch_ndim: 1,
                    };
                    return dst.device.dispatch_mlx_gemm(
                        &pipeline,
                        (lhs.0.buf(), a_byte_off as u64),
                        (rhs.0.buf(), b_byte_off as u64),
                        dst.buf(),
                        &params,
                        lhs_b as i32,
                        &[lhs_b_stride as isize, rhs_b_stride as isize],
                        (tiles_n as u64, tiles_m as u64, lhs_b as u64),
                    );
                }
            }
        }
        let push = Pc::new()
            .usize(m)
            .usize(n)
            .usize(k)
            .usize(lhs_b)
            .usize(lhs_b_stride)
            .usize(rhs_b_stride)
            .usize(lhs_cs)
            .usize(lhs_rs)
            .usize(rhs_cs)
            .usize(rhs_rs)
            .usize(dst_rs)
            .usize(dst_cs)
            .usize(lhs.1)
            .usize(rhs.1);
        let buffers: [&metal::BufferRef; 3] = [dst.buf(), lhs.0.buf(), rhs.0.buf()];
        if m == 1 {
            // Decode path: one simdgroup per output column, GEMV_NSG columns
            // per threadgroup, grid (ceil(n / GEMV_NSG), batch, 1).
            let groups = (div_ceil(n, GEMV_NSG), lhs_b as u32, 1);
            dst.device.dispatch_nd(&format!("gemv_{dt}"), &buffers, &push, groups)
        } else {
            // Tiled kernel: grid (ceil(n/16), ceil(m/16), batch), threadgroup (16, 16, 1).
            const TILE: u32 = 16;
            let groups = (div_ceil(n, TILE), div_ceil(m, TILE), lhs_b as u32);
            dst.device.dispatch_nd(&format!("gemm_tiled_{dt}"), &buffers, &push, groups)
        }
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        num_ids: usize,
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = dims[..dim].iter().product();
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = dims[dim];
        let total = left_size * num_ids * right_size;
        let dt = any_suffix::<T>();
        let push = Pc::new().usize(left_size).usize(num_ids).usize(right_size).usize(src_dim_size);
        dst.device.dispatch(
            &format!("index_select_{dt}"),
            &[src.buf(), dst.buf(), ids.buf()],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "apply_causality_mask")?;
        let total = bh * t1 * t2;
        let push = Pc::new().usize(bh).usize(t1).usize(t2).usize(offset);
        dst.device.dispatch(
            &format!("causality_mask_{dt}"),
            &[dst.buf()],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "softmax")?;
        let push = Pc::new().usize(dim_m1);
        dst.device.dispatch(&format!("softmax_{dt}"), &[src.buf(), dst.buf()], &push, d as u32)
    }

    fn snake<T: WithDTypeF>(
        _dst: &mut Self::Storage<T>,
        _src: &Self::Storage<T>,
        _alpha: &Self::Storage<T>,
        _beta_scale: &Self::Storage<T>,
        _channels: usize,
        _row_len: usize,
        _numel: usize,
    ) -> Result<()> {
        crate::bail!("snake is not implemented for this backend")
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "rms_norm")?;
        let push = Pc::new().usize(dim_m1).f32(eps);
        dst.device.dispatch(
            &format!("rmsnorm_{dt}"),
            &[src.buf(), dst.buf(), alpha.buf()],
            &push,
            d as u32,
        )
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
        let dt = dtype_suffix::<T>(&dst.device, "layer_norm")?;
        let push = Pc::new().usize(dim_m1).f32(eps).u32(if remove_mean { 1 } else { 0 });
        dst.device.dispatch(
            &format!("layernorm_{dt}"),
            &[src.buf(), dst.buf(), weight.buf(), bias.buf()],
            &push,
            d as u32,
        )
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 1)
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 2)
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 0)
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce_arg(dst, src, dim_size, outer_size, inner_size, 0)
    }

    fn reduce_argmax<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce_arg(dst, src, dim_size, outer_size, inner_size, 1)
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if numel == 0 {
            return Ok(());
        }
        let dt = any_suffix::<T>();
        let info: Vec<u32> = dims.iter().chain(src_strides.iter()).map(|&v| v as u32).collect();
        let scratch = dst.device.scratch_u32(&info)?;
        let push = Pc::new().usize(numel).usize(dims.len()).usize(src_offset);
        let res = dst.device.dispatch(
            &format!("copy_strided_{dt}"),
            &[src.buf(), dst.buf(), &scratch.buffer],
            &push,
            div_ceil(numel, WORKGROUP_SIZE),
        );
        // Only defer after the dispatch is recorded (see defer_free).
        dst.device.defer_free(scratch);
        res
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];
        let numel: usize = src_dims.iter().product();
        if numel == 0 {
            return Ok(());
        }
        let dt = any_suffix::<T>();
        let push = Pc::new().usize(numel).usize(right_size).usize(src_dim_size).usize(dst_dim_size);
        dst.device.dispatch(
            &format!("scatter_set_{dt}"),
            &[dst.buf(), src.buf(), ids.buf()],
            &push,
            div_ceil(numel, WORKGROUP_SIZE),
        )
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
        let numel: usize = dst_shape.iter().product();
        if numel == 0 {
            return Ok(());
        }
        let dt = any_suffix::<T>();
        let info: Vec<u32> = dst_shape
            .iter()
            .chain(lhs_strides.iter())
            .chain(rhs_strides.iter())
            .map(|&v| v as u32)
            .collect();
        let scratch = dst.device.scratch_u32(&info)?;
        let push = Pc::new().usize(numel).usize(dst_shape.len()).u32(binary_op_code(op));
        let res = dst.device.dispatch(
            &format!("broadcast_{dt}"),
            &[lhs.buf(), rhs.buf(), dst.buf(), &scratch.buffer],
            &push,
            div_ceil(numel, WORKGROUP_SIZE),
        );
        // Only defer after the dispatch is recorded (see defer_free).
        dst.device.defer_free(scratch);
        res
    }

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
        check_f32::<T>("conv1d")?;
        if groups == 1 {
            return conv1d_im2col(
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
            );
        }
        let total = batch * out_channels * out_length;
        let push = Pc::new()
            .usize(batch)
            .usize(in_channels)
            .usize(out_channels)
            .usize(length)
            .usize(out_length)
            .usize(kernel_size)
            .usize(stride)
            .usize(padding)
            .usize(dilation)
            .usize(groups);
        dst.device.dispatch(
            "conv1d_f32",
            &[dst.buf(), src.buf(), kernel.buf()],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }

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
        check_f32::<T>("conv_transpose1d")?;
        // col2im assumes groups == 1, no padding/output_padding and (per the
        // Backend trait, which has no dilation param here) dilation == 1.
        if groups == 1 && padding == 0 && output_padding == 0 {
            return conv_transpose1d_col2im(
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
            );
        }
        let total = batch * out_channels * out_length;
        let push = Pc::new()
            .usize(batch)
            .usize(in_channels)
            .usize(out_channels)
            .usize(length)
            .usize(out_length)
            .usize(kernel_size)
            .usize(stride)
            .usize(padding)
            .usize(groups);
        dst.device.dispatch(
            "conv_transpose1d_f32",
            &[dst.buf(), src.buf(), kernel.buf()],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }
}

/// `groups == 1` conv1d via im2col + GEMM + transpose (mirrors the CUDA and
/// Vulkan backends): unfold `src` into `col` [batch, out_length, in_channels*kernel_size],
/// multiply by the [out_channels, in_channels*kernel_size] weight matrix
/// (matmul_t), then transpose the [batch, out_length, out_channels] GEMM
/// result into dst's [batch, out_channels, out_length] layout. This trades
/// conv1d's naive per-output-element gather loop for the MLX GEMM/GEMV path
/// used everywhere else, at the cost of the `col` scratch buffer.
#[allow(clippy::too_many_arguments)]
fn conv1d_im2col<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<()> {
    let dev = dst.device.clone();
    let k = in_channels * kernel_size;

    let col =
        unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * out_length * k, &dev)? };
    let push = Pc::new()
        .usize(batch)
        .usize(in_channels)
        .usize(length)
        .usize(out_length)
        .usize(kernel_size)
        .usize(stride)
        .usize(padding)
        .usize(dilation);
    dev.dispatch(
        "im2col1d_f32",
        &[col.buf(), src.buf()],
        &push,
        div_ceil(batch * out_length * k, WORKGROUP_SIZE),
    )?;

    // result[b, l, oc] = sum_k col[b, l, k] * kernel[oc, k]
    let mut result = unsafe {
        <Device as crate::Backend>::alloc_uninit::<T>(batch * out_length * out_channels, &dev)?
    };
    <Device as crate::Backend>::gemm(
        &mut result,
        (&col, 0),
        (kernel, 0),
        out_length,
        out_channels,
        k,
        batch,
        out_length * k,
        0,
        (1, out_channels),
        (1, k),
        (k, 1),
    )?;

    // [batch, out_length, out_channels] -> dst's [batch, out_channels, out_length].
    <Device as crate::Backend>::transpose(dst, &result, 1, 2, &[batch, out_length, out_channels])
}

/// `groups == 1`, no padding/output_padding conv_transpose1d via transpose +
/// GEMM + col2im (mirrors the CUDA and Vulkan backends): transpose `src` to
/// [batch, length, in_channels], multiply by the [in_channels, out_channels*kernel_size]
/// weight matrix to get `col` [batch, length, out_channels*kernel_size], then
/// fold `col` into dst via col2im's gather (each output position sums the
/// compatible (input position, kernel offset) pairs directly, no atomics).
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_col2im<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    let dev = dst.device.clone();
    let n = out_channels * kernel_size;

    let mut src_t =
        unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * length * in_channels, &dev)? };
    <Device as crate::Backend>::transpose(&mut src_t, src, 1, 2, &[batch, in_channels, length])?;

    // col[b, l, j] = sum_c src_t[b, l, c] * kernel[c, j]
    let mut col =
        unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * length * n, &dev)? };
    <Device as crate::Backend>::gemm(
        &mut col,
        (&src_t, 0),
        (kernel, 0),
        length,
        n,
        in_channels,
        batch,
        length * in_channels,
        0,
        (1, n),
        (1, in_channels),
        (1, n),
    )?;

    let push = Pc::new()
        .usize(batch)
        .usize(length)
        .usize(out_channels)
        .usize(out_length)
        .usize(kernel_size)
        .usize(stride);
    let total = batch * out_channels * out_length;
    dev.dispatch("col2im1d_f32", &[dst.buf(), col.buf()], &push, div_ceil(total, WORKGROUP_SIZE))
}

impl Device {
    fn reduce<T: WithDTypeF>(
        &self,
        dst: &mut Storage<T>,
        src: &Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
        op: u32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "reduce")?;
        let num_outputs = outer_size * inner_size;
        if num_outputs == 0 {
            return Ok(());
        }
        let push = Pc::new().usize(num_outputs).usize(dim_size).usize(inner_size).u32(op);
        dst.device.dispatch(&format!("reduce_{dt}"), &[src.buf(), dst.buf()], &push, num_outputs as u32)
    }

    fn reduce_arg<T: WithDTypeF>(
        &self,
        dst: &mut Storage<i64>,
        src: &Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
        op: u32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&src.device, "reduce_arg")?;
        let num_outputs = outer_size * inner_size;
        if num_outputs == 0 {
            return Ok(());
        }
        let push = Pc::new().usize(num_outputs).usize(dim_size).usize(inner_size).u32(op);
        dst.device.dispatch(
            &format!("reduce_arg_{dt}"),
            &[src.buf(), dst.buf()],
            &push,
            num_outputs as u32,
        )
    }
}
