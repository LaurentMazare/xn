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

fn bin_apply<T: WithDType>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => {
            if a > b {
                a
            } else {
                b
            }
        }
        BinaryOp::Minimum => {
            if a < b {
                a
            } else {
                b
            }
        }
    }
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("Metal ({})", self.device_name)
    }

    fn synchronize(&self) -> Result<()> {
        // Submit any pending batch and wait for it.
        self.flush()
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let (buffer, ptr, class) = dev.alloc_buffer(len * T::BYTE_SIZE)?;
        Ok(Storage { buffer, ptr, len, class, device: dev.clone(), _t: PhantomData })
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
        dst.device.flush()?;
        dst.as_mut_slice()[..len].fill(elem);
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        dst.device.flush()?;
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
        dst.device.flush()?;
        let mut rng = rand::rng();
        for v in dst.as_mut_slice()[..len].iter_mut() {
            *v = distr.sample(&mut rng);
        }
        Ok(())
    }

    fn copy<T: WithDType>(dst: &mut Self::Storage<T>, src: &Self::Storage<T>, len: usize) -> Result<()> {
        // Recorded as a GPU blit so it stays in the batch.
        dst.device.record_copy(&dst.buffer, &src.buffer, len * T::BYTE_SIZE)
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        use half::{bf16, f16};
        src.device.flush()?;
        macro_rules! cast {
            ($s:ty, $d:ty, |$v:ident| $e:expr) => {{
                let s = unsafe { std::slice::from_raw_parts(src.ptr as *const $s, len) };
                let d = unsafe { std::slice::from_raw_parts_mut(dst.ptr as *mut $d, len) };
                for (o, i) in d.iter_mut().zip(s.iter()) {
                    let $v = *i;
                    *o = $e;
                }
            }};
        }
        use DType::*;
        match (T::DTYPE, U::DTYPE) {
            (F16, F16) => cast!(f16, f16, |v| v),
            (BF16, BF16) => cast!(bf16, bf16, |v| v),
            (F32, F32) => cast!(f32, f32, |v| v),
            (I64, I64) => cast!(i64, i64, |v| v),
            (U8, U8) => cast!(u8, u8, |v| v),
            (F32, F16) => cast!(f32, f16, |v| f16::from_f32(v)),
            (F32, BF16) => cast!(f32, bf16, |v| bf16::from_f32(v)),
            (F16, F32) => cast!(f16, f32, |v| v.to_f32()),
            (BF16, F32) => cast!(bf16, f32, |v| v.to_f32()),
            (F16, BF16) => cast!(f16, bf16, |v| bf16::from_f32(v.to_f32())),
            (BF16, F16) => cast!(bf16, f16, |v| f16::from_f32(v.to_f32())),
            (F32, I64) => cast!(f32, i64, |v| v as i64),
            (F32, U8) => cast!(f32, u8, |v| v as u8),
            (F16, I64) => cast!(f16, i64, |v| v.to_f32() as i64),
            (F16, U8) => cast!(f16, u8, |v| v.to_f32() as u8),
            (BF16, I64) => cast!(bf16, i64, |v| v.to_f32() as i64),
            (BF16, U8) => cast!(bf16, u8, |v| v.to_f32() as u8),
            (I64, F32) => cast!(i64, f32, |v| v as f32),
            (I64, F16) => cast!(i64, f16, |v| f16::from_f32(v as f32)),
            (I64, BF16) => cast!(i64, bf16, |v| bf16::from_f32(v as f32)),
            (U8, F32) => cast!(u8, f32, |v| v as f32),
            (U8, F16) => cast!(u8, f16, |v| f16::from_f32(v as f32)),
            (U8, BF16) => cast!(u8, bf16, |v| bf16::from_f32(v as f32)),
            (I64, U8) => cast!(i64, u8, |v| v as u8),
            (U8, I64) => cast!(u8, i64, |v| v as i64),
        }
        Ok(())
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        src.device.flush()?;
        Ok(std::borrow::Cow::Owned(src.as_slice()[..len].to_vec()))
    }

    fn inplace_unary<T: WithDTypeF>(dst: &mut Self::Storage<T>, len: usize, op: UnaryOp) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "inplace_unary")?;
        let (code, alpha) = unary_op_code(op);
        let push = Pc::new().usize(len).u32(code).f32(alpha);
        dst.device.dispatch(
            &format!("unary_{dt}"),
            &[&dst.buffer, &dst.buffer],
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
            &[&src.buffer, &dst.buffer],
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).u32(binary_op_code(op));
            dst.device.dispatch(
                &format!("binary_{dt}"),
                &[&dst.buffer, &s.buffer, &dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let src = s.as_slice()[..len].to_vec();
            for (d, sv) in dst.as_mut_slice()[..len].iter_mut().zip(src) {
                *d = bin_apply(op, *d, sv);
            }
            Ok(())
        }
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).u32(binary_op_code(op));
            dst.device.dispatch(
                &format!("binary_{dt}"),
                &[&lhs.buffer, &rhs.buffer, &dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let l = lhs.as_slice();
            let r = rhs.as_slice();
            let d = dst.as_mut_slice();
            for i in 0..len {
                d[i] = bin_apply(op, l[i], r[i]);
            }
            Ok(())
        }
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
                &[&src.buffer, &dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(numel).usize(d1).usize(d2).usize(d_i).usize(d_j).usize(d_k);
            dst.device.dispatch(
                &format!("transpose_{dt}"),
                &[&src.buffer, &dst.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for dst_idx in 0..numel {
                let mut rem = dst_idx;
                let i = rem / (d2 * d_j * d1 * d_k);
                rem -= i * (d2 * d_j * d1 * d_k);
                let a2 = rem / (d_j * d1 * d_k);
                rem -= a2 * (d_j * d1 * d_k);
                let j = rem / (d1 * d_k);
                rem -= j * (d1 * d_k);
                let a1 = rem / d_k;
                rem -= a1 * d_k;
                let k = rem;
                let src_idx = i * d1 * d_j * d2 * d_k
                    + a1 * d_j * d2 * d_k
                    + j * d2 * d_k
                    + a2 * d_k
                    + k;
                d[dst_idx] = s[src_idx];
            }
            Ok(())
        }
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push =
                Pc::new().usize(d1).usize(d2).usize(src_s).usize(dst_s).usize(src_o).usize(dst_o);
            dst.device.dispatch(
                &format!("copy2d_{dt}"),
                &[&src.buffer, &dst.buffer],
                &push,
                div_ceil(d1 * d2, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    d[dst_o + i1 * dst_s + i2] = s[src_o + i1 * src_s + i2];
                }
            }
            Ok(())
        }
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
            &[&cos.buffer, &sin.buffer, &src.buffer, &dst.buffer],
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
            &[&cos.buffer, &sin.buffer, &src.buffer, &dst.buffer],
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
        let buffers: [&metal::BufferRef; 3] = [&dst.buffer, &lhs.0.buffer, &rhs.0.buffer];
        if m == 1 {
            // Decode path: one threadgroup per output column, grid (n, batch, 1).
            dst.device.dispatch_nd(&format!("gemv_{dt}"), &buffers, &push, (n as u32, lhs_b as u32, 1))
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(left_size).usize(num_ids).usize(right_size).usize(src_dim_size);
            dst.device.dispatch(
                &format!("index_select_{dt}"),
                &[&src.buffer, &dst.buffer, &ids.buffer],
                &push,
                div_ceil(total, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let ids_h = ids.as_slice();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for left in 0..left_size {
                for id_i in 0..num_ids {
                    let idx = ids_h[id_i];
                    for r in 0..right_size {
                        let dst_off = (left * num_ids + id_i) * right_size + r;
                        if idx == -1 {
                            d[dst_off] = T::zero();
                        } else {
                            let src_off = (left * src_dim_size + idx as usize) * right_size + r;
                            d[dst_off] = s[src_off];
                        }
                    }
                }
            }
            Ok(())
        }
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
            &[&dst.buffer],
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
        dst.device.dispatch(&format!("softmax_{dt}"), &[&src.buffer, &dst.buffer], &push, d as u32)
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
            &[&src.buffer, &dst.buffer, &alpha.buffer],
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
            &[&src.buffer, &dst.buffer, &weight.buffer, &bias.buffer],
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let info: Vec<u32> =
                dims.iter().chain(src_strides.iter()).map(|&v| v as u32).collect();
            let scratch = dst.device.scratch_u32(&info)?;
            let push = Pc::new().usize(numel).usize(dims.len()).usize(src_offset);
            let res = dst.device.dispatch(
                &format!("copy_strided_{dt}"),
                &[&src.buffer, &dst.buffer, &scratch.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            );
            // Only defer after the dispatch is recorded (see defer_free).
            dst.device.defer_free(scratch);
            res
        } else {
            dst.device.flush()?;
            let n = dims.len();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for idx in 0..numel {
                let mut si = 0usize;
                let mut rem = idx;
                for di in (0..n).rev() {
                    si += (rem % dims[di]) * src_strides[di];
                    rem /= dims[di];
                }
                d[idx] = s[src_offset + si];
            }
            Ok(())
        }
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(numel).usize(right_size).usize(src_dim_size).usize(dst_dim_size);
            dst.device.dispatch(
                &format!("scatter_set_{dt}"),
                &[&dst.buffer, &src.buffer, &ids.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush()?;
            let ids_h = ids.as_slice();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for i in 0..numel {
                let right = i % right_size;
                let left = i / (right_size * src_dim_size);
                let idx = ids_h[i] as usize;
                let dst_off = left * dst_dim_size * right_size + idx * right_size + right;
                d[dst_off] = s[i];
            }
            Ok(())
        }
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
        if let Some(dt) = float_suffix::<T>(&dst.device) {
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
                &[&lhs.buffer, &rhs.buffer, &dst.buffer, &scratch.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            );
            // Only defer after the dispatch is recorded (see defer_free).
            dst.device.defer_free(scratch);
            res
        } else {
            dst.device.flush()?;
            let n = dst_shape.len();
            let l = lhs.as_slice();
            let r = rhs.as_slice();
            let d = dst.as_mut_slice();
            for idx in 0..numel {
                let mut li = 0usize;
                let mut ri = 0usize;
                let mut rem = idx;
                for di in (0..n).rev() {
                    let coord = rem % dst_shape[di];
                    rem /= dst_shape[di];
                    li += coord * lhs_strides[di];
                    ri += coord * rhs_strides[di];
                }
                d[idx] = bin_apply(op, l[li], r[ri]);
            }
            Ok(())
        }
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
            &[&dst.buffer, &src.buffer, &kernel.buffer],
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
        _output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        check_f32::<T>("conv_transpose1d")?;
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
            &[&dst.buffer, &src.buffer, &kernel.buffer],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }
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
        dst.device.dispatch(&format!("reduce_{dt}"), &[&src.buffer, &dst.buffer], &push, num_outputs as u32)
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
            &[&src.buffer, &dst.buffer],
            &push,
            num_outputs as u32,
        )
    }
}
