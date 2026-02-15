use crate::{Backend, BinaryOp, Dim, Error, Result, Tensor, TensorOrView, WithDType, WithDTypeF};

/// Compute the broadcast output shape for two input shapes.
fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let out_rank = lhs.len().max(rhs.len());
    let mut out_shape = vec![0usize; out_rank];
    for (i, out_dim) in out_shape.iter_mut().enumerate() {
        let lhs_dim = if i < out_rank - lhs.len() { 1 } else { lhs[i - (out_rank - lhs.len())] };
        let rhs_dim = if i < out_rank - rhs.len() { 1 } else { rhs[i - (out_rank - rhs.len())] };

        *out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            crate::bail!("cannot broadcast between shapes {lhs:?} and {rhs:?}");
        };
    }

    Ok(out_shape)
}

fn check_same_shape<T: WithDType, B: Backend>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    op: &'static str,
) -> Result<()> {
    if a.shape != b.shape {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
            op,
        }
        .bt());
    }
    Ok(())
}

macro_rules! binary_op {
    ($n:ident, $bn:ident, $v:ident) => {
        #[tracing::instrument(skip_all)]
        pub fn $n(&self, other: &Self) -> Result<Self> {
            self.binary(other, BinaryOp::$v)
        }

        #[tracing::instrument(skip_all)]
        pub fn $bn(&self, other: &Self) -> Result<Self> {
            self.broadcast_binary(other, BinaryOp::$v)
        }
    };
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn binary(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        check_same_shape(self, other, op.as_str())?;
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.binary_(self, other, op)?;
        Ok(result)
    }

    pub fn broadcast_binary(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        let out_shape = broadcast_shape(self.dims(), other.dims())?;
        let result = unsafe { Tensor::alloc_uninit(out_shape, self.device()) }?;
        result.broadcast_binary_(self, other, op)?;
        Ok(result)
    }

    binary_op!(add, broadcast_add, Add);
    binary_op!(sub, broadcast_sub, Sub);
    binary_op!(mul, broadcast_mul, Mul);
    binary_op!(div, broadcast_div, Div);
    binary_op!(minimum, broadcast_minimum, Minimum);
    binary_op!(maximum, broadcast_maximum, Maximum);

    /// Transpose two dimensions.
    /// Returns a `TensorView` (zero-copy). Call `.contiguous()?` on the result
    /// if you need a contiguous `Tensor`.
    #[tracing::instrument(skip_all)]
    pub fn transpose<D1: Dim, D2: Dim>(
        &self,
        dim1: D1,
        dim2: D2,
    ) -> Result<crate::TensorView<T, B>> {
        crate::TensorView::from(self).transpose(dim1, dim2)
    }

    pub fn copy(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.copy_(self)?;
        Ok(result)
    }

    pub fn full_like(&self, value: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.fill_(value)?;
        Ok(result)
    }

    pub fn scale(&self, m: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.scale_(self, m)?;
        Ok(result)
    }

    pub fn add_scalar(&self, a: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.add_scalar_(self, a)?;
        Ok(result)
    }

    pub fn scale_add(&self, scale: T, add: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.scale_add_(self, scale, add)?;
        Ok(result)
    }

    /// Flatten all dimensions into a single dimension.
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    /// Flatten dimensions from start to end (inclusive) into a single dimension.
    pub fn flatten<D: Dim>(&self, start_dim: D, end_dim: D) -> Result<Self> {
        let start_dim = start_dim.to_index(self.shape(), "flatten start_dim")?;
        let end_dim = end_dim.to_index(self.shape(), "flatten end_dim")?;
        let dims = self.dims();
        if start_dim > end_dim {
            crate::bail!("flatten: start_dim {start_dim} > end_dim {end_dim}");
        }
        let flat_size: usize = dims[start_dim..=end_dim].iter().product();
        let mut new_dims = Vec::with_capacity(dims.len() - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(flat_size);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.reshape(new_dims)
    }

    /// Create a tensor of zeros with the same shape.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape().clone(), self.device())
    }

    /// Transpose (swap last two dimensions).
    /// Returns a `TensorView` (zero-copy). Call `.contiguous()?` on the result
    /// if you need a contiguous `Tensor`.
    pub fn t(&self) -> Result<crate::TensorView<T, B>> {
        let rank = self.rank();
        if rank < 2 {
            crate::bail!("t requires at least 2 dimensions");
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        let mut new_dims = self.dims().to_vec();
        new_dims.insert(dim, 1);
        self.reshape(new_dims)
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn cos(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.cos_(self)?;
        Ok(result)
    }

    pub fn sin(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sin_(self)?;
        Ok(result)
    }

    pub fn silu(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.silu_(self)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn softmax(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.softmax_(self)?;
        Ok(result)
    }

    /// Apply causality mask and return a new tensor.
    /// Shape: (batch * heads, seq_q, seq_kv) or (batch, heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    #[tracing::instrument(skip_all)]
    pub fn apply_causality_mask(&self, offset: usize) -> Result<Self> {
        let result = self.copy()?;
        result.apply_causality_mask_(offset)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn rms_norm(&self, alpha: &Self, eps: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rms_norm_(self, alpha, eps)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn layer_norm(&self, weight: &Self, bias: &Self, eps: f32) -> Result<Self> {
        self.layer_norm_rm(weight, bias, eps, true)
    }

    #[tracing::instrument(skip_all)]
    pub fn layer_norm_rm(
        &self,
        weight: &Self,
        bias: &Self,
        eps: f32,
        remove_mean: bool,
    ) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.layer_norm_(self, weight, bias, eps, remove_mean)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn rope(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_(self, cos, sin, pos)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn rope_i(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_i_(self, cos, sin, pos)?;
        Ok(result)
    }

    /// 1D convolution.
    /// Input: (batch, in_channels, length)
    /// Kernel: (out_channels, in_channels/groups, kernel_size)
    /// Output: (batch, out_channels, out_length)
    #[tracing::instrument(skip_all)]
    pub fn conv1d(
        &self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_channels, length) = self.dims3()?;
        let (out_channels, kernel_in_channels, kernel_size) = kernel.dims3()?;

        if !in_channels.is_multiple_of(groups) {
            crate::bail!("in_channels ({in_channels}) must be divisible by groups ({groups})");
        }
        if !out_channels.is_multiple_of(groups) {
            crate::bail!("out_channels ({out_channels}) must be divisible by groups ({groups})",);
        }
        if kernel_in_channels != in_channels / groups {
            crate::bail!(
                "kernel in_channels/groups mismatch: expected {}, got {kernel_in_channels}",
                in_channels / groups,
            );
        }

        // Compute output length
        let out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        let mut result =
            unsafe { Tensor::alloc_uninit((batch, out_channels, out_length), self.device()) }?;
        result.conv1d_(self, kernel, stride, padding, dilation, groups)?;

        // Add bias if provided
        if let Some(bias) = bias {
            let bias_dims = bias.dims();
            if bias_dims != [out_channels] {
                crate::bail!(
                    "bias shape mismatch: expected [{out_channels}], got {:?}",
                    bias.shape()
                );
            }
            // Reshape bias to (1, out_channels, 1) for broadcasting
            let bias = bias.reshape((1, out_channels, 1))?;
            result = result.broadcast_add(&bias)?;
        }

        Ok(result)
    }

    /// 1D transposed convolution.
    /// Input: (batch, in_channels, length)
    /// Kernel: (in_channels, out_channels/groups, kernel_size)
    /// Output: (batch, out_channels, out_length)
    #[tracing::instrument(skip_all)]
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_channels, length) = self.dims3()?;
        let (k_in_channels, out_channels_per_group, kernel_size) = kernel.dims3()?;

        let out_channels = out_channels_per_group * groups;

        if !in_channels.is_multiple_of(groups) {
            crate::bail!("in_channels ({in_channels}) must be divisible by groups ({groups})");
        }
        if k_in_channels != in_channels {
            crate::bail!(
                "kernel in_channels mismatch: expected {in_channels}, got {k_in_channels}",
            );
        }

        // Compute output length for transposed convolution
        // out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding
        let out_length = (length - 1) * stride + kernel_size + output_padding - 2 * padding;

        let mut result =
            unsafe { Tensor::alloc_uninit((batch, out_channels, out_length), self.device()) }?;
        result.conv_transpose1d_(self, kernel, stride, padding, output_padding, groups)?;

        // Add bias if provided
        if let Some(bias) = bias {
            let bias_dims = bias.dims();
            if bias_dims != [out_channels] {
                crate::bail!(
                    "bias shape mismatch: expected [{out_channels}], got {:?}",
                    bias.shape()
                );
            }
            // Reshape bias to (1, out_channels, 1) for broadcasting
            let bias = bias.reshape((1, out_channels, 1))?;
            result = result.broadcast_add(&bias)?;
        }

        Ok(result)
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqr_(self)?;
        Ok(result)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqrt_(self)?;
        Ok(result)
    }

    /// Element-wise reciprocal square root.
    pub fn rsqrt(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rsqrt_(self)?;
        Ok(result)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.abs_(self)?;
        Ok(result)
    }

    /// Sum along dimensions, keeping the dimensions (with size 1).
    #[tracing::instrument(skip_all)]
    pub fn sum_keepdim(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let mut dims: Vec<usize> = dims.into();
        // Sort dims in descending order so we can reduce from the end
        dims.sort_by(|a, b| b.cmp(a));
        dims.dedup();

        let mut result = self.copy()?;
        for &dim in &dims {
            if dim >= result.rank() {
                crate::bail!(
                    "sum_keepdim: dimension {} out of range for tensor of rank {}",
                    dim,
                    result.rank()
                );
            }
            // Reduce along dim, then reshape to keep the dimension with size 1
            let current_dims = result.dims().to_vec();

            // Output shape has dim reduced (removed)
            let mut reduced_dims: Vec<usize> = current_dims.clone();
            reduced_dims.remove(dim);
            if reduced_dims.is_empty() {
                reduced_dims.push(1);
            }

            let reduced = unsafe { Tensor::alloc_uninit(reduced_dims, result.device()) }?;
            reduced.reduce_sum_(&result, dim)?;

            // Reshape to keep the dimension with size 1
            let mut keepdim_shape: Vec<usize> = current_dims;
            keepdim_shape[dim] = 1;
            result = reduced.reshape(keepdim_shape)?;
        }

        Ok(result)
    }

    /// Maximum value along dimension.
    #[tracing::instrument(skip_all)]
    pub fn max<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "max dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        result.reduce_max_(self, dim)?;
        Ok(result)
    }

    /// Minimum value along dimension.
    #[tracing::instrument(skip_all)]
    pub fn min<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "min dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        result.reduce_min_(self, dim)?;
        Ok(result)
    }

    /// Argmin along dimension.
    /// Returns i64 indices.
    #[tracing::instrument(skip_all)]
    pub fn argmin<D: Dim>(&self, dim: D) -> Result<Tensor<i64, B>> {
        let dim = dim.to_index(self.shape(), "argmin dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result: Tensor<i64, B> = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        Self::reduce_argmin_(&result, self, dim)?;
        Ok(result)
    }

    /// GELU activation with erf.
    #[tracing::instrument(skip_all)]
    pub fn gelu_erf(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.gelu_erf_(self)?;
        Ok(result)
    }

    /// ELU activation.
    #[tracing::instrument(skip_all)]
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.elu_(self, alpha)?;
        Ok(result)
    }

    /// ReLU activation.
    #[tracing::instrument(skip_all)]
    pub fn relu(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.relu_(self)?;
        Ok(result)
    }

    /// Tanh activation.
    #[tracing::instrument(skip_all)]
    pub fn tanh(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.tanh_(self)?;
        Ok(result)
    }

    /// Sigmoid activation.
    #[tracing::instrument(skip_all)]
    pub fn sigmoid(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sigmoid_(self)?;
        Ok(result)
    }

    /// Expand tensor to a new shape (broadcasting).
    pub fn expand(&self, _shape: impl Into<crate::Shape>) -> Result<Self> {
        todo!("expand")
    }

    /// Repeat tensor along dimensions.
    pub fn repeat(&self, _repeats: impl Into<Vec<usize>>) -> Result<Self> {
        todo!("repeat")
    }

    /// Where condition: select from self or other based on condition.
    /// The condition should be a tensor of the same shape where non-zero values
    /// select from self, and zero values select from other.
    pub fn where_cond(&self, _condition: &Self, _other: &Self) -> Result<Self> {
        todo!("where_cond")
    }

    /// Pad with zeros along a dimension.
    pub fn pad_with_zeros<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
        let dims = self.dims();
        let dim_size = dims[dim];

        // Compute new shape
        let mut new_dims = dims.to_vec();
        new_dims[dim] = dim_size + left + right;
        let new_shape = crate::Shape::from(new_dims);

        // Create output tensor filled with zeros
        let result = Self::zeros(new_shape, self.device())?;

        if dim_size == 0 || self.elem_count() == 0 {
            return Ok(result);
        }

        // Copy original data to the padded position
        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let new_dim_size = dim_size + left + right;

        {
            let mut dst = result.storage_mut()?;
            let src = self.storage()?;
            B::copy2d(
                &mut *dst,
                &*src,
                outer_size,                // d1: number of outer blocks
                dim_size * inner_size,     // d2: elements per block
                new_dim_size * inner_size, // dst_s: stride in output
                dim_size * inner_size,     // src_s: stride in source
                left * inner_size,         // dst_o: offset to skip left padding
                0,                         // src_o: start from beginning of source
            )?;
        }

        Ok(result)
    }

    /// Pad by replicating boundary values.
    pub fn pad_with_same<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "pad_with_same")?;
        let dims = self.dims();
        let dim_size = dims[dim];

        if dim_size == 0 {
            crate::bail!("cannot pad_with_same on dimension with size 0");
        }

        // Compute new shape
        let mut new_dims = dims.to_vec();
        new_dims[dim] = dim_size + left + right;

        let result = unsafe { Self::alloc_uninit(new_dims, self.device()) }?;

        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let new_dim_size = dim_size + left + right;

        {
            let mut dst = result.storage_mut()?;
            let src = self.storage()?;

            // Copy original data to the center position
            B::copy2d(
                &mut *dst,
                &*src,
                outer_size,                // d1: number of outer blocks
                dim_size * inner_size,     // d2: elements per block
                new_dim_size * inner_size, // dst_s: stride in output
                dim_size * inner_size,     // src_s: stride in source
                left * inner_size,         // dst_o: offset to skip left padding
                0,                         // src_o: start from beginning of source
            )?;

            // Replicate first slice for left padding
            for l in 0..left {
                B::copy2d(
                    &mut *dst,
                    &*src,
                    outer_size,                // d1: number of outer blocks
                    inner_size,                // d2: one slice
                    new_dim_size * inner_size, // dst_s: stride in output
                    dim_size * inner_size,     // src_s: stride in source
                    l * inner_size,            // dst_o: position l in left padding
                    0,                         // src_o: first slice of source
                )?;
            }

            // Replicate last slice for right padding
            for r in 0..right {
                B::copy2d(
                    &mut *dst,
                    &*src,
                    outer_size,                         // d1: number of outer blocks
                    inner_size,                         // d2: one slice
                    new_dim_size * inner_size,          // dst_s: stride in output
                    dim_size * inner_size,              // src_s: stride in source
                    (left + dim_size + r) * inner_size, // dst_o: position after original data
                    (dim_size - 1) * inner_size,        // src_o: last slice of source
                )?;
            }
        }

        Ok(result)
    }

    pub fn matmul_t<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Self> {
        matmul_t(self, rhs)
    }

    pub fn matmul<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Self> {
        matmul(self, rhs)
    }
}

#[tracing::instrument(skip_all)]
pub fn matmul_with_t<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
    rhs_t: bool,
) -> Result<Tensor<T, B>> {
    if lhs.shape().rank() < 2 || rhs.shape().rank() < 2 {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "matmul requires at least 2D tensors",
        }
        .bt());
    }

    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();

    // Get M, K from lhs (last two dims)
    let lhs_m = lhs_dims[lhs_dims.len() - 2];
    let lhs_k = lhs_dims[lhs_dims.len() - 1];

    // Get K, N from rhs (last two dims), accounting for transpose
    let (rhs_k, rhs_n) = if rhs_t {
        (rhs_dims[rhs_dims.len() - 1], rhs_dims[rhs_dims.len() - 2])
    } else {
        (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1])
    };

    if lhs_k != rhs_k {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "inner dimensions do not match in matmul",
        }
        .bt());
    }

    // Check batch dimensions are compatible
    // rhs can be 2D (no batch) which broadcasts to any lhs batch
    let lhs_batch = &lhs_dims[..lhs_dims.len() - 2];
    let rhs_batch = &rhs_dims[..rhs_dims.len() - 2];
    if !rhs_batch.is_empty() && lhs_batch != rhs_batch {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "batch dimensions do not match in matmul",
        }
        .bt());
    }

    // Build output shape: lhs batch dims + [M, N]
    let mut target_shape = lhs_batch.to_vec();
    target_shape.push(lhs_m);
    target_shape.push(rhs_n);

    let dev = lhs.device();
    let result = unsafe { Tensor::<T, B>::alloc_uninit(target_shape, dev) }?;
    result.matmul_(lhs, rhs, rhs_t)?;
    Ok(result)
}

pub fn matmul<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
) -> Result<Tensor<T, B>> {
    matmul_with_t(lhs, rhs, false)
}

pub fn matmul_t<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
) -> Result<Tensor<T, B>> {
    matmul_with_t(lhs, rhs, true)
}
