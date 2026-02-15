use crate::Result;
use crate::{Backend, Shape, Tensor, UnaryOp, WithDType, WithDTypeF, shape::Dim};
use std::ops::RangeBounds;
use std::sync::{Arc, RwLock, RwLockReadGuard};

/// Merge adjacent compatible dimensions to reduce per-element index computation in copy_strided.
///
/// Two adjacent dims can merge when `strides[i] == dims[i+1] * strides[i+1]`.
/// Size-1 dims are dropped since the index is always 0.
fn coalesce_dims(dims: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut c_dims: Vec<usize> = Vec::with_capacity(dims.len());
    let mut c_strides: Vec<usize> = Vec::with_capacity(strides.len());
    for (&d, &s) in dims.iter().zip(strides.iter()) {
        if d == 1 {
            continue;
        }
        if let Some(last_s) = c_strides.last_mut() {
            let last_d = c_dims.last_mut().unwrap();
            if *last_s == d * s {
                *last_d *= d;
                *last_s = s;
                continue;
            }
        }
        c_dims.push(d);
        c_strides.push(s);
    }
    if c_dims.is_empty() {
        c_dims.push(1);
        c_strides.push(1);
    }
    (c_dims, c_strides)
}

/// Try to compute strides for a reshaped view without copying.
///
/// Handles two cases beyond fully contiguous:
/// - Splitting one dimension into two: `old_dims[i]` matches `new_dims[j] * new_dims[j+1]` and
///   the rest of the dimensions are identical.
/// - Merging two adjacent dimensions: `old_dims[i] * old_dims[i+1]` matches `new_dims[j]`,
///   requires `old_strides[i] == old_dims[i+1] * old_strides[i+1]`.
fn reshape_strides(
    old_dims: &[usize],
    old_strides: &[usize],
    new_dims: &[usize],
) -> Option<Vec<usize>> {
    let old_rank = old_dims.len();
    let new_rank = new_dims.len();

    // Split: new_rank == old_rank + 1, one old dim becomes two new dims.
    if new_rank == old_rank + 1 {
        for i in 0..old_rank {
            // Try splitting old dim i into new dims i and i+1.
            if old_dims[i] != new_dims[i] {
                if old_dims[i] != new_dims[i] * new_dims[i + 1] {
                    return None;
                }
                // Remaining dims must match.
                if old_dims[..i] != new_dims[..i] || old_dims[i + 1..] != new_dims[i + 2..] {
                    return None;
                }
                let mut new_strides = Vec::with_capacity(new_rank);
                new_strides.extend_from_slice(&old_strides[..i]);
                new_strides.push(new_dims[i + 1] * old_strides[i]);
                new_strides.push(old_strides[i]);
                new_strides.extend_from_slice(&old_strides[i + 1..]);
                return Some(new_strides);
            }
        }
        // Difference is at the end — shouldn't happen if elem counts match.
        return None;
    }

    // Merge: new_rank == old_rank - 1, two adjacent old dims become one new dim.
    if new_rank + 1 == old_rank {
        for i in 0..new_rank {
            if new_dims[i] != old_dims[i] {
                if i + 1 >= old_rank || new_dims[i] != old_dims[i] * old_dims[i + 1] {
                    return None;
                }
                // Check stride compatibility.
                if old_strides[i] != old_dims[i + 1] * old_strides[i + 1] {
                    return None;
                }
                // Remaining dims must match.
                if new_dims[..i] != old_dims[..i] || new_dims[i + 1..] != old_dims[i + 2..] {
                    return None;
                }
                let mut new_strides = Vec::with_capacity(new_rank);
                new_strides.extend_from_slice(&old_strides[..i]);
                new_strides.push(old_strides[i + 1]);
                new_strides.extend_from_slice(&old_strides[i + 2..]);
                return Some(new_strides);
            }
        }
        return None;
    }

    None
}

#[derive(Clone)]
pub struct TensorView<T: WithDType, B: Backend> {
    pub(crate) data: Arc<RwLock<B::Storage<T>>>,
    pub(crate) shape: Shape,
    pub(crate) device: B,
    pub(crate) strides: Vec<usize>,
    pub(crate) start_offset: usize,
}

impl<T: WithDType, B: Backend> From<Tensor<T, B>> for TensorView<T, B> {
    fn from(inner: Tensor<T, B>) -> Self {
        let strides = inner.shape().stride_contiguous();
        Self {
            data: inner.data,
            shape: inner.shape,
            strides,
            device: inner.device,
            start_offset: 0,
        }
    }
}

impl<T: WithDType, B: Backend> From<&Tensor<T, B>> for TensorView<T, B> {
    fn from(inner: &Tensor<T, B>) -> Self {
        let strides = inner.shape().stride_contiguous();
        Self {
            data: inner.data.clone(),
            shape: inner.shape.clone(),
            strides,
            device: inner.device.clone(),
            start_offset: 0,
        }
    }
}

impl<T: WithDType, B: Backend> TensorView<T, B> {
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn storage_and_offset(
        &self,
    ) -> Result<(std::sync::RwLockReadGuard<'_, B::Storage<T>>, usize)> {
        let s = self.data.read().map_err(|e| {
            crate::Error::msg(format!("failed to borrow tensor storage immutably: {}", e))
        })?;
        Ok((s, self.start_offset))
    }

    pub fn storage_mut_and_offset(
        &self,
    ) -> Result<(std::sync::RwLockWriteGuard<'_, B::Storage<T>>, usize)> {
        let s = self.data.write().map_err(|e| {
            crate::Error::msg(format!("failed to borrow tensor storage mutably: {}", e))
        })?;
        Ok((s, self.start_offset))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Flatten dimensions d1 to d2 (inclusive on both sides).
    pub fn flatten<D1: Dim, D2: Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        let d1 = d1.to_index(&self.shape, "flatten")?;
        let d2 = d2.to_index(&self.shape, "flatten")?;
        if d2 < d1 {
            crate::bail!("flatten incorrect dim ordering {d1} {d2}")
        }
        let dims = self.dims();
        let strides = self.strides();
        for i in d1..d2 {
            if strides[i + 1] * dims[i + 1] != strides[i] {
                crate::bail!(
                    "cannot flatten, block is not contiguous {dims:?} {strides:?} {d1} {d2}"
                )
            }
        }
        let d = dims[d1..d2 + 1].iter().product();
        let dst_dims = [&dims[..d1], &[d], &dims[d2 + 1..]].concat();
        let dst_strides = [&strides[..d1], &strides[d2..]].concat();
        Ok(Self {
            data: self.data.clone(),
            shape: dst_dims.into(),
            strides: dst_strides,
            start_offset: self.start_offset,
            device: self.device.clone(),
        })
    }

    /// Expand the specified dimension into a list of subdimensions.
    pub fn expand<D: Dim, S: Into<Shape>>(&self, d: D, s: S) -> Result<Self> {
        let s = s.into();
        let d = d.to_index(&self.shape, "expand")?;
        let dims = self.dims();
        let strides = self.strides();
        if dims[d] != s.elem_count() {
            crate::bail!("expand incorrect number of elements in target {s:?} {}", dims[d])
        }
        let dst_dims = [&dims[..d], s.dims(), &dims[d + 1..]].concat();
        let s_strides = s.stride_contiguous();
        let dst_strides = [&strides[..d], &s_strides, &strides[d + 1..]].concat();
        Ok(Self {
            data: self.data.clone(),
            shape: dst_dims.into(),
            strides: dst_strides,
            start_offset: self.start_offset,
            device: self.device.clone(),
        })
    }

    pub fn narrow<D: Dim>(&self, dim: D, range: impl RangeBounds<usize>) -> Result<Self> {
        let dim = dim.to_index(&self.shape, "narrow")?;
        let mut dims = self.shape.dims().to_vec();
        let (start, len) = crate::tensor::resolve_range(range, dims[dim]);
        if start + len > dims[dim] {
            crate::bail!("out-of-bounds in narrow on {dim}, {start} + {len} > {}", dims[dim])
        }
        dims[dim] = len;
        Ok(Self {
            data: self.data.clone(),
            start_offset: self.start_offset + self.strides[dim] * start,
            shape: Shape::from(dims),
            strides: self.strides.clone(),
            device: self.device.clone(),
        })
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(&self.shape, "transpose")?;
        let dim2 = dim2.to_index(&self.shape, "transpose")?;
        let mut strides = self.strides.to_vec();
        let mut dims = self.dims().to_vec();
        dims.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Ok(Self {
            data: self.data.clone(),
            shape: Shape::from(dims),
            strides,
            start_offset: self.start_offset,
            device: self.device.clone(),
        })
    }

    pub fn contiguous(&self) -> Result<Tensor<T, B>> {
        if self.is_contiguous() && self.start_offset == 0 {
            return Ok(Tensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                device: self.device.clone(),
                _marker: std::marker::PhantomData,
            });
        }
        let result: Tensor<T, B> =
            unsafe { Tensor::alloc_uninit(self.shape.clone(), &self.device) }?;
        {
            let src_data: RwLockReadGuard<'_, B::Storage<T>> = self.data.read().map_err(|e| {
                crate::Error::msg(format!("failed to borrow tensor storage immutably: {}", e))
            })?;
            let mut dst_data = result.storage_mut()?;
            let (c_dims, c_strides) = coalesce_dims(self.dims(), &self.strides);
            B::copy_strided(&mut dst_data, &*src_data, self.start_offset, &c_dims, &c_strides)?;
        }
        Ok(result)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let target_shape = shape.into();
        let target_dims = target_shape.dims();
        let src_dims = self.dims();
        let src_strides = self.strides();
        let target_rank = target_dims.len();
        let src_rank = src_dims.len();

        if target_rank < src_rank {
            crate::bail!(
                "broadcast_as: target rank {target_rank} is less than source rank {src_rank}"
            )
        }

        let rank_diff = target_rank - src_rank;
        let mut new_strides = vec![0usize; target_rank];

        for i in 0..target_rank {
            if i < rank_diff {
                new_strides[i] = 0;
            } else {
                let src_i = i - rank_diff;
                if src_dims[src_i] == target_dims[i] {
                    new_strides[i] = src_strides[src_i];
                } else if src_dims[src_i] == 1 {
                    new_strides[i] = 0;
                } else {
                    crate::bail!(
                        "broadcast_as: cannot broadcast dim {i} from {} to {}",
                        src_dims[src_i],
                        target_dims[i]
                    )
                }
            }
        }

        Ok(Self {
            data: self.data.clone(),
            shape: target_shape,
            strides: new_strides,
            start_offset: self.start_offset,
            device: self.device.clone(),
        })
    }

    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation =
            idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            crate::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                idxs
            )
        }
        let strides = self.strides();
        let dims = self.dims();
        let mut perm_strides = strides.to_vec();
        let mut perm_dims = dims.to_vec();
        for (i, &idx) in idxs.iter().enumerate() {
            perm_strides[i] = strides[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self {
            data: self.data.clone(),
            shape: Shape::from(perm_dims),
            strides: perm_strides,
            start_offset: self.start_offset,
            device: self.device.clone(),
        })
    }

    pub fn reshape(&self, shape: impl crate::shape::ShapeWithOneHole) -> Result<Self> {
        let shape = shape.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            crate::bail!(
                "reshape: cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                self.elem_count(),
                shape,
                shape.elem_count()
            );
        }
        if let Some(new_strides) = reshape_strides(self.dims(), &self.strides, shape.dims()) {
            return Ok(Self {
                data: self.data.clone(),
                shape,
                strides: new_strides,
                start_offset: self.start_offset,
                device: self.device.clone(),
            });
        }
        let t = self.contiguous()?;
        TensorView::from(t).reshape(shape)
    }
}

impl<T: WithDTypeF, B: Backend> TensorView<T, B> {
    fn apply_unary(&self, op: UnaryOp) -> Result<Tensor<T, B>> {
        let result = self.contiguous()?;
        let len = result.elem_count();
        let mut dst = result.storage_mut()?;
        B::inplace_unary(&mut *dst, len, op)?;
        drop(dst);
        Ok(result)
    }

    pub fn matmul_t<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Tensor<T, B>> {
        crate::ops::matmul_t(self, rhs)
    }

    pub fn matmul<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Tensor<T, B>> {
        crate::ops::matmul(self, rhs)
    }

    pub fn sigmoid(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Sigmoid)
    }

    pub fn tanh(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Tanh)
    }

    pub fn relu(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Relu)
    }

    pub fn silu(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Silu)
    }

    pub fn gelu_erf(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::GeluErf)
    }

    pub fn elu(&self, alpha: f32) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Elu { alpha })
    }

    pub fn cos(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Cos)
    }

    pub fn sin(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Sin)
    }

    pub fn sqr(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Sqr)
    }

    pub fn sqrt(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Sqrt)
    }

    pub fn abs(&self) -> Result<Tensor<T, B>> {
        self.apply_unary(UnaryOp::Abs)
    }
}

pub trait TensorOrView<T: WithDType, B: Backend> {
    fn shape(&self) -> &Shape;
    fn strides(&self) -> std::borrow::Cow<'_, [usize]>;
    fn storage_and_offset(&self) -> Result<(std::sync::RwLockReadGuard<'_, B::Storage<T>>, usize)>;
    fn device(&self) -> &B;
    fn rank(&self) -> usize {
        self.shape().rank()
    }
    fn dims(&self) -> &[usize] {
        self.shape().dims()
    }
}

impl<T: WithDType, B: Backend> TensorOrView<T, B> for Tensor<T, B> {
    fn shape(&self) -> &Shape {
        self.shape()
    }

    fn storage_and_offset(&self) -> Result<(std::sync::RwLockReadGuard<'_, B::Storage<T>>, usize)> {
        let s = self.storage()?;
        Ok((s, 0))
    }

    fn strides(&self) -> std::borrow::Cow<'_, [usize]> {
        std::borrow::Cow::Owned(self.shape().stride_contiguous())
    }

    fn device(&self) -> &B {
        self.device()
    }
}

impl<T: WithDType, B: Backend> TensorOrView<T, B> for TensorView<T, B> {
    fn shape(&self) -> &Shape {
        self.shape()
    }
    fn storage_and_offset(&self) -> Result<(std::sync::RwLockReadGuard<'_, B::Storage<T>>, usize)> {
        self.storage_and_offset()
    }
    fn strides(&self) -> std::borrow::Cow<'_, [usize]> {
        std::borrow::Cow::Borrowed(self.strides())
    }
    fn device(&self) -> &B {
        &self.device
    }
}
