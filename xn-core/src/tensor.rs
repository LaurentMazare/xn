use crate::{Backend, DType, Error, Result, Shape, TensorView, WithDType, shape::Dim};
use std::ops::RangeBounds;
use std::sync::{Arc, RwLock};

/// Resolve a `RangeBounds<usize>` into `(start, len)` given a dimension size.
pub(crate) fn resolve_range(range: impl RangeBounds<usize>, dim_size: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        std::ops::Bound::Included(&s) => s,
        std::ops::Bound::Excluded(&s) => s + 1,
        std::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        std::ops::Bound::Included(&e) => e + 1,
        std::ops::Bound::Excluded(&e) => e,
        std::ops::Bound::Unbounded => dim_size,
    };
    (start, end.saturating_sub(start))
}

impl<T: WithDType, B: Backend> Clone for Tensor<T, B> {
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct Tensor<T: WithDType, B: Backend> {
    pub(crate) data: Arc<RwLock<B::Storage<T>>>,
    pub(crate) shape: Shape,
    pub(crate) device: B,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

pub enum TypedTensor<'a, B: Backend> {
    F16(&'a Tensor<half::f16, B>),
    BF16(&'a Tensor<half::bf16, B>),
    F32(&'a Tensor<f32, B>),
    I64(&'a Tensor<i64, B>),
    U8(&'a Tensor<u8, B>),
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dim(&self, index: impl Dim) -> Result<usize> {
        self.shape.dim(index)
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    /// Borrow the underlying storage immutably.
    /// Returns an error if the storage is currently mutably borrowed.
    pub fn storage(&self) -> Result<std::sync::RwLockReadGuard<'_, B::Storage<T>>> {
        let s = self.data.read().map_err(|e| {
            crate::Error::msg(format!("failed to borrow tensor storage immutably: {}", e))
        })?;
        Ok(s)
    }

    /// Borrow the underlying storage mutably.
    /// Returns an error if the storage is currently borrowed (mutably or immutably).
    pub fn storage_mut(&self) -> Result<std::sync::RwLockWriteGuard<'_, B::Storage<T>>> {
        let s = self.data.write().map_err(|e| {
            crate::Error::msg(format!("failed to borrow tensor storage mutably: {}", e))
        })?;
        Ok(s)
    }

    pub fn zeros(shape: impl Into<Shape>, device: &B) -> Result<Self> {
        Self::full(T::zero(), shape, device)
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let len = self.elem_count();
        let data = self.storage()?;
        let data_cow = B::data(&*data, len)?;
        Ok(data_cow.into_owned())
    }

    pub fn full(value: T, shape: impl Into<Shape>, device: &B) -> Result<Self> {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let mut data = unsafe { B::alloc_uninit(size, device)? };
        B::fill(&mut data, value, size)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(data)),
            shape,
            device: device.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<TensorView<T, B>> {
        let view = TensorView::from(self);
        view.broadcast_as(shape)
    }

    /// Reshape the tensor to a new shape with the same number of elements.
    /// This operation shares the underlying data (no copy).
    #[tracing::instrument(skip_all)]
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
        // Share the underlying data instead of copying
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape,
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Extract a slice of the tensor along a given dimension.
    /// Returns a `TensorView` (zero-copy). Call `.contiguous()?` on the result
    /// if you need a contiguous `Tensor`.
    #[tracing::instrument(skip_all)]
    pub fn narrow(&self, dim: usize, range: impl RangeBounds<usize>) -> Result<TensorView<T, B>> {
        TensorView::from(self).narrow(dim, range)
    }

    /// # Safety
    /// The returned tensor's data is uninitialized.
    pub unsafe fn alloc_uninit(shape: impl Into<Shape>, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let size = shape.elem_count();
        let data = unsafe { B::alloc_uninit(size, dev)? };
        Ok(Tensor {
            data: Arc::new(RwLock::new(data)),
            shape,
            device: dev.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn index_select(&self, indices: &[u32], dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index_select dim")?;
        let dim_size = self.dim(dim)?;
        for (i, &idx) in indices.iter().enumerate() {
            if idx as usize >= dim_size {
                crate::bail!(
                    "index_select: index {idx} at position {i} is out of bounds for dimension {dim} with size {dim_size}"
                );
            }
        }

        // Calculate output shape
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims[dim] = indices.len();
        let out_shape = Shape::from(out_dims);

        // Allocate output
        let dev = self.device();
        let out: Self = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;
        {
            let src_data = self.storage()?;
            let mut dst_data = out.storage_mut()?;
            B::index_select(&mut dst_data, &*src_data, indices, dim, self.dims())?;
        }
        Ok(out)
    }

    pub fn from_vec<S: crate::shape::ShapeWithOneHole>(
        data: Vec<T>,
        shape: S,
        dev: &B,
    ) -> Result<Self> {
        let shape = shape.into_shape(data.len())?;
        if data.len() != shape.elem_count() {
            crate::bail!(
                "from_vec: data length {} does not match shape {:?} with {} elements",
                data.len(),
                shape,
                shape.elem_count()
            );
        }
        let data = B::from_vec(data, dev)?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(data)),
            shape,
            device: dev.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Concatenate tensors along a given dimension.
    #[tracing::instrument(skip_all)]
    pub fn cat(tensors: &[&Self], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            crate::bail!("cat requires at least one tensor");
        }

        let first = tensors[0];
        let rank = first.rank();
        let dim = dim.to_index(first.shape(), "cat dim")?;

        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.rank() != rank {
                crate::bail!("cat: tensor {i} has rank {} but expected {rank}", t.rank());
            }
            for d in 0..rank {
                if d != dim && t.dims()[d] != first.dims()[d] {
                    crate::bail!(
                        "cat: tensor {i} has shape {:?} but expected dimension {d} to be {}",
                        t.shape(),
                        first.dims()[d]
                    );
                }
            }
        }

        // Calculate output shape
        let cat_dim_size: usize = tensors.iter().map(|t| t.dims()[dim]).sum();
        let mut out_dims: Vec<usize> = first.dims().to_vec();
        out_dims[dim] = cat_dim_size;
        let out_shape = Shape::from(out_dims);

        // Allocate output
        let dev = first.device();
        let out: Self = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;

        // Copy data from each tensor using copy2d
        // For contiguous tensors, data is laid out as: [outer dims][cat dim][inner dims]
        let outer_size: usize = if dim == 0 { 1 } else { out.dims()[..dim].iter().product() };
        let inner_size: usize = out.dims()[dim + 1..].iter().product::<usize>().max(1);

        let mut cat_offset = 0;
        {
            let mut out_data = out.storage_mut()?;
            for tensor in tensors {
                let t_cat_size = tensor.dims()[dim];
                let src_data = tensor.storage()?;
                // Copy using copy2d: outer_size rows of (t_cat_size * inner_size) elements
                B::copy2d(
                    &mut out_data,
                    &*src_data,
                    outer_size,                // d1: number of outer blocks
                    t_cat_size * inner_size,   // d2: elements per block from this tensor
                    cat_dim_size * inner_size, // dst_s: stride in output
                    t_cat_size * inner_size,   // src_s: stride in source
                    cat_offset * inner_size,   // dst_o: offset in output
                    0,                         // src_o: offset in source
                )?;
                cat_offset += t_cat_size;
            }
        }

        Ok(out)
    }

    /// Stack tensors along a new dimension.
    /// All tensors must have the same shape.
    /// The new dimension is inserted at position `dim`.
    pub fn stack(tensors: &[&Self], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            crate::bail!("stack requires at least one tensor");
        }

        let first = tensors[0];
        // For stack, dim can be 0..=rank (inserting a new dimension)
        let dim = dim.to_index_plus_one(first.shape(), "stack dim")?;

        // All tensors must have the same shape
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != first.shape() {
                crate::bail!(
                    "stack: tensor {i} has shape {:?} but expected {:?}",
                    t.shape(),
                    first.shape()
                );
            }
        }

        // Unsqueeze each tensor at dim, then concatenate
        let unsqueezed: Vec<Self> =
            tensors.iter().map(|t| t.unsqueeze(dim)).collect::<Result<Vec<_>>>()?;
        let unsqueezed_refs: Vec<&Self> = unsqueezed.iter().collect();
        Self::cat(&unsqueezed_refs, dim)
    }

    pub fn downcast(&self) -> Result<TypedTensor<'_, B>> {
        use crate::error::Context;
        let slf = self as &dyn std::any::Any;
        let tt = match T::DTYPE {
            DType::F16 => TypedTensor::F16(slf.downcast_ref().context("downcast to f16")?),
            DType::BF16 => TypedTensor::BF16(slf.downcast_ref().context("downcast to bf16")?),
            DType::F32 => TypedTensor::F32(slf.downcast_ref().context("downcast to f32")?),
            DType::I64 => TypedTensor::I64(slf.downcast_ref().context("downcast to i64")?),
            DType::U8 => TypedTensor::U8(slf.downcast_ref().context("downcast to u8")?),
        };
        Ok(tt)
    }

    /// Set the values on `self` using values from `src`. The copy starts at the specified
    /// `offset` for the target dimension `dim` on `self`.
    ///
    /// `self` and `src` must have the same shape except on dimension `dim` where the `self` size
    /// has to be greater than or equal to `offset` plus the `src` size.
    ///
    /// Note that this modifies `self` in place.
    #[tracing::instrument(skip_all)]
    pub fn slice_set(&self, src: &Self, dim: impl Dim, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice_set")?;

        // Check that tensors don't share storage
        if Arc::ptr_eq(&self.data, &src.data) {
            crate::bail!("slice_set: cannot use when self and src share their storage");
        }

        // Check ranks match
        if self.rank() != src.rank() {
            crate::bail!(
                "slice_set: rank mismatch, self has rank {} but src has rank {}",
                self.rank(),
                src.rank()
            );
        }

        // Check shapes are compatible
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim {
                if *v2 + offset > *v1 {
                    crate::bail!(
                        "slice_set: shape mismatch on target dim {dim}, dst size: {v1}, src size: {v2} + offset {offset}"
                    );
                }
            } else if v1 != v2 {
                crate::bail!(
                    "slice_set: shape mismatch on dim {dim_idx}, self has {v1} but src has {v2}"
                );
            }
        }

        // Compute copy parameters
        let block_size: usize = src.dims().iter().skip(1 + dim).product::<usize>().max(1);
        let d1: usize = src.dims().iter().take(dim).product::<usize>().max(1);
        let d2 = block_size * src.dims()[dim];
        let dst_o = offset * block_size;

        // Perform the copy
        let src_data = src.storage()?;
        let mut dst_data = self.storage_mut()?;
        B::copy2d(
            &mut dst_data,
            &*src_data,
            d1,
            d2,
            /* dst_s */ block_size * self.dims()[dim],
            /* src_s */ d2,
            dst_o,
            /* src_o */ 0,
        )?;

        Ok(())
    }
    pub fn slice_assign<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if self.rank() != src.rank() {
            crate::bail!("rank mismatch in slice_assign {} <> {}", self.rank(), src.rank())
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                crate::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                crate::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = offset * block_size;
        let src_o = 0;
        let dst_s = block_size * self.dims()[dim];
        let src_s = d2;
        let src_data = src.storage()?;
        let mut dst_data = self.storage_mut()?;
        B::copy2d(&mut dst_data, &*src_data, d1, d2, dst_s, src_s, dst_o, src_o)?;
        Ok(())
    }

    pub(crate) fn same_storage(&self, rhs: &Self) -> bool {
        let lhs: &RwLock<_> = self.data.as_ref();
        let rhs: &RwLock<_> = rhs.data.as_ref();
        std::ptr::eq(lhs, rhs)
    }

    fn scatter_checks(&self, indexes: &Tensor<i64, B>, source: &Self, dim: usize) -> Result<()> {
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        if indexes.dims() != source.dims() {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter (indexes, src)",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        Ok(())
    }

    pub fn scatter<D: Dim>(&self, indexes: &Tensor<i64, B>, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter")?;
        self.scatter_checks(indexes, source, dim)?;
        let result: Self = unsafe { Tensor::alloc_uninit(self.shape().clone(), &self.device) }?;
        {
            let src_data = self.storage()?;
            let mut dst_data = result.storage_mut()?;
            B::copy(&mut dst_data, &*src_data, self.elem_count())?;
        }
        {
            let mut dst_data = result.storage_mut()?;
            let src_data = source.storage()?;
            let ids_data = indexes.storage()?;
            B::scatter_set(&mut dst_data, &*src_data, &*ids_data, dim, self.dims(), source.dims())?;
        }
        Ok(result)
    }

    pub fn scatter_set<D: Dim>(
        &self,
        indexes: &Tensor<i64, B>,
        source: &Self,
        dim: D,
    ) -> Result<()> {
        if self.same_storage(source) {
            crate::bail!("cannot use scatter_set when self and src share their storage")
        }
        let dim = dim.to_index(self.shape(), "scatter-set")?;
        self.scatter_checks(indexes, source, dim)?;
        let mut dst_data = self.storage_mut()?;
        let src_data = source.storage()?;
        let ids_data = indexes.storage()?;
        B::scatter_set(&mut dst_data, &*src_data, &*ids_data, dim, self.dims(), source.dims())?;
        Ok(())
    }
}
