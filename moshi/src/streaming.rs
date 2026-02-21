use xn::{Backend, Result, Tensor, WithDType, WithDTypeF};

/// A tensor that may be empty, used in streaming contexts.
pub struct StreamTensor<T: WithDType, B: Backend>(Option<Tensor<T, B>>);

impl<T: WithDType, B: Backend> StreamTensor<T, B> {
    pub fn empty() -> Self {
        Self(None)
    }

    pub fn from_tensor(tensor: Tensor<T, B>) -> Self {
        Self(Some(tensor))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn as_option(&self) -> Option<&Tensor<T, B>> {
        self.0.as_ref()
    }

    pub fn reset(&mut self) {
        self.0 = None;
    }

    pub fn cat2(&self, rhs: &Self, dim: usize) -> Result<Self> {
        let xs = match (&self.0, &rhs.0) {
            (Some(lhs), Some(rhs)) => Some(Tensor::cat(&[lhs, rhs], dim)?),
            (Some(xs), None) | (None, Some(xs)) => Some(xs.clone()),
            (None, None) => None,
        };
        Ok(Self(xs))
    }

    pub fn seq_len(&self, dim: usize) -> Result<usize> {
        match &self.0 {
            None => Ok(0),
            Some(v) => v.dim(dim),
        }
    }

    pub fn narrow(&self, dim: usize, offset: usize, len: usize) -> Result<Self> {
        match &self.0 {
            None => Ok(Self::empty()),
            Some(t) => {
                let seq_len = t.dim(dim)?;
                if seq_len <= offset {
                    Ok(Self::empty())
                } else {
                    let actual_len = usize::min(len, seq_len - offset);
                    let t = t.narrow(dim, offset..offset + actual_len)?.contiguous()?;
                    Ok(Self::from_tensor(t))
                }
            }
        }
    }

    pub fn split(&self, dim: usize, lhs_len: usize) -> Result<(Self, Self)> {
        match &self.0 {
            None => Ok((Self::empty(), Self::empty())),
            Some(t) => {
                let seq_len = t.dim(dim)?;
                let lhs_len = usize::min(seq_len, lhs_len);
                if lhs_len == 0 {
                    Ok((Self::empty(), Self::from_tensor(t.clone())))
                } else {
                    let lhs = Self::from_tensor(t.narrow(dim, ..lhs_len)?.contiguous()?);
                    let rhs_len = seq_len - lhs_len;
                    let rhs = if rhs_len == 0 {
                        Self::empty()
                    } else {
                        Self::from_tensor(t.narrow(dim, lhs_len..lhs_len + rhs_len)?.contiguous()?)
                    };
                    Ok((lhs, rhs))
                }
            }
        }
    }
}

impl<T: WithDType, B: Backend> Default for StreamTensor<T, B> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: WithDType, B: Backend> From<()> for StreamTensor<T, B> {
    fn from(_: ()) -> Self {
        Self::empty()
    }
}

impl<T: WithDTypeF, B: Backend> From<Tensor<T, B>> for StreamTensor<T, B> {
    fn from(t: Tensor<T, B>) -> Self {
        Self::from_tensor(t)
    }
}

impl<T: WithDTypeF, B: Backend> From<Option<Tensor<T, B>>> for StreamTensor<T, B> {
    fn from(t: Option<Tensor<T, B>>) -> Self {
        Self(t)
    }
}

/// Mask for batch elements in streaming mode.
#[derive(Clone, Default)]
pub struct StreamMask(Option<Vec<bool>>);

impl StreamMask {
    pub fn empty() -> Self {
        Self(None)
    }

    pub fn new(mask: Vec<bool>) -> Self {
        Self(Some(mask))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn is_active(&self, batch_idx: usize) -> bool {
        self.0.as_ref().is_none_or(|v| v[batch_idx])
    }
}

impl From<()> for StreamMask {
    fn from(_: ()) -> Self {
        Self::empty()
    }
}
