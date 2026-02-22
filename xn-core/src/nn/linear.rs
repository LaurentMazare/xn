use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDTypeF};

pub struct Linear<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Option<Tensor<T, B>>,
}

impl<T: WithDTypeF, B: Backend> Linear<T, B> {
    pub fn new(weight: Tensor<T, B>) -> Self {
        Self { weight, bias: None }
    }

    pub fn with_bias(self, bias: Tensor<T, B>) -> Self {
        Self { bias: Some(bias), ..self }
    }

    pub fn load<V: std::borrow::Borrow<Path<B>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let vb = vb.borrow();
        let weight = vb.tensor("weight", (out_features, in_features))?;
        Ok(Self::new(weight))
    }

    pub fn load_b<V: std::borrow::Borrow<Path<B>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let vb = vb.borrow();
        let weight = vb.tensor("weight", (out_features, in_features))?;
        let bias = vb.tensor("bias", (out_features,))?;
        Ok(Self::new(weight).with_bias(bias))
    }

    pub fn load_o<V: std::borrow::Borrow<Path<B>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb = vb.borrow();
        let weight = vb.tensor("weight", (out_features, in_features))?;
        let slf = Self::new(weight);
        if bias {
            let bias = vb.tensor("bias", (out_features,))?;
            Ok(slf.with_bias(bias))
        } else {
            Ok(slf)
        }
    }

    pub fn forward<X: crate::TensorOrView<T, B>>(&self, x: &X) -> Result<Tensor<T, B>> {
        // weight: (out_features, in_features)
        // x: (..., in_features)
        // output: (..., out_features)
        let x = crate::ops::matmul_t(x, &self.weight)?;
        let x = match &self.bias {
            Some(bias) => x.broadcast_add(bias)?,
            None => x,
        };
        Ok(x)
    }

    pub fn device(&self) -> &B {
        self.weight.device()
    }
}
