use crate::{Backend, Result, Tensor, WithDTypeF};

pub mod var_builder;
use var_builder::Path;
pub use var_builder::VB;

pub struct RmsNorm<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> RmsNorm<T, B> {
    pub fn new(weight: Tensor<T, B>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn load(vb: &Path<B>, dim: usize, eps: f32) -> Result<Self> {
        let weight = vb.tensor("weight", (dim,))?;
        Ok(Self::new(weight, eps))
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.rms_norm(&self.weight, self.eps)
    }

    pub fn device(&self) -> &B {
        self.weight.device()
    }
}

pub struct LayerNorm<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Tensor<T, B>,
    remove_mean: bool,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> LayerNorm<T, B> {
    pub fn new(weight: Tensor<T, B>, bias: Tensor<T, B>, eps: f32) -> Self {
        Self { weight, bias, eps, remove_mean: true }
    }

    pub fn remove_mean(mut self, remove_mean: bool) -> Self {
        self.remove_mean = remove_mean;
        self
    }

    pub fn load(vb: &Path<B>, dim: usize, eps: f32) -> Result<Self> {
        let weight = vb.tensor("weight", (dim,))?;
        let bias = vb.tensor("bias", (dim,))?;
        Ok(Self::new(weight, bias, eps))
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.layer_norm_rm(&self.weight, &self.bias, self.eps, self.remove_mean)
    }

    pub fn device(&self) -> &B {
        self.weight.device()
    }
}

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

    pub fn load(vb: &Path<B>, in_features: usize, out_features: usize) -> Result<Self> {
        let weight = vb.tensor("weight", (out_features, in_features))?;
        Ok(Self::new(weight))
    }

    pub fn load_b(vb: &Path<B>, in_features: usize, out_features: usize) -> Result<Self> {
        let weight = vb.tensor("weight", (out_features, in_features))?;
        let bias = vb.tensor("bias", (out_features,))?;
        Ok(Self::new(weight).with_bias(bias))
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
