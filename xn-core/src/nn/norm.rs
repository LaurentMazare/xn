use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDTypeF};

pub struct RmsNorm<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> RmsNorm<T, B> {
    pub fn new(weight: Tensor<T, B>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn load<V: std::borrow::Borrow<Path<B>>>(vb: V, dim: usize, eps: f32) -> Result<Self> {
        let vb = vb.borrow();
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

    pub fn load<V: std::borrow::Borrow<Path<B>>>(vb: V, dim: usize, eps: f32) -> Result<Self> {
        let vb = vb.borrow();
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
