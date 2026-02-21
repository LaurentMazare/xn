use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct LayerScale<T: WithDTypeF, B: Backend> {
    scale: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> LayerScale<T, B> {
    pub fn load(vb: &Path<B>, channels: usize) -> Result<Self> {
        let scale = vb.tensor("scale", (channels,))?;
        Ok(Self { scale })
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.broadcast_mul(&self.scale)
    }
}
