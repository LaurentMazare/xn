use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

/// Simplified quantizer that only provides output projection for TTS.
pub struct DummyQuantizer<T: WithDTypeF, B: Backend> {
    output_proj_weight: Tensor<T, B>,
    output_proj_bias: Option<Tensor<T, B>>,
    pub dimension: usize,
    pub output_dimension: usize,
}

impl<T: WithDTypeF, B: Backend> DummyQuantizer<T, B> {
    pub fn load(vb: &Path<B>, dimension: usize, output_dimension: usize) -> Result<Self> {
        let vb = vb.pp("output_proj");
        let output_proj_weight = vb.tensor("weight", (output_dimension, dimension, 1))?;
        let output_proj_bias =
            if vb.contains("bias") { Some(vb.tensor("bias", (output_dimension,))?) } else { None };
        Ok(Self { output_proj_weight, output_proj_bias, dimension, output_dimension })
    }

    /// Forward pass: Conv1d with kernel_size=1, no bias.
    /// Input: [B, dimension, T] -> Output: [B, output_dimension, T]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.conv1d(&self.output_proj_weight, self.output_proj_bias.as_ref(), 1, 0, 1, 1)
    }
}
