use crate::{Backend, Dim, Result, Tensor, WithDTypeF};

/// Sample using to Gumbel max trick.
pub fn gumbel_max_f<T: WithDTypeF, B: Backend, D: Dim>(
    logits: &Tensor<T, B>,
    temperature: f32,
    dim: D,
) -> Result<Tensor<i64, B>> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        let logits = logits.to::<f32>()?;
        let rand_uniform = logits.rand_uniform_like(1e-7, 0.999)?;
        let minus_g = rand_uniform.log()?.neg()?.log()?;
        logits.sub(&minus_g.scale(temperature)?)?.argmax(dim)
    }
}

/// Sample according to the Gumbel-Softmax distribution.
pub fn gumbel_max_t<T: WithDTypeF, B: Backend, D: Dim>(
    logits: &Tensor<T, B>,
    temperature: &Tensor<f32, B>,
    dim: D,
) -> Result<Tensor<i64, B>> {
    let logits = logits.to::<f32>()?;
    let gumbel_noise = logits.rand_uniform_like(1e-7, 0.999)?.log()?.neg()?.log()?;
    let adjusted_logits = logits.sub(&gumbel_noise.broadcast_mul(temperature)?)?;
    adjusted_logits.argmax(dim)
}

pub enum FloatOrTensor<'a, B: Backend> {
    Float(f32),
    Tensor(&'a Tensor<f32, B>),
}

impl<'a, B: Backend> From<f32> for FloatOrTensor<'a, B> {
    fn from(value: f32) -> Self {
        FloatOrTensor::Float(value)
    }
}

impl<'a, B: Backend> From<&'a Tensor<f32, B>> for FloatOrTensor<'a, B> {
    fn from(value: &'a Tensor<f32, B>) -> Self {
        FloatOrTensor::Tensor(value)
    }
}

pub fn gumbel_max<'a, T: WithDTypeF, B: Backend, D: Dim>(
    logits: &Tensor<T, B>,
    temperature: impl Into<FloatOrTensor<'a, B>>,
    dim: D,
) -> Result<Tensor<i64, B>> {
    match temperature.into() {
        FloatOrTensor::Float(temp) => gumbel_max_f(logits, temp, dim),
        FloatOrTensor::Tensor(temp) => gumbel_max_t(logits, temp, dim),
    }
}
