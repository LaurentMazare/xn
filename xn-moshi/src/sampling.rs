use xn::{Backend, Result, Tensor, WithDTypeF};

/// Sample according to the Gumbel-Softmax distribution.
/// `logits` has shape `(batch, vocab)` and `temperature` shape `(batch, 1)`.
/// When the temperature is zero, this degenerates to argmax (greedy).
pub fn gumbel_max<T: WithDTypeF, B: Backend>(
    logits: &Tensor<T, B>,
    temperature: &Tensor<f32, B>,
) -> Result<Tensor<i64, B>> {
    // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
    let logits = logits.to::<f32>()?;
    let gumbel_noise = logits.rand_uniform_like(1e-7, 0.999)?.log()?.neg()?.log()?;
    let adjusted_logits = logits.sub(&gumbel_noise.broadcast_mul(temperature)?)?;
    adjusted_logits.argmax(xn::D::Minus1)
}
