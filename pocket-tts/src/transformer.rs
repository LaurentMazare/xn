use crate::rope::RotaryEmbedding;
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

/// State for StreamingMultiheadAttention.
#[derive(Debug)]
pub struct StreamingMHAState<T: WithDTypeF, B: Backend> {
    /// Key cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub k_cache: Tensor<T, B>,
    /// Value cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub v_cache: Tensor<T, B>,
    /// Current end position (number of tokens seen so far).
    pub current_end: usize,
}

#[allow(clippy::type_complexity)]
fn complete_kv<T: WithDTypeF, B: Backend>(
    k_cache: &Tensor<T, B>,
    v_cache: &Tensor<T, B>,
    current_end: usize,
    k: &Tensor<T, B>,
    v: &Tensor<T, B>,
) -> Result<(Tensor<T, B>, Tensor<T, B>, usize)> {
    let t = k.dim(1usize)?;

    k_cache.slice_set(k, 1usize, current_end)?;
    v_cache.slice_set(v, 1usize, current_end)?;

    let new_end = current_end + t;
    let keys = k_cache.narrow(1, 0..new_end)?.contiguous()?;
    let values = v_cache.narrow(1, 0..new_end)?.contiguous()?;
    Ok((keys, values, new_end))
}

fn materialize_causal_mask<T: WithDTypeF, B: Backend>(
    num_queries: usize,
    num_keys: usize,
    device: &B,
) -> Result<Tensor<T, B>> {
    let shift = num_keys - num_queries;
    // Upper-left triangular mask (causal)
    let mut data = Vec::with_capacity(num_queries * num_keys);
    for q in 0..num_queries {
        for k in 0..num_keys {
            if k <= q + shift {
                data.push(T::from_f32(0.0));
            } else {
                data.push(T::from_f32(f32::NEG_INFINITY));
            }
        }
    }
    Tensor::from_vec(data, (num_queries, num_keys), device)
}

/// Streaming multi-head attention (used by the flow LM transformer).
pub struct StreamingMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj: Linear<T, B>,
    out_proj: Linear<T, B>,
    pub embed_dim: usize,
    pub num_heads: usize,
    name: String,
}

impl<T: WithDTypeF, B: Backend> StreamingMultiheadAttention<T, B> {
    pub fn load(vb: &Path<B>, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = Linear::load(&vb.pp("in_proj"), embed_dim, out_dim)?;
        let out_proj = Linear::load(&vb.pp("out_proj"), embed_dim, embed_dim)?;
        let name = vb.prefix();
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, name })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingMHAState<T, B>> {
        let dim_per_head = self.embed_dim / self.num_heads;
        let shape = (batch_size, sequence_length, self.num_heads, dim_per_head);
        let dev = self.in_proj.device();
        let k_cache = Tensor::zeros(shape, dev)?;
        let v_cache = Tensor::zeros(shape, dev)?;
        Ok(StreamingMHAState { k_cache, v_cache, current_end: 0 })
    }

    #[tracing::instrument(name = "attn", skip_all)]
    pub fn forward(
        &self,
        query: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        state: &mut StreamingMHAState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let offset = state.current_end;

        let projected = self.in_proj.forward(query)?;
        // Split into q, k, v by narrowing on the last dimension
        let ed = self.embed_dim;
        let q = projected.narrow(2, 0..ed)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k =
            projected.narrow(2, ed..2 * ed)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = projected.narrow(2, 2 * ed..3 * ed)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads,
            d,
        ))?;

        // Apply RoPE: q, k are [b, t, h, d]
        let (q, k) = rope.forward(&q, &k, offset)?;

        // Complete KV cache: k, v are [b, t, h, d]
        let (k, v, new_end) = complete_kv(&state.k_cache, &state.v_cache, offset, &k, &v)?;
        state.current_end = new_end;

        // Causal mask
        let kv_len = k.dim(1usize)?;
        let mask = materialize_causal_mask::<T, B>(t, kv_len, query.device())?;

        // Transpose to [b, h, t, d] for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        // Back to [b, t, h*d]
        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}
