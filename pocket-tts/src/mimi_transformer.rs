use crate::layer_scale::LayerScale;
use crate::rope::RotaryEmbedding;
use crate::transformer::{StreamingMHAState, StreamingMultiheadAttention};
use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

// ---- KV Cache ----

/// Simple append-based KV cache with optional context window trimming.
pub struct KvCache<T: WithDTypeF, B: Backend> {
    k: Option<Tensor<T, B>>,
    v: Option<Tensor<T, B>>,
    max_seq_len: usize,
    absolute_offset: usize,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    pub fn new(max_seq_len: usize) -> Self {
        Self { k: None, v: None, max_seq_len, absolute_offset: 0 }
    }

    pub fn current_seq_len(&self) -> Result<usize> {
        let l = match &self.k {
            Some(k) => k.dim(2)?, // k shape: [b, h, seq, d]
            None => 0,
        };
        Ok(l)
    }

    /// Append new k, v (shape [b, h, t, d]) and return full (k, v).
    /// Trims to max_seq_len if exceeded.
    pub fn append(
        &mut self,
        new_k: &Tensor<T, B>,
        new_v: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        let seq_len = k.dim(2)?;
        let (k, v) = if seq_len > self.max_seq_len {
            let trim = seq_len - self.max_seq_len;
            (
                k.narrow(2, trim..trim + self.max_seq_len)?.contiguous()?,
                v.narrow(2, trim..trim + self.max_seq_len)?.contiguous()?,
            )
        } else {
            (k, v)
        };

        let new_tokens = new_k.dim(2)?;
        self.absolute_offset += new_tokens;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

// ---- State types ----

pub enum LayerAttentionState<T: WithDTypeF, B: Backend> {
    Mimi(KvCache<T, B>),
    FlowLm(StreamingMHAState<T, B>),
}

pub struct StreamingTransformerState<T: WithDTypeF, B: Backend> {
    pub layer_states: Vec<LayerAttentionState<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformerState<T, B> {
    pub fn current_seq_len(&self) -> Result<usize> {
        if self.layer_states.is_empty() {
            return Ok(0);
        }
        let v = match &self.layer_states[0] {
            LayerAttentionState::Mimi(cache) => cache.current_seq_len()?,
            LayerAttentionState::FlowLm(state) => state.current_end,
        };
        Ok(v)
    }
}

// ---- MimiStreamingMultiheadAttention ----
// Uses KV cache with context window.

pub struct MimiStreamingMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj: Linear<T, B>,
    out_proj: Linear<T, B>,
    embed_dim: usize,
    num_heads: usize,
    context: usize,
}

impl<T: WithDTypeF, B: Backend> MimiStreamingMultiheadAttention<T, B> {
    pub fn load(vb: &Path<B>, embed_dim: usize, num_heads: usize, context: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = Linear::load(&vb.pp("in_proj"), embed_dim, out_dim)?;
        let out_proj = Linear::load(&vb.pp("out_proj"), embed_dim, embed_dim)?;
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, context })
    }

    pub fn init_state(&self) -> Result<KvCache<T, B>> {
        Ok(KvCache::new(self.context))
    }

    pub fn forward(
        &self,
        query: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        state: &mut KvCache<T, B>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let offset = state.absolute_offset;

        let projected = self.in_proj.forward(query)?;
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let q = packed.narrow(2, 0..1)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k = packed.narrow(2, 1..2)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = packed.narrow(2, 2..3)?.contiguous()?.reshape((b, t, self.num_heads, d))?;

        // RoPE on [b, t, h, d]
        let (q, k) = rope.forward(&q, &k, offset)?;

        // To [b, h, t, d]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // KV cache with context trimming
        let (k, v) = state.append(&k, &v)?;

        // Attention with causal mask
        let scale = T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = attn.apply_causality_mask(offset)?;
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

// ---- StreamingTransformerLayer ----

enum AttentionKind<T: WithDTypeF, B: Backend> {
    Mimi(MimiStreamingMultiheadAttention<T, B>),
    FlowLm(StreamingMultiheadAttention<T, B>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mimi,
    FlowLm,
}

pub struct StreamingTransformerLayer<T: WithDTypeF, B: Backend> {
    self_attn: AttentionKind<T, B>,
    norm1: LayerNorm<T, B>,
    norm2: LayerNorm<T, B>,
    linear1: Linear<T, B>,
    linear2: Linear<T, B>,
    layer_scale_1: Option<LayerScale<T, B>>,
    layer_scale_2: Option<LayerScale<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformerLayer<T, B> {
    pub fn load(
        vb: &Path<B>,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        layer_scale: Option<f64>,
        kind: Kind,
    ) -> Result<Self> {
        let self_attn = match kind {
            Kind::Mimi => AttentionKind::Mimi(MimiStreamingMultiheadAttention::load(
                &vb.pp("self_attn"),
                d_model,
                num_heads,
                context.unwrap_or(250),
            )?),
            Kind::FlowLm => AttentionKind::FlowLm(StreamingMultiheadAttention::load(
                &vb.pp("self_attn"),
                d_model,
                num_heads,
            )?),
        };

        let norm1 = LayerNorm::load(&vb.pp("norm1"), d_model, 1e-5)?;
        let norm2 = LayerNorm::load(&vb.pp("norm2"), d_model, 1e-5)?;
        let linear1 = Linear::load(&vb.pp("linear1"), d_model, dim_feedforward)?;
        let linear2 = Linear::load(&vb.pp("linear2"), dim_feedforward, d_model)?;

        let layer_scale_1 = if layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_1"), d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_2"), d_model)?)
        } else {
            None
        };

        Ok(Self { self_attn, norm1, norm2, linear1, linear2, layer_scale_1, layer_scale_2 })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<LayerAttentionState<T, B>> {
        let s = match &self.self_attn {
            AttentionKind::Mimi(attn) => LayerAttentionState::Mimi(attn.init_state()?),
            AttentionKind::FlowLm(attn) => {
                LayerAttentionState::FlowLm(attn.init_state(batch_size, sequence_length)?)
            }
        };
        Ok(s)
    }

    #[tracing::instrument(name = "transformer-layer", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        state: &mut LayerAttentionState<T, B>,
    ) -> Result<Tensor<T, B>> {
        // Self-attention block: x + layer_scale_1(attn(norm1(x)))
        let norm1 = self.norm1.forward(x)?;
        let mut attn_out = match (&self.self_attn, state) {
            (AttentionKind::Mimi(attn), LayerAttentionState::Mimi(cache)) => {
                attn.forward(&norm1, rope, cache)?
            }
            (AttentionKind::FlowLm(attn), LayerAttentionState::FlowLm(mha_state)) => {
                attn.forward(&norm1, rope, mha_state)?
            }
            _ => xn::bail!("attention kind and state type mismatch"),
        };
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let x = x.add(&attn_out)?;

        // FF block: x + layer_scale_2(ff(norm2(x)))
        let norm2 = self.norm2.forward(&x)?;
        let mut ff_out = self.linear1.forward(&norm2)?;
        ff_out = ff_out.gelu_erf()?;
        ff_out = self.linear2.forward(&ff_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            ff_out = ls.forward(&ff_out)?;
        }
        x.add(&ff_out)
    }
}

// ---- StreamingTransformer ----

pub struct StreamingTransformer<T: WithDTypeF, B: Backend> {
    pub layers: Vec<StreamingTransformerLayer<T, B>>,
    rope: RotaryEmbedding<T, B>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformer<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: Kind,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let max_seq_len = 8192;
        let rope = RotaryEmbedding::new(head_dim, max_seq_len, max_period, vb.device())?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(StreamingTransformerLayer::load(
                &vb.pp("layers").pp(i),
                d_model,
                num_heads,
                dim_feedforward,
                context,
                layer_scale,
                kind,
            )?);
        }

        Ok(Self { layers, rope })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<T, B>> {
        let layer_states = self
            .layers
            .iter()
            .map(|l| l.init_state(batch_size, sequence_length))
            .collect::<Result<Vec<_>>>()?;
        Ok(StreamingTransformerState { layer_states })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingTransformerState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut x = x.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &self.rope, layer_state)?;
        }
        Ok(x)
    }
}

// ---- ProjectedTransformer ----

pub struct ProjectedTransformer<T: WithDTypeF, B: Backend> {
    pub transformer: StreamingTransformer<T, B>,
    input_proj: Option<Linear<T, B>>,
    output_projs: Vec<Option<Linear<T, B>>>,
}

impl<T: WithDTypeF, B: Backend> ProjectedTransformer<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        input_dimension: usize,
        output_dimensions: &[usize],
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        context: usize,
        max_period: f32,
        dim_feedforward: usize,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            d_model,
            num_heads,
            num_layers,
            layer_scale,
            dim_feedforward,
            Some(context),
            max_period,
            Kind::Mimi,
        )?;

        let input_proj = if d_model != input_dimension {
            Some(Linear::load(&vb.pp("input_proj"), input_dimension, d_model)?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &out_dim) in output_dimensions.iter().enumerate() {
            if d_model == out_dim {
                output_projs.push(None);
            } else {
                let proj = Linear::load(&vb.pp("output_proj").pp(i), d_model, out_dim)?;
                output_projs.push(Some(proj));
            }
        }

        Ok(Self { transformer, input_proj, output_projs })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<T, B>> {
        self.transformer.init_state(batch_size, sequence_length)
    }

    /// Forward pass. Input x is [B, C, T] (conv layout).
    #[tracing::instrument(name = "transformer", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingTransformerState<T, B>,
    ) -> Result<Vec<Tensor<T, B>>> {
        // [B, C, T] -> [B, T, C]
        let x = x.transpose(1, 2)?.contiguous()?;

        let x = match &self.input_proj {
            Some(proj) => proj.forward(&x)?,
            None => x,
        };

        let z = self.transformer.forward(&x, state)?;

        let mut ys = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let y = match proj {
                Some(p) => p.forward(&z)?,
                None => z.clone(),
            };
            ys.push(y.transpose(1, 2)?.contiguous()?);
        }
        Ok(ys)
    }
}
