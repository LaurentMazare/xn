use crate::layer_scale::LayerScale;
use crate::rope::RotaryEmbedding;
use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

/// State for StreamingMultiheadAttention.
#[derive(Debug, Clone)]
pub struct StreamingMHAState<T: WithDTypeF, B: Backend> {
    /// Key cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub k_cache: Tensor<T, B>,
    /// Value cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub v_cache: Tensor<T, B>,
    /// Current end position (number of tokens seen so far).
    pub current_end: usize,
}

impl<T: WithDTypeF, B: Backend> StreamingMHAState<T, B> {
    pub fn device(&self) -> &B {
        self.k_cache.device()
    }

    #[allow(clippy::type_complexity)]
    pub fn complete_kv(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let t = k.dim(1usize)?;

        self.k_cache.slice_set(k, 1usize, self.current_end)?;
        self.v_cache.slice_set(v, 1usize, self.current_end)?;

        let new_end = self.current_end + t;
        let keys = self.k_cache.narrow(1, 0..new_end)?.contiguous()?;
        let values = self.v_cache.narrow(1, 0..new_end)?.contiguous()?;
        self.current_end = new_end;
        Ok((keys, values))
    }

    pub fn materialize_causal_mask(&self, num_queries: usize) -> Result<Tensor<T, B>> {
        let num_keys = self.current_end + num_queries;
        // Upper-left triangular mask (causal)
        let mut data = Vec::with_capacity(num_queries * num_keys);
        for q in 0..num_queries {
            for k in 0..num_keys {
                if k <= q + self.current_end {
                    data.push(T::from_f32(0.0));
                } else {
                    data.push(T::from_f32(f32::NEG_INFINITY));
                }
            }
        }
        Tensor::from_vec(data, (num_queries, num_keys), self.k_cache.device())
    }
}

/// Streaming multi-head attention (used by the flow LM transformer).
pub struct StreamingMultiheadAttention<Q: BackendQ> {
    in_proj: Q::LinearQ,
    out_proj: Q::LinearQ,
    pub embed_dim: usize,
    pub num_heads: usize,
    name: String,
    device: Q::B,
}

impl<Q: BackendQ> StreamingMultiheadAttention<Q> {
    pub fn load(vb: &Path<Q::B>, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = Q::linear_load(vb.pp("in_proj"), embed_dim, out_dim)?;
        let out_proj = Q::linear_load(vb.pp("out_proj"), embed_dim, embed_dim)?;
        let name = vb.prefix();
        let device = vb.device().clone();
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, name, device })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingMHAState<Q::T, Q::B>> {
        let dim_per_head = self.embed_dim / self.num_heads;
        let shape = (batch_size, sequence_length, self.num_heads, dim_per_head);
        let dev = &self.device;
        let k_cache = Tensor::zeros(shape, dev)?;
        let v_cache = Tensor::zeros(shape, dev)?;
        Ok(StreamingMHAState { k_cache, v_cache, current_end: 0 })
    }

    #[tracing::instrument(name = "attn", skip_all)]
    pub fn forward(
        &self,
        query: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        state: &mut StreamingMHAState<Q::T, Q::B>,
        mask: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
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
        let (q, k) = rope.forward(&q, &k)?;
        let (k, v) = state.complete_kv(&k, &v)?;

        // Transpose to [b, h, t, d] for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = Q::T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        // Back to [b, t, h*d]
        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?.contiguous()?;
        self.out_proj.forward(&x)
    }
}
// ---- KV Cache ----

/// Simple append-based KV cache with optional context window trimming.
#[derive(Clone, Debug)]
pub struct KvCache<T: WithDTypeF, B: Backend> {
    k: Option<Tensor<T, B>>,
    v: Option<Tensor<T, B>>,
    context: usize,
    absolute_offset: usize,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    pub fn new(context: usize) -> Self {
        Self { k: None, v: None, context, absolute_offset: 0 }
    }

    pub fn current_seq_len(&self) -> Result<usize> {
        let l = match &self.k {
            Some(k) => k.dim(2)?, // k shape: [b, h, seq, d]
            None => 0,
        };
        Ok(l)
    }

    /// Append new k, v (shape [b, h, t, d]) and return full (k, v).
    /// Trims to context if exceeded.
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

        let new_tokens = new_k.dim(2)?;
        self.absolute_offset += new_tokens;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn trim(&mut self) -> Result<()> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(()),
        };
        let seq_len = k.dim(2)?;
        if seq_len > self.context {
            let trim = seq_len - self.context;
            let k = k.narrow(2, trim..trim + self.context)?.contiguous()?;
            let v = v.narrow(2, trim..trim + self.context)?.contiguous()?;
            self.k = Some(k);
            self.v = Some(v);
        };
        Ok(())
    }
}

// ---- State types ----

#[derive(Clone, Debug)]
pub enum LayerAttentionState<T: WithDTypeF, B: Backend> {
    Mimi(KvCache<T, B>),
    FlowLm(StreamingMHAState<T, B>),
}

#[derive(Clone, Debug)]
pub struct StreamingTransformerState<T: WithDTypeF, B: Backend> {
    pub layer_states: Vec<LayerAttentionState<T, B>>,
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
        let in_proj = Linear::load(vb.pp("in_proj"), embed_dim, out_dim)?;
        let out_proj = Linear::load(vb.pp("out_proj"), embed_dim, embed_dim)?;
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
        mask: Option<&Tensor<T, B>>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;

        let projected = self.in_proj.forward(query)?;
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let q = packed.narrow(2, 0..1)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k = packed.narrow(2, 1..2)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = packed.narrow(2, 2..3)?.contiguous()?.reshape((b, t, self.num_heads, d))?;

        // RoPE on [b, t, h, d]
        let (q, k) = rope.forward(&q, &k)?;

        // To [b, h, t, d]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // KV cache with context trimming
        let (k, v) = state.append(&k, &v)?;

        // Attention with causal mask
        let scale = T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        state.trim()?;

        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

// ---- StreamingTransformerLayer ----

enum AttentionKind<Q: BackendQ> {
    Mimi(MimiStreamingMultiheadAttention<Q::T, Q::B>),
    FlowLm(StreamingMultiheadAttention<Q>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mimi,
    FlowLm,
}

pub struct StreamingTransformerLayer<Q: BackendQ> {
    self_attn: AttentionKind<Q>,
    norm1: LayerNorm<Q::T, Q::B>,
    norm2: LayerNorm<Q::T, Q::B>,
    linear1: Q::LinearQ,
    linear2: Q::LinearQ,
    layer_scale_1: Option<LayerScale<Q::T, Q::B>>,
    layer_scale_2: Option<LayerScale<Q::T, Q::B>>,
}

impl<Q: BackendQ> StreamingTransformerLayer<Q> {
    pub fn load(
        vb: &Path<Q::B>,
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

        let norm1 = LayerNorm::load(vb.pp("norm1"), d_model, 1e-5)?;
        let norm2 = LayerNorm::load(vb.pp("norm2"), d_model, 1e-5)?;
        let linear1 = Q::linear_load(vb.pp("linear1"), d_model, dim_feedforward)?;
        let linear2 = Q::linear_load(vb.pp("linear2"), dim_feedforward, d_model)?;

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
    ) -> Result<LayerAttentionState<Q::T, Q::B>> {
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
        x: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        state: &mut LayerAttentionState<Q::T, Q::B>,
        mask: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;

        // Self-attention block: x + layer_scale_1(attn(norm1(x)))
        let norm1 = self.norm1.forward(x)?;
        let mut attn_out = match (&self.self_attn, state) {
            (AttentionKind::Mimi(attn), LayerAttentionState::Mimi(cache)) => {
                attn.forward(&norm1, rope, cache, mask)?
            }
            (AttentionKind::FlowLm(attn), LayerAttentionState::FlowLm(mha_state)) => {
                attn.forward(&norm1, rope, mha_state, mask)?
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

pub struct StreamingTransformer<Q: BackendQ> {
    pub layers: Vec<StreamingTransformerLayer<Q>>,
    max_period: f32,
    head_dim: usize,
}

impl<Q: BackendQ> StreamingTransformer<Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<Q::B>,
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

        Ok(Self { layers, head_dim, max_period })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<Q::T, Q::B>> {
        let layer_states = self
            .layers
            .iter()
            .map(|l| l.init_state(batch_size, sequence_length))
            .collect::<Result<Vec<_>>>()?;
        Ok(StreamingTransformerState { layer_states })
    }

    pub fn forward(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut StreamingTransformerState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let mut x = x.clone();
        let (_, seq_len, _) = x.dims3()?;
        let mask = match state.layer_states.first() {
            Some(LayerAttentionState::Mimi(kv_cache)) => {
                let kv_seq_len = kv_cache.current_seq_len()?;
                let context = kv_cache.context;
                // Causal mask of shape (1, 1, seq_len, kv_seq_len + seq_len) with -inf in upper triangle
                let mask_data = (0..seq_len)
                    .flat_map(|seq_idx| {
                        let seq_idx = seq_idx + kv_seq_len;
                        (0..kv_seq_len + seq_len).map(move |attn_idx| {
                            if seq_idx.saturating_sub(context) <= attn_idx && attn_idx <= seq_idx {
                                Q::T::from_f32(0.0)
                            } else {
                                Q::T::from_f32(f32::NEG_INFINITY)
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                let mask =
                    Tensor::from_vec(mask_data, (1, 1, seq_len, kv_seq_len + seq_len), x.device())?;
                Some(mask)
            }
            Some(LayerAttentionState::FlowLm(kv_cache)) => {
                Some(kv_cache.materialize_causal_mask(seq_len)?)
            }
            _ => None,
        };
        let offset = state
            .layer_states
            .first()
            .map(|s| match s {
                LayerAttentionState::Mimi(kv_cache) => kv_cache.absolute_offset,
                LayerAttentionState::FlowLm(mha_state) => mha_state.current_end,
            })
            .unwrap_or(0);
        let rope =
            RotaryEmbedding::new(self.head_dim, offset, seq_len, self.max_period, x.device())?;
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &rope, layer_state, mask.as_ref())?;
        }
        Ok(x)
    }
}

// ---- ProjectedTransformer ----

pub struct ProjectedTransformer<Q: BackendQ> {
    pub transformer: StreamingTransformer<Q>,
    input_proj: Option<Linear<Q::T, Q::B>>,
    output_projs: Vec<Option<Linear<Q::T, Q::B>>>,
}

impl<Q: BackendQ> ProjectedTransformer<Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<Q::B>,
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
            Some(Linear::load(vb.pp("input_proj"), input_dimension, d_model)?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &out_dim) in output_dimensions.iter().enumerate() {
            if d_model == out_dim {
                output_projs.push(None);
            } else {
                let proj = Linear::load(vb.pp("output_proj").pp(i), d_model, out_dim)?;
                output_projs.push(Some(proj));
            }
        }

        Ok(Self { transformer, input_proj, output_projs })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<Q::T, Q::B>> {
        self.transformer.init_state(batch_size, sequence_length)
    }

    /// Forward pass. Input x is [B, C, T] (conv layout).
    #[tracing::instrument(name = "transformer", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut StreamingTransformerState<Q::T, Q::B>,
    ) -> Result<Vec<Tensor<Q::T, Q::B>>> {
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
