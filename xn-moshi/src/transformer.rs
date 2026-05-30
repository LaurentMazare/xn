use xn::BackendQ;
use xn::models::kv_cache::{IndicesAndMask, ScatteredCacheBuilder, ScatteredKvCache};
use xn::nn::{Linear, var_builder::Path};
use xn::streaming::{StreamMask, StreamTensor};
use xn::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f64>,
    pub positional_embedding: PositionalEmbedding,
    pub use_conv_block: bool,
    pub gating: Option<crate::seanet::Activation>,
    pub norm: crate::NormType,
    pub context: usize,
    pub max_period: f64,
    pub kv_repeat: usize,
    pub dim_feedforward: usize,
    pub conv_layout: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

// ============================================================================
// Streaming State Types
// ============================================================================

pub struct KvCacheState<Q: BackendQ> {
    pub k: Option<Tensor<Q::T, Q::B>>,
    pub v: Option<Tensor<Q::T, Q::B>>,
}

pub struct TransformerState<Q: BackendQ> {
    pub layers: Vec<KvCacheState<Q>>,
}

impl<Q: BackendQ> KvCacheState<Q> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn current_seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dims()[2], // [b, h, seq, d]
            None => 0,
        }
    }
}

impl<Q: BackendQ> Default for KvCacheState<Q> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Layer Scale
// ============================================================================

pub(crate) struct LayerScale<T: WithDTypeF, B: Backend> {
    scale: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> LayerScale<T, B> {
    pub(crate) fn load(vb: &Path<B>, d_model: usize) -> Result<Self> {
        let scale = vb.tensor("scale", (d_model,))?;
        Ok(Self { scale })
    }

    pub(crate) fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.broadcast_mul(&self.scale)
    }
}

// ============================================================================
// Normalization
// ============================================================================

pub(crate) enum Norm<T: WithDTypeF, B: Backend> {
    LayerNorm { weight: Tensor<T, B>, bias: Tensor<T, B>, eps: f32 },
    RmsNorm { alpha: Tensor<T, B>, eps: f32 },
}

impl<T: WithDTypeF, B: Backend> Norm<T, B> {
    pub(crate) fn load<V: std::borrow::Borrow<Path<B>>>(
        vb: V,
        d_model: usize,
        norm_type: crate::NormType,
    ) -> Result<Self> {
        let vb = vb.borrow();
        match norm_type {
            crate::NormType::LayerNorm => {
                let weight = if vb.contains("alpha") {
                    vb.tensor("alpha", (1, 1, d_model))?.reshape((d_model,))?
                } else {
                    vb.tensor("weight", (d_model,))?
                };
                let bias = vb.tensor("bias", (d_model,))?;
                Ok(Self::LayerNorm { weight, bias, eps: 1e-5 })
            }
            crate::NormType::RmsNorm => {
                let alpha = vb.tensor("alpha", (1, 1, d_model))?.reshape((d_model,))?;
                Ok(Self::RmsNorm { alpha, eps: 1e-8 })
            }
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Self::LayerNorm { weight, bias, eps } => xs.layer_norm(weight, bias, *eps),
            Self::RmsNorm { alpha, eps } => xs.rms_norm(alpha, *eps),
        }
    }
}

// ============================================================================
// MLP
// ============================================================================

pub(crate) enum Mlp<Q: BackendQ> {
    NoGating { linear1: Q::LinearQ, linear2: Q::LinearQ },
    Gating { linear_in: Q::LinearQ, linear_out: Q::LinearQ, activation: crate::seanet::Activation },
}

impl<Q: BackendQ> Mlp<Q> {
    pub(crate) fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                let linear1 =
                    Linear::load_o(vb.pp("linear1"), d_model, cfg.dim_feedforward, cfg.bias_ff)?;
                let linear1 = Q::from_linear(linear1)?;
                let linear2 =
                    Linear::load_o(vb.pp("linear2"), cfg.dim_feedforward, d_model, cfg.bias_ff)?;
                let linear2 = Q::from_linear(linear2)?;
                Ok(Self::NoGating { linear1, linear2 })
            }
            Some(activation) => {
                let hidden = if cfg.dim_feedforward == 4 * d_model {
                    11 * d_model / 4
                } else {
                    2 * cfg.dim_feedforward / 3
                };
                let vb = vb.pp("gating");
                let linear_in =
                    Linear::load_o(vb.pp("linear_in"), d_model, 2 * hidden, cfg.bias_ff)?;
                let linear_in = Q::from_linear(linear_in)?;
                let linear_out = Linear::load_o(vb.pp("linear_out"), hidden, d_model, cfg.bias_ff)?;
                let linear_out = Q::from_linear(linear_out)?;
                Ok(Self::Gating { linear_in, linear_out, activation })
            }
        }
    }

    #[tracing::instrument(name = "mlp-forward", skip_all)]
    pub(crate) fn forward(&self, xs: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        match self {
            Self::NoGating { linear1, linear2 } => {
                let xs = linear1.forward(xs)?.gelu_erf()?;
                let xs = linear2.forward(&xs)?;
                Ok(xs)
            }
            Self::Gating { linear_in, linear_out, activation } => {
                let (b, t, _) = xs.dims3()?;
                let xs = linear_in.forward(xs)?;
                let xs = xs.reshape((b, t, 2, ()))?;
                let x1 = xs.narrow(2, ..1)?.contiguous()?.reshape((b, t, ()))?;
                let x2 = xs.narrow(2, 1..2)?.contiguous()?.reshape((b, t, ()))?;
                let xs = activation.apply(&x1)?.mul(&x2)?;
                let xs = linear_out.forward(&xs)?;
                Ok(xs)
            }
        }
    }
}

// ============================================================================
// State Types
// ============================================================================

pub struct BatchedTransformerState<T: WithDTypeF, B: Backend> {
    pub builder: ScatteredCacheBuilder<B>,
    pub kv_caches: Vec<ScatteredKvCache<T, B>>,
}

impl<T: WithDTypeF, B: Backend> BatchedTransformerState<T, B> {
    pub fn batch_size(&self) -> usize {
        self.builder.batch_size()
    }

    pub fn reset(&mut self) {
        self.builder.reset();
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        if batch_idx >= self.batch_size() {
            xn::bail!("batch_idx {batch_idx} is out of bounds")
        }
        self.builder.reset_batch_index(batch_idx);
        Ok(())
    }
}

// ============================================================================
// Rotary Embeddings (per-batch positions)
// ============================================================================

pub(crate) struct RotaryEmbedding<B: Backend> {
    inv_freq: Tensor<f32, B>, // (1, 1, half_dim)
}

/// Precomputed cos/sin for a specific forward pass.
pub(crate) struct Rope<B: Backend> {
    cos: Tensor<f32, B>, // (batch, t, half_dim)
    sin: Tensor<f32, B>, // (batch, t, half_dim)
}

impl<B: Backend> RotaryEmbedding<B> {
    pub(crate) fn new(head_dim: usize, max_period: f32, device: &B) -> Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> =
            (0..half_dim).map(|i| 1.0 / max_period.powf(i as f32 / half_dim as f32)).collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half_dim), device)?;
        Ok(Self { inv_freq })
    }

    /// Compute per-batch rope from a positions tensor of shape (batch, t).
    pub(crate) fn rope(&self, pos: &Tensor<i64, B>) -> Result<Rope<B>> {
        // pos: (batch, t) -> unsqueeze to (batch, t, 1)
        let pos = pos.to::<f32>()?;
        let pos = pos.unsqueeze(2)?;
        // inv_freq: (1, 1, half_dim)
        // broadcast_mul: (batch, t, 1) * (1, 1, half_dim) -> (batch, t, half_dim)
        // (equivalent to matmul when inner dim is 1, but supports batch broadcasting)
        let freqs = pos.broadcast_mul(&self.inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Rope { cos, sin })
    }
}

impl<B: Backend> Rope<B> {
    pub(crate) fn apply_rotary_emb<T: WithDTypeF>(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.to::<f32>()?.rope_i(&self.cos, &self.sin, 0)?.to::<T>()
    }
}

// ============================================================================
// Multi-head Self-Attention (with ScatteredKvCache)
// ============================================================================

pub(crate) struct BatchedMultiheadAttention<Q: BackendQ> {
    in_proj: Q::LinearQ,
    out_proj: Q::LinearQ,
    num_heads: usize,
    head_dim: usize,
    context: usize,
}

impl<Q: BackendQ> BatchedMultiheadAttention<Q> {
    pub(crate) fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        let num_heads = cfg.num_heads;
        let head_dim = d_model / num_heads;
        let num_kv = num_heads / cfg.kv_repeat;
        let out_dim = d_model + 2 * num_kv * head_dim;

        let vb_attn = vb.pp("self_attn");
        let in_proj_weight = vb_attn.tensor("in_proj_weight", (out_dim, d_model))?;
        let in_proj = Linear::new(in_proj_weight);
        let in_proj = if cfg.bias_attn {
            let bias = vb_attn.tensor("in_proj_bias", (out_dim,))?;
            in_proj.with_bias(bias)
        } else {
            in_proj
        };
        let in_proj = Q::from_linear(in_proj)?;

        let out_proj = Linear::load_o(vb_attn.pp("out_proj"), d_model, d_model, cfg.bias_attn)?;
        let out_proj = Q::from_linear(out_proj)?;
        Ok(Self { in_proj, out_proj, num_heads, head_dim, context: cfg.context })
    }

    #[tracing::instrument(name = "batched-mha", skip_all)]
    pub(crate) fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: Option<&Rope<Q::B>>,
        kv_cache: &mut ScatteredKvCache<Q::T, Q::B>,
        iam: &IndicesAndMask<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;

        let dims = xs.dims();
        let (b, t) = (dims[0], dims[1]);
        let d_model = self.num_heads * self.head_dim;

        let qkv = self.in_proj.forward(xs)?;

        let q = qkv
            .narrow(2, ..d_model)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, d_model..2 * d_model)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, 2 * d_model..3 * d_model)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply rotary embeddings
        let (q, k) = if let Some(rope) = rope {
            (rope.apply_rotary_emb(&q)?, rope.apply_rotary_emb(&k)?)
        } else {
            (q, k)
        };

        // Append to scattered KV cache
        let (k, v) = kv_cache.append(&k, &v, iam)?;

        // Trim to context if needed
        let k_len = k.dims()[2];
        let k_target_len = t + usize::min(self.context, k_len - t);
        let (k, v) = if k_target_len < k_len {
            let k = k.narrow(2, k_len - k_target_len..k_len)?.contiguous()?;
            let v = v.narrow(2, k_len - k_target_len..k_len)?.contiguous()?;
            (k, v)
        } else {
            (k, v)
        };

        // Attention: q @ k^T * scale + mask -> softmax -> @ v
        let scale = Q::T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?; // (b, h, t, k)

        let mask = iam.mask(); // &Tensor<T, B>, shape (b, 1, t, context)
        let mask_context = mask.dim(3)?;
        // Trim mask to match k/v length if needed
        let mask_t = if k_target_len < mask_context {
            mask.narrow(3, mask_context - k_target_len..mask_context)?.contiguous()?
        } else {
            mask.clone()
        };

        let attn_weights = attn_weights.broadcast_add(&mask_t)?;
        let attn_weights = attn_weights.softmax()?; // (b, h, t, k)
        let attn_output = attn_weights.matmul(&v)?; // (b, h, t, d)

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, t, d_model))?;

        let out = self.out_proj.forward(&attn_output)?;
        Ok(out)
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

struct BatchedTransformerLayer<Q: BackendQ> {
    self_attn: BatchedMultiheadAttention<Q>,
    mlp: Mlp<Q>,
    norm1: Norm<Q::T, Q::B>,
    norm2: Norm<Q::T, Q::B>,
    layer_scale_1: Option<LayerScale<Q::T, Q::B>>,
    layer_scale_2: Option<LayerScale<Q::T, Q::B>>,
}

impl<Q: BackendQ> BatchedTransformerLayer<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        if cfg.use_conv_block {
            xn::bail!("conv-block is not supported")
        }
        let self_attn = BatchedMultiheadAttention::load(vb, cfg)?;
        let mlp = Mlp::load(vb, cfg)?;
        let norm1 = Norm::load(vb.pp("norm1"), cfg.d_model, cfg.norm)?;
        let norm2 = Norm::load(vb.pp("norm2"), cfg.d_model, cfg.norm)?;

        let layer_scale_1 = if cfg.layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_1"), cfg.d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if cfg.layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_2"), cfg.d_model)?)
        } else {
            None
        };

        Ok(Self { self_attn, mlp, norm1, norm2, layer_scale_1, layer_scale_2 })
    }

    fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: Option<&Rope<Q::B>>,
        kv_cache: &mut ScatteredKvCache<Q::T, Q::B>,
        iam: &IndicesAndMask<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        // norm_first path only
        let norm1_out = self.norm1.forward(xs)?;
        let mut attn_out = self.self_attn.forward(&norm1_out, rope, kv_cache, iam)?;
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let xs = xs.add(&attn_out)?;

        let norm2_out = self.norm2.forward(&xs)?;
        let mut mlp_out = self.mlp.forward(&norm2_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            mlp_out = ls.forward(&mlp_out)?;
        }
        xs.add(&mlp_out)
    }
}

// ============================================================================
// Batched Streaming Transformer
// ============================================================================

pub struct BatchedTransformer<Q: BackendQ> {
    layers: Vec<BatchedTransformerLayer<Q>>,
    rope: Option<RotaryEmbedding<Q::B>>,
    positional_embedding: PositionalEmbedding,
    num_kv: usize,
    head_dim: usize,
    context: usize,
    device: Q::B,
}

impl<Q: BackendQ> BatchedTransformer<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        if !cfg.causal {
            xn::bail!("only causal mode is supported")
        }
        if !cfg.norm_first {
            xn::bail!("only norm_first = true is supported")
        }
        if cfg.kv_repeat != 1 {
            xn::bail!("only kv_repeat = 1 is supported")
        }

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(BatchedTransformerLayer::load(&vb_layers.pp(i), cfg)?);
        }

        let rope = if cfg.positional_embedding == PositionalEmbedding::Rope {
            let head_dim = cfg.d_model / cfg.num_heads;
            Some(RotaryEmbedding::new(head_dim, cfg.max_period as f32, vb.device())?)
        } else {
            None
        };

        let num_kv = cfg.num_heads / cfg.kv_repeat;
        let head_dim = cfg.d_model / cfg.num_heads;

        Ok(Self {
            layers,
            rope,
            positional_embedding: cfg.positional_embedding,
            num_kv,
            head_dim,
            context: cfg.context,
            device: vb.device().clone(),
        })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<BatchedTransformerState<Q::T, Q::B>> {
        let builder = ScatteredCacheBuilder::new(batch_size, self.context, &self.device)?;
        let mut kv_caches = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            kv_caches.push(builder.make_cache(self.num_kv, self.head_dim)?);
        }
        Ok(BatchedTransformerState { builder, kv_caches })
    }

    pub fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        state: &mut BatchedTransformerState<Q::T, Q::B>,
        mask: &StreamMask,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let dims = xs.dims();
        let (b, t) = (dims[0], dims[1]);
        if b != state.batch_size() {
            xn::bail!("unexpected batch size {b} != {}", state.batch_size())
        }

        let batch_mask = match mask.cpu() {
            None => xn::bail!("batched-transformer expects a mask"),
            Some(m) => m,
        };

        // Save positions BEFORE indices_and_mask updates them (fixes off-by-t bug in reference).
        let pos: Vec<i64> = state
            .builder
            .positions()
            .iter()
            .flat_map(|v| (0..t).map(|i| (*v + i) as i64))
            .collect();

        let iam = state.builder.indices_and_mask(t, batch_mask)?;

        let rope = match &self.rope {
            Some(rope) => {
                let pos = Tensor::from_vec(pos, (b, t), xs.device())?;
                Some(rope.rope(&pos)?)
            }
            None => None,
        };

        let mut xs = match self.positional_embedding {
            PositionalEmbedding::Rope | PositionalEmbedding::None => xs.clone(),
            PositionalEmbedding::Sin => xn::bail!("sin positional embedding is not supported"),
        };

        for (layer, kv_cache) in self.layers.iter().zip(state.kv_caches.iter_mut()) {
            xs = layer.forward(&xs, rope.as_ref(), kv_cache, &iam)?;
        }
        Ok(xs)
    }
}

// ============================================================================
// Projected Batched Transformer (public)
// ============================================================================

pub struct BatchedProjectedTransformer<Q: BackendQ> {
    input_proj: Option<Q::LinearQ>,
    output_proj: Option<Q::LinearQ>,
    transformer: BatchedTransformer<Q>,
    conv_layout: bool,
}

impl<Q: BackendQ> BatchedProjectedTransformer<Q> {
    pub fn load(vb: &Path<Q::B>, input_dim: usize, cfg: &Config) -> Result<Self> {
        let input_proj = if input_dim != cfg.d_model {
            let linear = Linear::load(vb.pp("input_proj"), input_dim, cfg.d_model)?;
            Some(Q::from_linear(linear)?)
        } else {
            None
        };
        let output_proj = if input_dim != cfg.d_model {
            let linear = Linear::load(vb.pp("output_proj").pp(0), cfg.d_model, input_dim)?;
            Some(Q::from_linear(linear)?)
        } else {
            None
        };

        let transformer = BatchedTransformer::load(&vb.pp("transformer"), cfg)?;

        Ok(Self { input_proj, output_proj, transformer, conv_layout: cfg.conv_layout })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<BatchedTransformerState<Q::T, Q::B>> {
        self.transformer.init_state(batch_size)
    }

    pub fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        state: &mut BatchedTransformerState<Q::T, Q::B>,
        mask: &StreamMask,
    ) -> Result<Vec<Tensor<Q::T, Q::B>>> {
        use xn::ModuleT;

        let xs = if self.conv_layout { xs.transpose(1, 2)?.contiguous()? } else { xs.clone() };
        let xs = match &self.input_proj {
            Some(proj) => proj.forward(&xs)?,
            None => xs,
        };
        let xs = self.transformer.forward(&xs, state, mask)?;
        let ys = match &self.output_proj {
            Some(proj) => proj.forward(&xs)?,
            None => xs,
        };
        let ys = if self.conv_layout { ys.transpose(1, 2)?.contiguous()? } else { ys };
        Ok(vec![ys])
    }

    pub fn step(
        &self,
        xs: &StreamTensor<Q::T, Q::B>,
        state: &mut BatchedTransformerState<Q::T, Q::B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<Q::T, Q::B>> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs,
        };
        let ys = self.forward(xs, state, mask)?;
        Ok(StreamTensor::from_tensor(ys[0].clone()))
    }
}
