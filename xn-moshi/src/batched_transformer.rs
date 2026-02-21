use crate::streaming::{StreamMask, StreamTensor};
use crate::transformer::{Config, LayerScale, Mlp, Norm, PositionalEmbedding};
use xn::models::kv_cache::{IndicesAndMask, ScatteredCacheBuilder, ScatteredKvCache};
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

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

struct RotaryEmbedding<T: WithDTypeF, B: Backend> {
    inv_freq: Tensor<T, B>, // (1, 1, half_dim)
}

/// Precomputed cos/sin for a specific forward pass.
struct Rope<T: WithDTypeF, B: Backend> {
    cos: Tensor<T, B>, // (batch, t, half_dim)
    sin: Tensor<T, B>, // (batch, t, half_dim)
}

impl<T: WithDTypeF, B: Backend> RotaryEmbedding<T, B> {
    fn new(head_dim: usize, max_period: f32, device: &B) -> Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<T> = (0..half_dim)
            .map(|i| T::from_f32(1.0 / max_period.powf(i as f32 / half_dim as f32)))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half_dim), device)?;
        Ok(Self { inv_freq })
    }

    /// Compute per-batch rope from a positions tensor of shape (batch, t).
    fn rope(&self, pos: &Tensor<T, B>) -> Result<Rope<T, B>> {
        // pos: (batch, t) -> unsqueeze to (batch, t, 1)
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

impl<T: WithDTypeF, B: Backend> Rope<T, B> {
    /// Apply rotary embeddings to x of shape (b, h, t, d) using interleaved pairs.
    fn apply_rotary_emb(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let dims = x.dims();
        let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
        let half_d = d / 2;

        // Reshape to (b, h, t, half_d, 2)
        let x = x.clone().reshape((b, h, t, half_d, 2))?;
        let x0 = x.narrow(4, ..1)?.contiguous()?.reshape((b, h, t, half_d))?;
        let x1 = x
            .narrow(4, 1..2)?
            .contiguous()?
            .reshape((b, h, t, half_d))?;

        // cos, sin: (b, t, half_d) -> unsqueeze(1) -> (b, 1, t, half_d)
        let cos = self.cos.unsqueeze(1)?;
        let sin = self.sin.unsqueeze(1)?;

        // y0 = x0 * cos - x1 * sin
        // y1 = x0 * sin + x1 * cos
        let y0 = x0.broadcast_mul(&cos)?.sub(&x1.broadcast_mul(&sin)?)?;
        let y1 = x0.broadcast_mul(&sin)?.add(&x1.broadcast_mul(&cos)?)?;

        // Stack on dim 4 to get (b, h, t, half_d, 2), then reshape to (b, h, t, d)
        let rope = Tensor::stack(&[&y0, &y1], 4)?;
        rope.reshape((b, h, t, d))
    }
}

// ============================================================================
// Multi-head Self-Attention (with ScatteredKvCache)
// ============================================================================

struct BatchedMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj_weight: Tensor<T, B>,
    in_proj_bias: Option<Tensor<T, B>>,
    out_proj: Linear<T, B>,
    num_heads: usize,
    head_dim: usize,
    context: usize,
}

impl<T: WithDTypeF, B: Backend> BatchedMultiheadAttention<T, B> {
    fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        let num_heads = cfg.num_heads;
        let head_dim = d_model / num_heads;
        let num_kv = num_heads / cfg.kv_repeat;
        let out_dim = d_model + 2 * num_kv * head_dim;

        let vb_attn = vb.pp("self_attn");
        let in_proj_weight = vb_attn.tensor("in_proj_weight", (out_dim, d_model))?;
        let in_proj_bias = if cfg.bias_attn {
            Some(vb_attn.tensor("in_proj_bias", (out_dim,))?)
        } else {
            None
        };

        let out_proj = Linear::load_o(&vb_attn.pp("out_proj"), d_model, d_model, cfg.bias_attn)?;
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim,
            context: cfg.context,
        })
    }

    #[tracing::instrument(name = "batched-mha", skip_all)]
    fn forward(
        &self,
        xs: &Tensor<T, B>,
        rope: Option<&Rope<T, B>>,
        kv_cache: &mut ScatteredKvCache<T, B>,
        iam: &IndicesAndMask<T, B>,
    ) -> Result<Tensor<T, B>> {
        let dims = xs.dims();
        let (b, t) = (dims[0], dims[1]);
        let d_model = self.num_heads * self.head_dim;

        let mut qkv = xs.matmul_t(&self.in_proj_weight)?;
        if let Some(bias) = &self.in_proj_bias {
            qkv = qkv.broadcast_add(bias)?;
        }

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
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?; // (b, h, t, k)

        let mask = iam.mask(); // &Tensor<T, B>, shape (b, 1, t, context)
        let mask_dims = mask.dims();
        // Trim mask to match k/v length if needed
        let mask_context = mask_dims[3];
        let mask_t = if k_target_len < mask_context {
            mask.narrow(3, mask_context - k_target_len..mask_context)?
                .contiguous()?
        } else {
            mask.clone()
        };

        let attn_weights = attn_weights.broadcast_add(&mask_t)?;
        let attn_weights = attn_weights.softmax()?; // (b, h, t, k)
        let attn_output = attn_weights.matmul(&v)?; // (b, h, t, d)

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, d_model))?;

        let out = self.out_proj.forward(&attn_output)?;
        Ok(out)
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

struct BatchedTransformerLayer<T: WithDTypeF, B: Backend> {
    self_attn: BatchedMultiheadAttention<T, B>,
    mlp: Mlp<T, B>,
    norm1: Norm<T, B>,
    norm2: Norm<T, B>,
    layer_scale_1: Option<LayerScale<T, B>>,
    layer_scale_2: Option<LayerScale<T, B>>,
}

impl<T: WithDTypeF, B: Backend> BatchedTransformerLayer<T, B> {
    fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        if cfg.use_conv_block {
            xn::bail!("conv-block is not supported")
        }
        let self_attn = BatchedMultiheadAttention::load(vb, cfg)?;
        let mlp = Mlp::load(vb, cfg)?;
        let norm1 = Norm::load(&vb.pp("norm1"), cfg.d_model, cfg.norm)?;
        let norm2 = Norm::load(&vb.pp("norm2"), cfg.d_model, cfg.norm)?;

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

        Ok(Self {
            self_attn,
            mlp,
            norm1,
            norm2,
            layer_scale_1,
            layer_scale_2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<T, B>,
        rope: Option<&Rope<T, B>>,
        kv_cache: &mut ScatteredKvCache<T, B>,
        iam: &IndicesAndMask<T, B>,
    ) -> Result<Tensor<T, B>> {
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

pub struct BatchedTransformer<T: WithDTypeF, B: Backend> {
    layers: Vec<BatchedTransformerLayer<T, B>>,
    rope: Option<RotaryEmbedding<T, B>>,
    positional_embedding: PositionalEmbedding,
    num_kv: usize,
    head_dim: usize,
    context: usize,
    device: B,
}

impl<T: WithDTypeF, B: Backend> BatchedTransformer<T, B> {
    pub fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
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
            Some(RotaryEmbedding::new(
                head_dim,
                cfg.max_period as f32,
                vb.device(),
            )?)
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

    pub fn init_state(&self, batch_size: usize) -> Result<BatchedTransformerState<T, B>> {
        let builder = ScatteredCacheBuilder::new(batch_size, self.context, &self.device)?;
        let mut kv_caches = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            kv_caches.push(builder.make_cache(self.num_kv, self.head_dim)?);
        }
        Ok(BatchedTransformerState { builder, kv_caches })
    }

    pub fn forward(
        &self,
        xs: &Tensor<T, B>,
        state: &mut BatchedTransformerState<T, B>,
        mask: &StreamMask,
    ) -> Result<Tensor<T, B>> {
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
        let positions: Vec<Vec<T>> = state
            .builder
            .positions()
            .iter()
            .map(|&v| (0..t).map(|i| T::from_f32((v + i) as f32)).collect())
            .collect();

        let iam = state.builder.indices_and_mask(t, batch_mask)?;

        let rope = match &self.rope {
            Some(rope) => {
                let pos_flat: Vec<T> = positions.into_iter().flatten().collect();
                let pos = Tensor::from_vec(pos_flat, (b, t), xs.device())?;
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

pub struct BatchedProjectedTransformer<T: WithDTypeF, B: Backend> {
    input_proj: Option<Linear<T, B>>,
    output_proj: Option<Linear<T, B>>,
    transformer: BatchedTransformer<T, B>,
    conv_layout: bool,
}

impl<T: WithDTypeF, B: Backend> BatchedProjectedTransformer<T, B> {
    pub fn load(vb: &Path<B>, input_dim: usize, cfg: &Config) -> Result<Self> {
        let input_proj = if input_dim != cfg.d_model {
            Some(Linear::load(&vb.pp("input_proj"), input_dim, cfg.d_model)?)
        } else {
            None
        };
        let output_proj = if input_dim != cfg.d_model {
            Some(Linear::load(
                &vb.pp("output_proj").pp(0),
                cfg.d_model,
                input_dim,
            )?)
        } else {
            None
        };

        let transformer = BatchedTransformer::load(&vb.pp("transformer"), cfg)?;

        Ok(Self {
            input_proj,
            output_proj,
            transformer,
            conv_layout: cfg.conv_layout,
        })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<BatchedTransformerState<T, B>> {
        self.transformer.init_state(batch_size)
    }

    pub fn forward(
        &self,
        xs: &Tensor<T, B>,
        state: &mut BatchedTransformerState<T, B>,
        mask: &StreamMask,
    ) -> Result<Vec<Tensor<T, B>>> {
        let xs = if self.conv_layout {
            xs.transpose(1, 2)?.contiguous()?
        } else {
            xs.clone()
        };
        let xs = match &self.input_proj {
            Some(proj) => proj.forward(&xs)?,
            None => xs,
        };
        let xs = self.transformer.forward(&xs, state, mask)?;
        let ys = match &self.output_proj {
            Some(proj) => proj.forward(&xs)?,
            None => xs,
        };
        let ys = if self.conv_layout {
            ys.transpose(1, 2)?.contiguous()?
        } else {
            ys
        };
        Ok(vec![ys])
    }

    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut BatchedTransformerState<T, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs,
        };
        let ys = self.forward(xs, state, mask)?;
        Ok(StreamTensor::from_tensor(ys[0].clone()))
    }
}
