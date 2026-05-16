use crate::transformer::{
    BatchedMultiheadAttention, BatchedTransformerState, Config, LayerScale, Mlp, Norm,
    PositionalEmbedding, Rope, RotaryEmbedding,
};
use xn::BackendQ;
use xn::nn::{Linear, var_builder::Path};
use xn::streaming::StreamMask;
use xn::{Result, Tensor, WithDTypeF};

// ============================================================================
// Cross-Attention Source
// ============================================================================

/// Input to cross-attention. Either raw tokens that still need a KV projection,
/// or pre-computed (K, V) tensors that can be reused across timesteps/layers.
pub enum CaSrc<Q: BackendQ> {
    Tokens(Tensor<Q::T, Q::B>),
    KeysValues(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>),
}

// ============================================================================
// Multi-head Cross-Attention
// ============================================================================

struct BatchedMultiheadCrossAttention<Q: BackendQ> {
    in_proj_q: Q::LinearQ,
    in_proj_kv: Q::LinearQ,
    out_proj: Q::LinearQ,
    num_heads: usize,
    head_dim: usize,
}

impl<Q: BackendQ> BatchedMultiheadCrossAttention<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        let num_heads = cfg.num_heads;
        let head_dim = d_model / num_heads;
        let num_kv = num_heads / cfg.kv_repeat;
        let out_kv_dim = num_kv * head_dim;
        // KV input dim defaults to d_model.
        let kv_in_dim = d_model;
        let out_dim = d_model + 2 * out_kv_dim;

        // Two weight layouts are supported:
        //  - combined: a single `in_proj_weight` of shape (d_model + 2*kv_dim, d_model),
        //    sliced along dim 0 into Q and KV (requires kv_in_dim == d_model).
        //  - split:    separate `in_proj_weight_q` and `in_proj_weight_kv` tensors.
        let (in_proj_q, in_proj_kv) = if vb.contains("in_proj_weight") {
            let combined = vb.tensor("in_proj_weight", (out_dim, d_model))?;
            let w_q = combined.narrow(0, ..d_model)?.contiguous()?;
            let w_kv = combined.narrow(0, d_model..out_dim)?.contiguous()?;
            let (b_q, b_kv) = if cfg.bias_attn {
                let bias = vb.tensor("in_proj_bias", (out_dim,))?;
                let b_q = bias.narrow(0, ..d_model)?.contiguous()?;
                let b_kv = bias.narrow(0, d_model..out_dim)?.contiguous()?;
                (Some(b_q), Some(b_kv))
            } else {
                (None, None)
            };
            let q = Linear::new(w_q);
            let q = match b_q {
                Some(b) => q.with_bias(b),
                None => q,
            };
            let kv = Linear::new(w_kv);
            let kv = match b_kv {
                Some(b) => kv.with_bias(b),
                None => kv,
            };
            (Q::from_linear(q)?, Q::from_linear(kv)?)
        } else {
            let w_q = vb.tensor("in_proj_weight_q", (d_model, d_model))?;
            let q = Linear::new(w_q);
            let q = if cfg.bias_attn {
                q.with_bias(vb.tensor("in_proj_bias_q", (d_model,))?)
            } else {
                q
            };
            let w_kv = vb.tensor("in_proj_weight_kv", (2 * out_kv_dim, kv_in_dim))?;
            let kv = Linear::new(w_kv);
            let kv = if cfg.bias_attn {
                kv.with_bias(vb.tensor("in_proj_bias_kv", (2 * out_kv_dim,))?)
            } else {
                kv
            };
            (Q::from_linear(q)?, Q::from_linear(kv)?)
        };

        let out_proj = Linear::load_o(vb.pp("out_proj"), d_model, d_model, cfg.bias_attn)?;
        let out_proj = Q::from_linear(out_proj)?;

        Ok(Self { in_proj_q, in_proj_kv, out_proj, num_heads, head_dim })
    }

    /// Project the cross-attention source into (K, V). When the source is
    /// already pre-projected, this is a no-op.
    #[allow(clippy::type_complexity)]
    fn compute_kv(&self, ca_src: &CaSrc<Q>) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        use xn::ModuleT;
        match ca_src {
            CaSrc::KeysValues(k, v) => Ok((k.clone(), v.clone())),
            CaSrc::Tokens(xs) => {
                let kv = self.in_proj_kv.forward(xs)?;
                let (cab, cat, _) = kv.dims3()?;
                let kv_dim = self.num_heads * self.head_dim;
                let k = kv
                    .narrow(2, ..kv_dim)?
                    .reshape((cab, cat, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = kv
                    .narrow(2, kv_dim..2 * kv_dim)?
                    .reshape((cab, cat, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                Ok((k, v))
            }
        }
    }

    #[tracing::instrument(name = "batched-mhca", skip_all)]
    fn forward(&self, xs: &Tensor<Q::T, Q::B>, ca_src: &CaSrc<Q>) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;

        let (b, t, _) = xs.dims3()?;
        let d_model = self.num_heads * self.head_dim;

        let q = self
            .in_proj_q
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // (b, h, t, d)

        let (k, v) = self.compute_kv(ca_src)?; // (b, h, k, d)

        let scale = Q::T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?; // (b, h, t, k)
        let attn_weights = attn_weights.softmax()?;
        let attn_output = attn_weights.matmul(&v)?; // (b, h, t, d)

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, t, d_model))?;
        let out = self.out_proj.forward(&attn_output)?;
        Ok(out)
    }
}

// ============================================================================
// Transformer Layer (with cross-attention)
// ============================================================================

struct Layer<Q: BackendQ> {
    self_attn: BatchedMultiheadAttention<Q>,
    cross_attn: BatchedMultiheadCrossAttention<Q>,
    norm_cross: Norm<Q::T, Q::B>,
    mlp: Mlp<Q>,
    norm1: Norm<Q::T, Q::B>,
    norm2: Norm<Q::T, Q::B>,
    layer_scale_1: Option<LayerScale<Q::T, Q::B>>,
    layer_scale_2: Option<LayerScale<Q::T, Q::B>>,
}

impl<Q: BackendQ> Layer<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        if cfg.use_conv_block {
            xn::bail!("conv-block is not supported")
        }
        let self_attn = BatchedMultiheadAttention::load(vb, cfg)?;
        let mlp = Mlp::load(vb, cfg)?;
        let norm1 = Norm::load(vb.pp("norm1"), cfg.d_model, cfg.norm)?;
        let norm2 = Norm::load(vb.pp("norm2"), cfg.d_model, cfg.norm)?;
        let norm_cross = Norm::load(vb.pp("norm_cross"), cfg.d_model, crate::NormType::LayerNorm)?;
        let cross_attn = BatchedMultiheadCrossAttention::load(&vb.pp("cross_attention"), cfg)?;

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
            cross_attn,
            norm_cross,
            mlp,
            norm1,
            norm2,
            layer_scale_1,
            layer_scale_2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: Option<&Rope<Q::B>>,
        kv_cache: &mut xn::models::kv_cache::ScatteredKvCache<Q::T, Q::B>,
        iam: &xn::models::kv_cache::IndicesAndMask<Q::T, Q::B>,
        ca_src: &CaSrc<Q>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        // Self-attention.
        let norm1_out = self.norm1.forward(xs)?;
        let mut attn_out = self.self_attn.forward(&norm1_out, rope, kv_cache, iam)?;
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let xs = xs.add(&attn_out)?;

        // Cross-attention (always present, no gating beyond identity).
        let norm_cross_out = self.norm_cross.forward(&xs)?;
        let ca_out = self.cross_attn.forward(&norm_cross_out, ca_src)?;
        let xs = xs.add(&ca_out)?;

        // Feed-forward.
        let norm2_out = self.norm2.forward(&xs)?;
        let mut mlp_out = self.mlp.forward(&norm2_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            mlp_out = ls.forward(&mlp_out)?;
        }
        xs.add(&mlp_out)
    }
}

// ============================================================================
// Batched Streaming Transformer (with cross-attention)
// ============================================================================

pub struct Transformer<Q: BackendQ> {
    layers: Vec<Layer<Q>>,
    rope: Option<RotaryEmbedding<Q::B>>,
    positional_embedding: PositionalEmbedding,
    num_kv: usize,
    head_dim: usize,
    context: usize,
    device: Q::B,
}

impl<Q: BackendQ> Transformer<Q> {
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
            layers.push(Layer::load(&vb_layers.pp(i), cfg)?);
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
        let builder = xn::models::kv_cache::ScatteredCacheBuilder::new(
            batch_size,
            self.context,
            &self.device,
        )?;
        let mut kv_caches = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            kv_caches.push(builder.make_cache(self.num_kv, self.head_dim)?);
        }
        Ok(BatchedTransformerState { builder, kv_caches })
    }

    pub fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        ca_src: &CaSrc<Q>,
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
            xs = layer.forward(&xs, rope.as_ref(), kv_cache, &iam, ca_src)?;
        }
        Ok(xs)
    }
}
