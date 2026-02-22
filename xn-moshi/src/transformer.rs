use crate::streaming::StreamTensor;
use xn::nn::{var_builder::Path, Linear};
use xn::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone)]
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
    pub conv_kernel_size: usize,
    pub use_conv_bias: bool,
    pub gating: Option<crate::seanet::Activation>,
    pub norm: crate::NormType,
    pub context: usize,
    pub max_period: usize,
    pub max_seq_len: usize,
    pub kv_repeat: usize,
    pub dim_feedforward: usize,
    pub conv_layout: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

// ============================================================================
// Streaming State Types
// ============================================================================

pub struct KvCacheState<T: WithDTypeF, B: Backend> {
    pub k: Option<Tensor<T, B>>,
    pub v: Option<Tensor<T, B>>,
}

pub struct TransformerState<T: WithDTypeF, B: Backend> {
    pub layers: Vec<KvCacheState<T, B>>,
}

impl<T: WithDTypeF, B: Backend> KvCacheState<T, B> {
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

impl<T: WithDTypeF, B: Backend> Default for KvCacheState<T, B> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rotary Embeddings
// ============================================================================

struct RotaryEmbedding<T: WithDTypeF, B: Backend> {
    cos: Tensor<T, B>,
    sin: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> RotaryEmbedding<T, B> {
    // TODO: This precomputes cos/sin for all positions up to max_seq_len on the CPU.
    // The reference implementation computes them on-the-fly per forward call using
    // Tensor::arange() + matmul with inv_freq, which avoids the upfront allocation and
    // works for arbitrary sequence lengths. Would require xn Tensor::arange() support.
    fn new(head_dim: usize, max_seq_len: usize, theta: f32, device: &B) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0f32 / theta.powf(i as f32 / half_dim as f32));
        }

        let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_data.push(T::from_f32(angle.cos()));
                sin_data.push(T::from_f32(angle.sin()));
            }
        }

        let cos = Tensor::from_vec(cos_data, (max_seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (max_seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
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
    LayerNorm {
        weight: Tensor<T, B>,
        bias: Tensor<T, B>,
        eps: f32,
    },
    RmsNorm {
        alpha: Tensor<T, B>,
        eps: f32,
    },
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
                Ok(Self::LayerNorm {
                    weight,
                    bias,
                    eps: 1e-5,
                })
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

pub(crate) enum Mlp<T: WithDTypeF, B: Backend> {
    NoGating {
        linear1: Linear<T, B>,
        linear2: Linear<T, B>,
    },
    Gating {
        linear_in: Linear<T, B>,
        linear_out: Linear<T, B>,
        activation: crate::seanet::Activation,
    },
}

impl<T: WithDTypeF, B: Backend> Mlp<T, B> {
    pub(crate) fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                let linear1 =
                    Linear::load_o(vb.pp("linear1"), d_model, cfg.dim_feedforward, cfg.bias_ff)?;
                let linear2 =
                    Linear::load_o(vb.pp("linear2"), cfg.dim_feedforward, d_model, cfg.bias_ff)?;
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
                let linear_out = Linear::load_o(vb.pp("linear_out"), hidden, d_model, cfg.bias_ff)?;
                Ok(Self::Gating {
                    linear_in,
                    linear_out,
                    activation,
                })
            }
        }
    }

    #[tracing::instrument(name = "mlp-forward", skip_all)]
    pub(crate) fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Self::NoGating { linear1, linear2 } => {
                let xs = linear1.forward(xs)?.gelu_erf()?;
                let xs = linear2.forward(&xs)?;
                Ok(xs)
            }
            Self::Gating {
                linear_in,
                linear_out,
                activation,
            } => {
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
// Multi-head Self-Attention
// ============================================================================

struct StreamingMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj_weight: Tensor<T, B>,
    in_proj_bias: Option<Tensor<T, B>>,
    out_proj: Linear<T, B>,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl<T: WithDTypeF, B: Backend> StreamingMultiheadAttention<T, B> {
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

        let out_proj = Linear::load_o(vb_attn.pp("out_proj"), d_model, d_model, cfg.bias_attn)?;
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim,
            max_seq_len: cfg.context,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<T, B>,
        rope: Option<&RotaryEmbedding<f32, B>>,
        offset: usize,
        kv_cache: &mut KvCacheState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _hd) = xs.dims3()?;

        let mut qkv = xs.matmul_t(&self.in_proj_weight)?;
        if let Some(bias) = &self.in_proj_bias {
            qkv = qkv.broadcast_add(bias)?;
        }

        let d_model = self.num_heads * self.head_dim;
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
            let q_shape = q.shape();
            let q_data = q.to_vec()?;
            let q_data = q_data.iter().map(|&x| x.to_f32()).collect();
            let q = Tensor::from_vec(q_data, q.shape(), q.device())?;
            let k_shape = k.shape();
            let k_data = k.to_vec()?;
            let k_data = k_data.iter().map(|&x| x.to_f32()).collect();
            let k = Tensor::from_vec(k_data, k.shape(), k.device())?;
            let q = q.rope_i(&rope.cos, &rope.sin, offset)?;
            let k = k.rope_i(&rope.cos, &rope.sin, offset)?;
            let q_data = q.to_vec()?;
            let q_data = q_data.iter().map(|&x| T::from_f32(x)).collect();
            let q = Tensor::from_vec(q_data, q_shape, q.device())?;
            let k_data = k.to_vec()?;
            let k_data = k_data.iter().map(|&x| T::from_f32(x)).collect();
            let k = Tensor::from_vec(k_data, k_shape, k.device())?;
            (q, k)
        } else {
            (q, k)
        };

        // KV cache append
        let (k, v) = self.kv_cache_append(kv_cache, &k, &v)?;

        // Attention
        // TODO: This uses apply_causality_mask which is a simple causal mask based on a global
        // offset. The reference implementation builds per-batch streaming masks using
        // last_reset_pos, expand(), arange(), and where_cond() to support per-batch streaming
        // resets (different sequences in a batch can reset at different positions). This would
        // require xn expand(), Tensor::arange(), and where_cond() support.
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?;
        let attn_weights = attn_weights.apply_causality_mask(offset)?;
        let attn_weights = attn_weights.softmax()?;

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, t, self.num_heads * self.head_dim))?;

        let out = self.out_proj.forward(&attn_output)?;
        Ok(out)
    }

    #[tracing::instrument(name = "kv-append", skip_all)]
    fn kv_cache_append(
        &self,
        cache: &mut KvCacheState<T, B>,
        new_k: &Tensor<T, B>,
        new_v: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let (k, v) = match (&cache.k, &cache.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        let seq_len = k.dims()[2];
        let (k, v) = if seq_len > self.max_seq_len {
            let trim = seq_len - self.max_seq_len;
            (
                k.narrow(2, trim..trim + self.max_seq_len)?.contiguous()?,
                v.narrow(2, trim..trim + self.max_seq_len)?.contiguous()?,
            )
        } else {
            (k, v)
        };

        cache.k = Some(k.clone());
        cache.v = Some(v.clone());
        Ok((k, v))
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

struct StreamingTransformerLayer<T: WithDTypeF, B: Backend> {
    self_attn: StreamingMultiheadAttention<T, B>,
    mlp: Mlp<T, B>,
    norm1: Norm<T, B>,
    norm2: Norm<T, B>,
    layer_scale_1: Option<LayerScale<T, B>>,
    layer_scale_2: Option<LayerScale<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformerLayer<T, B> {
    fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        if cfg.use_conv_block {
            xn::bail!("conv-block is not supported")
        }
        let self_attn = StreamingMultiheadAttention::load(vb, cfg)?;
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
        rope: Option<&RotaryEmbedding<f32, B>>,
        offset: usize,
        kv_cache: &mut KvCacheState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let norm1_out = self.norm1.forward(xs)?;
        let mut attn_out = self.self_attn.forward(&norm1_out, rope, offset, kv_cache)?;
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
// Streaming Transformer
// ============================================================================

struct StreamingTransformer<T: WithDTypeF, B: Backend> {
    layers: Vec<StreamingTransformerLayer<T, B>>,
    rope: Option<RotaryEmbedding<f32, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformer<T, B> {
    fn load(vb: &Path<B>, cfg: &Config, device: &B) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(StreamingTransformerLayer::load(&vb_layers.pp(i), cfg)?);
        }

        let rope = if cfg.positional_embedding == PositionalEmbedding::Rope {
            let head_dim = cfg.d_model / cfg.num_heads;
            Some(RotaryEmbedding::new(
                head_dim,
                cfg.max_seq_len,
                cfg.max_period as f32,
                device,
            )?)
        } else {
            None
        };

        Ok(Self { layers, rope })
    }

    fn init_state(&self) -> TransformerState<T, B> {
        TransformerState {
            layers: self.layers.iter().map(|_| KvCacheState::new()).collect(),
        }
    }

    fn forward(
        &self,
        xs: &Tensor<T, B>,
        state: &mut TransformerState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let offset = state.layers[0].current_seq_len();
        let mut xs = xs.clone();
        for (layer, kv_cache) in self.layers.iter().zip(state.layers.iter_mut()) {
            xs = layer.forward(&xs, self.rope.as_ref(), offset, kv_cache)?;
        }
        Ok(xs)
    }
}

// ============================================================================
// Projected Transformer (public)
// ============================================================================

pub struct ProjectedTransformer<T: WithDTypeF, B: Backend> {
    input_proj: Option<Linear<T, B>>,
    output_proj: Option<Linear<T, B>>,
    transformer: StreamingTransformer<T, B>,
    conv_layout: bool,
}

impl<T: WithDTypeF, B: Backend> ProjectedTransformer<T, B> {
    pub fn load(vb: &Path<B>, input_dim: usize, cfg: &Config, device: &B) -> Result<Self> {
        let input_proj = if input_dim != cfg.d_model {
            Some(Linear::load(vb.pp("input_proj"), input_dim, cfg.d_model)?)
        } else {
            None
        };

        let output_proj = if input_dim != cfg.d_model {
            Some(Linear::load(
                vb.pp("output_proj").pp(0),
                cfg.d_model,
                input_dim,
            )?)
        } else {
            None
        };

        let transformer = StreamingTransformer::load(&vb.pp("transformer"), cfg, device)?;

        Ok(Self {
            input_proj,
            output_proj,
            transformer,
            conv_layout: cfg.conv_layout,
        })
    }

    pub fn init_state(&self) -> TransformerState<T, B> {
        self.transformer.init_state()
    }

    pub fn forward(
        &self,
        xs: &Tensor<T, B>,
        state: &mut TransformerState<T, B>,
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
        let xs = self.transformer.forward(&xs, state)?;
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

    #[tracing::instrument(name = "transformer", skip_all)]
    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut TransformerState<T, B>,
    ) -> Result<StreamTensor<T, B>> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                let results = self.forward(xs, state)?;
                Ok(StreamTensor::from_tensor(
                    results.into_iter().next().unwrap(),
                ))
            }
        }
    }
}
