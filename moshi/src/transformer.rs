use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use xn::nn::var_builder::Path;
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
// KV Cache
// ============================================================================

struct KvCache<T: WithDTypeF, B: Backend> {
    k: Option<Tensor<T, B>>,
    v: Option<Tensor<T, B>>,
    max_seq_len: usize,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
        }
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    fn current_seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dims()[2], // [b, h, seq, d]
            None => 0,
        }
    }

    #[tracing::instrument(name = "kv-append", skip_all)]
    fn append(
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

        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
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

struct LayerScale<T: WithDTypeF, B: Backend> {
    scale: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> LayerScale<T, B> {
    fn load(vb: &Path<B>, d_model: usize) -> Result<Self> {
        let scale = vb.tensor("scale", (d_model,))?;
        Ok(Self { scale })
    }

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.broadcast_mul(&self.scale)
    }
}

// ============================================================================
// Normalization
// ============================================================================

enum Norm<T: WithDTypeF, B: Backend> {
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
    fn load(vb: &Path<B>, d_model: usize, norm_type: crate::NormType) -> Result<Self> {
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

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Self::LayerNorm { weight, bias, eps } => xs.layer_norm(weight, bias, *eps),
            Self::RmsNorm { alpha, eps } => xs.rms_norm(alpha, *eps),
        }
    }
}

// ============================================================================
// MLP
// ============================================================================

enum Mlp<T: WithDTypeF, B: Backend> {
    NoGating {
        linear1_weight: Tensor<T, B>,
        linear1_bias: Option<Tensor<T, B>>,
        linear2_weight: Tensor<T, B>,
        linear2_bias: Option<Tensor<T, B>>,
    },
    Gating {
        linear_in_weight: Tensor<T, B>,
        linear_in_bias: Option<Tensor<T, B>>,
        linear_out_weight: Tensor<T, B>,
        linear_out_bias: Option<Tensor<T, B>>,
        activation: crate::seanet::Activation,
    },
}

impl<T: WithDTypeF, B: Backend> Mlp<T, B> {
    fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                let linear1_weight = vb
                    .pp("linear1")
                    .tensor("weight", (cfg.dim_feedforward, d_model))?;
                let linear1_bias = if cfg.bias_ff {
                    Some(vb.pp("linear1").tensor("bias", (cfg.dim_feedforward,))?)
                } else {
                    None
                };
                let linear2_weight = vb
                    .pp("linear2")
                    .tensor("weight", (d_model, cfg.dim_feedforward))?;
                let linear2_bias = if cfg.bias_ff {
                    Some(vb.pp("linear2").tensor("bias", (d_model,))?)
                } else {
                    None
                };
                Ok(Self::NoGating {
                    linear1_weight,
                    linear1_bias,
                    linear2_weight,
                    linear2_bias,
                })
            }
            Some(activation) => {
                let hidden = if cfg.dim_feedforward == 4 * d_model {
                    11 * d_model / 4
                } else {
                    2 * cfg.dim_feedforward / 3
                };
                let vb = vb.pp("gating");
                let linear_in_weight =
                    vb.pp("linear_in").tensor("weight", (2 * hidden, d_model))?;
                let linear_in_bias = if cfg.bias_ff {
                    Some(vb.pp("linear_in").tensor("bias", (2 * hidden,))?)
                } else {
                    None
                };
                let linear_out_weight = vb.pp("linear_out").tensor("weight", (d_model, hidden))?;
                let linear_out_bias = if cfg.bias_ff {
                    Some(vb.pp("linear_out").tensor("bias", (d_model,))?)
                } else {
                    None
                };
                Ok(Self::Gating {
                    linear_in_weight,
                    linear_in_bias,
                    linear_out_weight,
                    linear_out_bias,
                    activation,
                })
            }
        }
    }

    #[tracing::instrument(name = "mlp-forward", skip_all)]
    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Self::NoGating {
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
            } => {
                let mut xs = xs.matmul_t(linear1_weight)?;
                if let Some(bias) = linear1_bias {
                    xs = xs.broadcast_add(bias)?;
                }
                xs = xs.gelu_erf()?;
                xs = xs.matmul_t(linear2_weight)?;
                if let Some(bias) = linear2_bias {
                    xs = xs.broadcast_add(bias)?;
                }
                Ok(xs)
            }
            Self::Gating {
                linear_in_weight,
                linear_in_bias,
                linear_out_weight,
                linear_out_bias,
                activation,
            } => {
                let mut xs = xs.matmul_t(linear_in_weight)?;
                if let Some(bias) = linear_in_bias {
                    xs = xs.broadcast_add(bias)?;
                }
                let (b, t, _) = xs.dims3()?;
                let xs = xs.reshape((b, t, 2, ()))?;
                let x1 = xs.narrow(2, ..1)?.contiguous()?.reshape((b, t, ()))?;
                let x2 = xs.narrow(2, 1..2)?.contiguous()?.reshape((b, t, ()))?;
                let xs = activation.apply(&x1)?.mul(&x2)?;
                let mut xs = xs.matmul_t(linear_out_weight)?;
                if let Some(bias) = linear_out_bias {
                    xs = xs.broadcast_add(bias)?;
                }
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
    out_proj_weight: Tensor<T, B>,
    out_proj_bias: Option<Tensor<T, B>>,
    num_heads: usize,
    head_dim: usize,
    kv_cache: KvCache<T, B>,
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

        let vb_out = vb_attn.pp("out_proj");
        let out_proj_weight = vb_out.tensor("weight", (d_model, d_model))?;
        let out_proj_bias = if cfg.bias_attn {
            Some(vb_out.tensor("bias", (d_model,))?)
        } else {
            None
        };

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            num_heads,
            head_dim,
            kv_cache: KvCache::new(cfg.context),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor<T, B>,
        rope: Option<&RotaryEmbedding<T, B>>,
        offset: usize,
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
            let q = q.rope_i(&rope.cos, &rope.sin, offset)?;
            let k = k.rope_i(&rope.cos, &rope.sin, offset)?;
            (q, k)
        } else {
            (q, k)
        };

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Attention
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?;
        let attn_weights = attn_weights.apply_causality_mask(offset)?;
        let attn_weights = attn_weights.softmax()?;

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, t, self.num_heads * self.head_dim))?;

        let mut out = xn::ops::matmul_t(&attn_output, &self.out_proj_weight)?;
        if let Some(bias) = &self.out_proj_bias {
            out = out.broadcast_add(bias)?;
        }
        Ok(out)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache.reset();
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
        &mut self,
        xs: &Tensor<T, B>,
        rope: Option<&RotaryEmbedding<T, B>>,
        offset: usize,
    ) -> Result<Tensor<T, B>> {
        // Pre-norm: xs + layer_scale_1(self_attn(norm1(xs)))
        let norm1_out = self.norm1.forward(xs)?;
        let mut attn_out = self.self_attn.forward(&norm1_out, rope, offset)?;
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let xs = xs.add(&attn_out)?;

        // xs + layer_scale_2(mlp(norm2(xs)))
        let norm2_out = self.norm2.forward(&xs)?;
        let mut mlp_out = self.mlp.forward(&norm2_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            mlp_out = ls.forward(&mlp_out)?;
        }
        xs.add(&mlp_out)
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }
}

// ============================================================================
// Streaming Transformer
// ============================================================================

struct StreamingTransformer<T: WithDTypeF, B: Backend> {
    layers: Vec<StreamingTransformerLayer<T, B>>,
    rope: Option<RotaryEmbedding<T, B>>,
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

    fn forward(&mut self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let offset = self.current_seq_len();
        let mut xs = xs.clone();
        for layer in &mut self.layers {
            xs = layer.forward(&xs, self.rope.as_ref(), offset)?;
        }
        Ok(xs)
    }

    fn current_seq_len(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].self_attn.kv_cache.current_seq_len()
        }
    }

    fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_kv_cache();
        }
    }
}

// ============================================================================
// Projected Transformer (public)
// ============================================================================

pub struct ProjectedTransformer<T: WithDTypeF, B: Backend> {
    input_proj: Option<Tensor<T, B>>,
    output_proj: Option<Tensor<T, B>>,
    transformer: StreamingTransformer<T, B>,
    conv_layout: bool,
}

impl<T: WithDTypeF, B: Backend> ProjectedTransformer<T, B> {
    pub fn load(vb: &Path<B>, input_dim: usize, cfg: &Config, device: &B) -> Result<Self> {
        let input_proj = if input_dim != cfg.d_model {
            Some(
                vb.pp("input_proj")
                    .tensor("weight", (cfg.d_model, input_dim))?,
            )
        } else {
            None
        };

        let output_proj = if input_dim != cfg.d_model {
            Some(
                vb.pp("output_projs")
                    .pp(0)
                    .tensor("weight", (input_dim, cfg.d_model))?,
            )
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

    pub fn forward(&mut self, xs: &Tensor<T, B>) -> Result<Vec<Tensor<T, B>>> {
        let xs = if self.conv_layout {
            xs.transpose(1, 2)?.contiguous()?
        } else {
            xs.clone()
        };
        let xs = match &self.input_proj {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };
        let xs = self.transformer.forward(&xs)?;
        let ys = match &self.output_proj {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };
        let ys = if self.conv_layout {
            ys.transpose(1, 2)?.contiguous()?
        } else {
            ys
        };
        Ok(vec![ys])
    }

    pub fn reset_state(&mut self) {
        self.transformer.reset_state();
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for ProjectedTransformer<T, B> {
    fn reset_state(&mut self) {
        ProjectedTransformer::reset_state(self);
    }

    #[tracing::instrument(name = "transformer", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, _mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                let results = self.forward(xs)?;
                Ok(StreamTensor::from_tensor(
                    results.into_iter().next().unwrap(),
                ))
            }
        }
    }
}
