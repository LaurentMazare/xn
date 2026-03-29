use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

#[cfg(feature = "cuda")]
use xn::cuda_backend::{Device as CudaDevice, quantization::Fp8Linear};

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
    #[cfg(feature = "cuda")]
    Fp8Gating {
        linear_in: Fp8Linear,
        linear_out: Fp8Linear,
        activation: crate::seanet::Activation,
        hidden: usize,
        _phantom: std::marker::PhantomData<(T, B)>,
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
                Ok(Self::Gating { linear_in, linear_out, activation })
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
            #[cfg(feature = "cuda")]
            Self::Fp8Gating { linear_in, linear_out, activation, hidden, .. } => {
                use std::any::Any;
                let xs_any: &dyn Any = xs;
                let xs_bf16: &Tensor<half::bf16, CudaDevice> =
                    xs_any.downcast_ref().expect("Fp8Gating requires bf16 CUDA tensors");

                let (b, t, _) = xs_bf16.dims3()?;
                let xs = linear_in.forward(xs_bf16)?;
                let xs = xs.reshape((b, t, 2, *hidden))?;
                let x1 = xs.narrow(2, ..1)?.contiguous()?.reshape((b, t, *hidden))?;
                let x2 = xs.narrow(2, 1..2)?.contiguous()?.reshape((b, t, *hidden))?;
                let xs = activation.apply(&x1)?.mul(&x2)?;
                let xs = linear_out.forward(&xs)?;

                let result_any: Box<dyn Any> = Box::new(xs);
                Ok(*result_any.downcast::<Tensor<T, B>>().expect("type mismatch"))
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl Mlp<half::bf16, CudaDevice> {
    pub(crate) fn load_fp8(vb: &Path<CudaDevice>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                // No FP8 variant for NoGating, fall back to standard
                Mlp::load(vb, cfg)
            }
            Some(activation) => {
                let hidden = if cfg.dim_feedforward == 4 * d_model {
                    11 * d_model / 4
                } else {
                    2 * cfg.dim_feedforward / 3
                };
                let vb = vb.pp("gating");
                let linear_in =
                    Fp8Linear::load(&vb.pp("linear_in"), d_model, 2 * hidden, cfg.bias_ff)?;
                let linear_out =
                    Fp8Linear::load(&vb.pp("linear_out"), hidden, d_model, cfg.bias_ff)?;
                Ok(Self::Fp8Gating {
                    linear_in,
                    linear_out,
                    activation,
                    hidden,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }
}
