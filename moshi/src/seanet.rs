use crate::conv::{Norm, PadMode, StreamableConv1d, StreamableConvTranspose1d};
use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Activation {
    Elu(f32),
    Gelu,
    Relu,
    Silu,
    Tanh,
    Sigmoid,
}

impl Activation {
    pub fn apply<T: WithDTypeF, B: Backend>(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Activation::Elu(alpha) => xs.elu(*alpha),
            Activation::Gelu => xs.gelu_erf(),
            Activation::Relu => xs.relu(),
            Activation::Silu => xs.silu(),
            Activation::Tanh => xs.tanh(),
            Activation::Sigmoid => xs.sigmoid(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub dimension: usize,
    pub channels: usize,
    pub causal: bool,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub activation: Activation,
    pub norm: Norm,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: PadMode,
    pub true_skip: bool,
    pub compress: usize,
    pub lstm: usize,
    pub disable_norm_outer_blocks: usize,
    pub final_activation: Option<Activation>,
}

// ============================================================================
// SeaNetResnetBlock
// ============================================================================

pub struct SeaNetResnetBlock<T: WithDTypeF, B: Backend> {
    block: Vec<StreamableConv1d<T, B>>,
    shortcut: Option<StreamableConv1d<T, B>>,
    activation: Activation,
}

impl<T: WithDTypeF, B: Backend> SeaNetResnetBlock<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        k_sizes_and_dilations: &[(usize, usize)],
        activation: Activation,
        norm: Option<Norm>,
        causal: bool,
        pad_mode: PadMode,
        compress: usize,
        true_skip: bool,
    ) -> Result<Self> {
        let hidden = dim / compress;
        let vb_b = vb.pp("block");
        let mut block = Vec::with_capacity(k_sizes_and_dilations.len());

        for (i, &(k_size, dilation)) in k_sizes_and_dilations.iter().enumerate() {
            let in_c = if i == 0 { dim } else { hidden };
            let out_c = if i == k_sizes_and_dilations.len() - 1 {
                dim
            } else {
                hidden
            };
            let c = StreamableConv1d::load(
                &vb_b.pp(2 * i + 1),
                in_c,
                out_c,
                k_size,
                /* stride */ 1,
                /* dilation */ dilation,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ causal,
                /* norm */ norm,
                /* pad_mode */ pad_mode,
            )?;
            block.push(c);
        }

        let shortcut = if true_skip {
            None
        } else {
            let c = StreamableConv1d::load(
                &vb.pp("shortcut"),
                dim,
                dim,
                /* k_size */ 1,
                /* stride */ 1,
                /* dilation */ 1,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ causal,
                /* norm */ norm,
                /* pad_mode */ pad_mode,
            )?;
            Some(c)
        };

        Ok(Self {
            block,
            shortcut,
            activation,
        })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let mut ys = xs.clone();
        for conv in &self.block {
            ys = self.activation.apply(&ys)?;
            ys = conv.forward(&ys)?;
        }
        match &self.shortcut {
            None => ys.add(xs),
            Some(shortcut) => ys.add(&shortcut.forward(xs)?),
        }
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for SeaNetResnetBlock<T, B> {
    fn reset_state(&mut self) {
        for conv in &mut self.block {
            conv.reset_state();
        }
        if let Some(shortcut) = &mut self.shortcut {
            shortcut.reset_state();
        }
    }

    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs,
        };

        let mut ys = StreamTensor::from_tensor(xs.clone());
        for conv in &mut self.block {
            if let Some(y) = ys.as_option() {
                let y = self.activation.apply(y)?;
                ys = conv.step(&StreamTensor::from_tensor(y), mask)?;
            }
        }

        let ys = match ys.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(ys) => ys,
        };

        let result = match &mut self.shortcut {
            None => ys.add(xs)?,
            Some(shortcut) => {
                let short = shortcut.step(&StreamTensor::from_tensor(xs.clone()), mask)?;
                match short.as_option() {
                    Some(s) => ys.add(s)?,
                    None => return Ok(StreamTensor::empty()),
                }
            }
        };
        Ok(StreamTensor::from_tensor(result))
    }
}

// ============================================================================
// SeaNetEncoder
// ============================================================================

struct EncoderLayer<T: WithDTypeF, B: Backend> {
    residuals: Vec<SeaNetResnetBlock<T, B>>,
    downsample: StreamableConv1d<T, B>,
}

pub struct SeaNetEncoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamableConv1d<T, B>,
    activation: Activation,
    layers: Vec<EncoderLayer<T, B>>,
    final_conv: StreamableConv1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> SeaNetEncoder<T, B> {
    pub fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        if cfg.lstm > 0 {
            xn::bail!("seanet lstm is not supported")
        }
        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1usize;
        let init_norm = if cfg.disable_norm_outer_blocks >= 1 {
            None
        } else {
            Some(cfg.norm)
        };
        let mut layer_idx = 0;
        let vb = vb.pp("model");

        let init_conv = StreamableConv1d::load(
            &vb.pp(layer_idx),
            cfg.channels,
            mult * cfg.n_filters,
            cfg.kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ init_norm,
            /* pad_mode */ cfg.pad_mode,
        )?;
        layer_idx += 1;

        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().rev().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks >= i + 2 {
                None
            } else {
                Some(cfg.norm)
            };

            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let block = SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * cfg.n_filters,
                    &[
                        (cfg.residual_kernel_size, cfg.dilation_base.pow(j as u32)),
                        (1, 1),
                    ],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress,
                    cfg.true_skip,
                )?;
                residuals.push(block);
                layer_idx += 1;
            }

            let downsample = StreamableConv1d::load(
                &vb.pp(layer_idx + 1),
                mult * cfg.n_filters,
                mult * cfg.n_filters * 2,
                /* k_size */ ratio * 2,
                /* stride */ ratio,
                /* dilation */ 1,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ true,
                /* norm */ norm,
                /* pad_mode */ cfg.pad_mode,
            )?;
            layer_idx += 2;
            layers.push(EncoderLayer {
                residuals,
                downsample,
            });
            mult *= 2;
        }

        let final_norm = if cfg.disable_norm_outer_blocks >= n_blocks {
            None
        } else {
            Some(cfg.norm)
        };
        let final_conv = StreamableConv1d::load(
            &vb.pp(layer_idx + 1),
            mult * cfg.n_filters,
            cfg.dimension,
            cfg.last_kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ final_norm,
            /* pad_mode */ cfg.pad_mode,
        )?;

        Ok(Self {
            init_conv,
            activation: cfg.activation,
            layers,
            final_conv,
        })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let mut xs = self.init_conv.forward(xs)?;
        for layer in &self.layers {
            for residual in &layer.residuals {
                xs = residual.forward(&xs)?;
            }
            xs = self.activation.apply(&xs)?;
            xs = layer.downsample.forward(&xs)?;
        }
        xs = self.activation.apply(&xs)?;
        self.final_conv.forward(&xs)
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for SeaNetEncoder<T, B> {
    fn reset_state(&mut self) {
        self.init_conv.reset_state();
        for layer in &mut self.layers {
            for residual in &mut layer.residuals {
                residual.reset_state();
            }
            layer.downsample.reset_state();
        }
        self.final_conv.reset_state();
    }

    #[tracing::instrument(name = "sea-encoder", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, m: &StreamMask) -> Result<StreamTensor<T, B>> {
        let mut xs = self.init_conv.step(xs, m)?;
        for layer in &mut self.layers {
            for residual in &mut layer.residuals {
                xs = residual.step(&xs, m)?;
            }
            if let Some(x) = xs.as_option() {
                let x = self.activation.apply(x)?;
                xs = layer.downsample.step(&StreamTensor::from_tensor(x), m)?;
            }
        }
        if let Some(x) = xs.as_option() {
            let x = self.activation.apply(x)?;
            self.final_conv.step(&StreamTensor::from_tensor(x), m)
        } else {
            Ok(StreamTensor::empty())
        }
    }
}

// ============================================================================
// SeaNetDecoder
// ============================================================================

struct DecoderLayer<T: WithDTypeF, B: Backend> {
    upsample: StreamableConvTranspose1d<T, B>,
    residuals: Vec<SeaNetResnetBlock<T, B>>,
}

pub struct SeaNetDecoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamableConv1d<T, B>,
    activation: Activation,
    layers: Vec<DecoderLayer<T, B>>,
    final_conv: StreamableConv1d<T, B>,
    final_activation: Option<Activation>,
}

impl<T: WithDTypeF, B: Backend> SeaNetDecoder<T, B> {
    pub fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        if cfg.lstm > 0 {
            xn::bail!("seanet lstm is not supported")
        }
        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1 << cfg.ratios.len();
        let init_norm = if cfg.disable_norm_outer_blocks == n_blocks {
            None
        } else {
            Some(cfg.norm)
        };
        let mut layer_idx = 0;
        let vb = vb.pp("model");

        let init_conv = StreamableConv1d::load(
            &vb.pp(layer_idx),
            cfg.dimension,
            mult * cfg.n_filters,
            cfg.kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ init_norm,
            /* pad_mode */ cfg.pad_mode,
        )?;
        layer_idx += 1;

        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks + i + 1 >= n_blocks {
                None
            } else {
                Some(cfg.norm)
            };

            let upsample = StreamableConvTranspose1d::load(
                &vb.pp(layer_idx + 1),
                mult * cfg.n_filters,
                mult * cfg.n_filters / 2,
                /* k_size */ ratio * 2,
                /* stride */ ratio,
                /* groups */ 1,
                /* bias */ true,
                /* causal */ true,
                /* norm */ norm,
            )?;
            layer_idx += 2;

            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let block = SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * cfg.n_filters / 2,
                    &[
                        (cfg.residual_kernel_size, cfg.dilation_base.pow(j as u32)),
                        (1, 1),
                    ],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress,
                    cfg.true_skip,
                )?;
                residuals.push(block);
                layer_idx += 1;
            }

            layers.push(DecoderLayer {
                upsample,
                residuals,
            });
            mult /= 2;
        }

        let final_norm = if cfg.disable_norm_outer_blocks >= 1 {
            None
        } else {
            Some(cfg.norm)
        };
        let final_conv = StreamableConv1d::load(
            &vb.pp(layer_idx + 1),
            cfg.n_filters,
            cfg.channels,
            cfg.last_kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ true,
            /* causal */ cfg.causal,
            /* norm */ final_norm,
            /* pad_mode */ cfg.pad_mode,
        )?;

        Ok(Self {
            init_conv,
            activation: cfg.activation,
            layers,
            final_conv,
            final_activation: cfg.final_activation,
        })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let mut xs = self.init_conv.forward(xs)?;
        for layer in &self.layers {
            xs = self.activation.apply(&xs)?;
            xs = layer.upsample.forward(&xs)?;
            for residual in &layer.residuals {
                xs = residual.forward(&xs)?;
            }
        }
        xs = self.activation.apply(&xs)?;
        xs = self.final_conv.forward(&xs)?;
        if let Some(act) = &self.final_activation {
            xs = act.apply(&xs)?;
        }
        Ok(xs)
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for SeaNetDecoder<T, B> {
    fn reset_state(&mut self) {
        self.init_conv.reset_state();
        for layer in &mut self.layers {
            layer.upsample.reset_state();
            for residual in &mut layer.residuals {
                residual.reset_state();
            }
        }
        self.final_conv.reset_state();
    }

    #[tracing::instrument(name = "sea-decoder", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, m: &StreamMask) -> Result<StreamTensor<T, B>> {
        let mut xs = self.init_conv.step(xs, m)?;
        for layer in &mut self.layers {
            if let Some(x) = xs.as_option() {
                let x = self.activation.apply(x)?;
                xs = layer.upsample.step(&StreamTensor::from_tensor(x), m)?;
            }
            for residual in &mut layer.residuals {
                xs = residual.step(&xs, m)?;
            }
        }
        if let Some(x) = xs.as_option() {
            let mut x = self.activation.apply(x)?;
            let result = self
                .final_conv
                .step(&StreamTensor::from_tensor(x.clone()), m)?;
            if let (Some(r), Some(act)) = (result.as_option(), &self.final_activation) {
                x = act.apply(r)?;
                return Ok(StreamTensor::from_tensor(x));
            }
            return Ok(result);
        }
        Ok(StreamTensor::empty())
    }
}
