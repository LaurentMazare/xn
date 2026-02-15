#![allow(clippy::too_many_arguments)]
// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Mimi audio tokenizer model - compatible with the candle implementation.

use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDType, WithDTypeF};

// ============================================================================
// Streaming primitives
// ============================================================================

/// A tensor that may be empty, used in streaming contexts.
pub struct StreamTensor<T: WithDType, B: Backend>(Option<Tensor<T, B>>);

impl<T: WithDType, B: Backend> StreamTensor<T, B> {
    pub fn empty() -> Self {
        Self(None)
    }

    pub fn from_tensor(tensor: Tensor<T, B>) -> Self {
        Self(Some(tensor))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn as_option(&self) -> Option<&Tensor<T, B>> {
        self.0.as_ref()
    }

    pub fn take(self) -> Option<Tensor<T, B>> {
        self.0
    }
}

impl<T: WithDType, B: Backend> Default for StreamTensor<T, B> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: WithDType, B: Backend> From<()> for StreamTensor<T, B> {
    fn from(_: ()) -> Self {
        Self::empty()
    }
}

impl<T: WithDTypeF, B: Backend> From<Tensor<T, B>> for StreamTensor<T, B> {
    fn from(t: Tensor<T, B>) -> Self {
        Self::from_tensor(t)
    }
}

impl<T: WithDTypeF, B: Backend> From<Option<Tensor<T, B>>> for StreamTensor<T, B> {
    fn from(t: Option<Tensor<T, B>>) -> Self {
        Self(t)
    }
}

/// Mask for batch elements in streaming mode.
#[derive(Clone, Default)]
pub struct StreamMask(Option<Vec<bool>>);

impl StreamMask {
    pub fn empty() -> Self {
        Self(None)
    }

    pub fn new(mask: Vec<bool>) -> Self {
        Self(Some(mask))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn is_active(&self, batch_idx: usize) -> bool {
        self.0.as_ref().is_none_or(|v| v[batch_idx])
    }
}

impl From<()> for StreamMask {
    fn from(_: ()) -> Self {
        Self::empty()
    }
}

/// Trait for streaming modules that process data step by step.
pub trait StreamingModule<T: WithDTypeF, B: Backend> {
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>>;
    fn reset_state(&mut self);
}

// ============================================================================
// Convolution types
// ============================================================================

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Norm {
    WeightNorm,
    SpectralNorm,
    TimeGroupNorm,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

/// Activation function.
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

/// 1D Convolution.
pub struct Conv1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Option<Tensor<T, B>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl<T: WithDTypeF, B: Backend> Conv1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (out_channels, in_channels / groups, kernel_size))?;
        let bias = if use_bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, padding, dilation, groups })
    }

    /// Load with weight norm (weight_g, weight_v instead of weight).
    pub fn load_weight_norm(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight_g = vb.tensor("weight_g", (out_channels, 1, 1))?;
        let weight_v = vb.tensor("weight_v", (out_channels, in_channels / groups, kernel_size))?;
        // Compute normalized weight: weight = weight_g * weight_v / ||weight_v||
        let norm_v = weight_v.sqr()?.sum_keepdim(vec![1, 2])?.sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
        let bias = if use_bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, padding, dilation, groups })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

/// 1D Transposed Convolution.
pub struct ConvTranspose1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Option<Tensor<T, B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
}

impl<T: WithDTypeF, B: Backend> ConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (in_channels, out_channels / groups, kernel_size))?;
        let bias = if use_bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, padding, output_padding, groups })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.conv_transpose1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
    }
}

/// Streamable 1D convolution with causal padding support.
pub struct StreamableConv1d<T: WithDTypeF, B: Backend> {
    conv: Conv1d<T, B>,
    causal: bool,
    pad_mode: PadMode,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    state_prev_xs: Option<Tensor<T, B>>,
    left_pad_applied: bool,
}

impl<T: WithDTypeF, B: Backend> StreamableConv1d<T, B> {
    pub fn new(
        conv: Conv1d<T, B>,
        causal: bool,
        pad_mode: PadMode,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
    ) -> Self {
        Self {
            conv,
            causal,
            pad_mode,
            kernel_size,
            stride,
            dilation,
            state_prev_xs: None,
            left_pad_applied: false,
        }
    }

    fn padding_total(&self) -> usize {
        // Effective kernel size with dilations
        let k_size = (self.kernel_size - 1) * self.dilation + 1;
        k_size - self.stride
    }

    fn pad1d(&self, xs: &Tensor<T, B>, pad_l: usize, pad_r: usize) -> Result<Tensor<T, B>> {
        match self.pad_mode {
            PadMode::Constant => xs.pad_with_zeros(2, pad_l, pad_r), // dim 2 = last dim for [B, C, T]
            PadMode::Replicate => xs.pad_with_same(2, pad_l, pad_r),
            PadMode::Reflect => xs.pad_with_zeros(2, pad_l, pad_r), // fallback to zeros for now
        }
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let padding_total = self.padding_total();
        let xs = if self.causal {
            self.pad1d(xs, padding_total, 0)?
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            self.pad1d(xs, padding_left, padding_right)?
        };
        self.conv.forward(&xs)
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for StreamableConv1d<T, B> {
    #[tracing::instrument(name = "streamable-conv1d", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, _mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs,
        };

        // Apply left padding on first step if not done yet
        let xs = if self.left_pad_applied {
            xs.clone()
        } else {
            self.left_pad_applied = true;
            self.pad1d(xs, self.padding_total(), 0)?
        };

        // Concatenate with previous state
        let xs = match &self.state_prev_xs {
            None => xs,
            Some(prev) => Tensor::cat(&[prev, &xs], 2)?, // cat along time dim
        };

        let seq_len = xs.dim(2)?;
        let kernel = (self.kernel_size - 1) * self.dilation + 1;
        let num_frames = seq_len.saturating_sub(kernel) / self.stride + 1;

        if num_frames > 0 {
            let offset = num_frames * self.stride;
            // Save remaining for next step
            if seq_len > offset {
                self.state_prev_xs = Some(xs.narrow(2, offset..seq_len)?.contiguous()?);
            } else {
                self.state_prev_xs = None;
            }
            // Process current frames
            let in_len = (num_frames - 1) * self.stride + kernel;
            let xs_in = xs.narrow(2, ..in_len)?.contiguous()?;
            Ok(StreamTensor::from_tensor(self.conv.forward(&xs_in)?))
        } else {
            self.state_prev_xs = Some(xs);
            Ok(StreamTensor::empty())
        }
    }

    fn reset_state(&mut self) {
        self.state_prev_xs = None;
        self.left_pad_applied = false;
    }
}

/// Streamable 1D transposed convolution.
pub struct StreamableConvTranspose1d<T: WithDTypeF, B: Backend> {
    convtr: ConvTranspose1d<T, B>,
    causal: bool,
    kernel_size: usize,
    stride: usize,
    state_prev_ys: Option<Tensor<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamableConvTranspose1d<T, B> {
    pub fn new(
        convtr: ConvTranspose1d<T, B>,
        causal: bool,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self { convtr, causal, kernel_size, stride, state_prev_ys: None }
    }

    fn unpad1d(xs: &Tensor<T, B>, unpad_l: usize, unpad_r: usize) -> Result<Tensor<T, B>> {
        let len = xs.dim(2)?;
        if len < unpad_l + unpad_r {
            crate::bail!("unpad1d: tensor len {len} is too low for unpad {unpad_l} + {unpad_r}");
        }
        xs.narrow(2, unpad_l..len - unpad_r)?.contiguous()
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let padding_total = self.kernel_size.saturating_sub(self.stride);
        let xs = self.convtr.forward(xs)?;
        if self.causal {
            Self::unpad1d(&xs, 0, padding_total)
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            Self::unpad1d(&xs, padding_left, padding_right)
        }
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for StreamableConvTranspose1d<T, B> {
    #[tracing::instrument(name = "streamable-convtr1d", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, _mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            Some(xs) => xs,
            None => return Ok(StreamTensor::empty()),
        };

        // Apply convtr without unpadding
        let ys = self.convtr.forward(xs)?;
        let ot = ys.dim(2)?;

        // Add overlap from previous step
        let ys = match &self.state_prev_ys {
            None => ys,
            Some(prev_ys) => {
                let pt = prev_ys.dim(2)?;
                // Subtract bias from prev (as it will be added again)
                let prev_ys = match &self.convtr.bias {
                    None => prev_ys.clone(),
                    Some(bias) => {
                        let bias = bias.reshape((1, bias.elem_count(), 1))?;
                        prev_ys.broadcast_sub(&bias)?
                    }
                };
                let ys1 = ys.narrow(2, ..pt)?.contiguous()?.add(&prev_ys)?;
                let ys2 = ys.narrow(2, pt..ot)?.contiguous()?;
                Tensor::cat(&[&ys1, &ys2], 2)?
            }
        };

        // Split into valid output and overlap for next step
        let invalid_steps = self.kernel_size - self.stride;
        let valid_len = ot.saturating_sub(invalid_steps);
        if valid_len > 0 {
            let valid = ys.narrow(2, ..valid_len)?.contiguous()?;
            if ot > valid_len {
                self.state_prev_ys = Some(ys.narrow(2, valid_len..ot)?.contiguous()?);
            } else {
                self.state_prev_ys = None;
            }
            Ok(StreamTensor::from_tensor(valid))
        } else {
            self.state_prev_ys = Some(ys);
            Ok(StreamTensor::empty())
        }
    }

    fn reset_state(&mut self) {
        self.state_prev_ys = None;
    }
}

/// Downsampling via learned convolution.
pub struct ConvDownsample1d<T: WithDTypeF, B: Backend> {
    conv: StreamableConv1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvDownsample1d<T, B> {
    pub fn load(vb: &Path<B>, stride: usize, dim: usize, causal: bool) -> Result<Self> {
        let kernel_size = 2 * stride;
        let conv_vb = vb.pp("conv").pp("conv").pp("conv");
        let inner = Conv1d::load(&conv_vb, dim, dim, kernel_size, stride, 0, 1, 1, false)?;
        let conv = StreamableConv1d::new(inner, causal, PadMode::Replicate, kernel_size, stride, 1);
        Ok(Self { conv })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.conv.forward(xs)
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for ConvDownsample1d<T, B> {
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        self.conv.step(xs, mask)
    }

    fn reset_state(&mut self) {
        self.conv.reset_state();
    }
}

/// Upsampling via learned transposed convolution.
pub struct ConvTrUpsample1d<T: WithDTypeF, B: Backend> {
    convtr: StreamableConvTranspose1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvTrUpsample1d<T, B> {
    pub fn load(vb: &Path<B>, stride: usize, dim: usize, causal: bool) -> Result<Self> {
        let kernel_size = 2 * stride;
        let convtr_vb = vb.pp("convtr").pp("convtr").pp("convtr");
        let inner = ConvTranspose1d::load(
            &convtr_vb,
            dim,
            dim,
            kernel_size,
            stride,
            0,
            0,
            dim, // depthwise
            false,
        )?;
        let convtr = StreamableConvTranspose1d::new(inner, causal, kernel_size, stride);
        Ok(Self { convtr })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.convtr.forward(xs)
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for ConvTrUpsample1d<T, B> {
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        self.convtr.step(xs, mask)
    }

    fn reset_state(&mut self) {
        self.convtr.reset_state();
    }
}

// ============================================================================
// SeaNet Encoder/Decoder
// ============================================================================

/// SeaNet configuration.
#[derive(Debug, Clone)]
pub struct SeaNetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub causal: bool,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub activation: Activation,
    pub compress: usize,
    pub dilation_base: usize,
    pub disable_norm_outer_blocks: usize,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub lstm: usize,
    pub norm: Norm,
    pub pad_mode: PadMode,
    pub ratios: Vec<usize>,
    pub true_skip: bool,
    pub final_activation: Option<Activation>,
}

/// SeaNet resnet block with skip connection.
pub struct SeaNetResnetBlock<T: WithDTypeF, B: Backend> {
    block: Vec<StreamableConv1d<T, B>>,
    shortcut: Option<StreamableConv1d<T, B>>,
    activation: Activation,
}

impl<T: WithDTypeF, B: Backend> SeaNetResnetBlock<T, B> {
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
            let out_c = if i == k_sizes_and_dilations.len() - 1 { dim } else { hidden };

            let conv_vb = vb_b.pp(2 * i + 1).pp("conv").pp("conv");
            let inner = match norm {
                Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                    &conv_vb, in_c, out_c, k_size, 1, 0, dilation, 1, true,
                )?,
                _ => Conv1d::load(&conv_vb, in_c, out_c, k_size, 1, 0, dilation, 1, true)?,
            };
            let conv = StreamableConv1d::new(inner, causal, pad_mode, k_size, 1, dilation);
            block.push(conv);
        }

        let shortcut = if true_skip {
            None
        } else {
            let conv_vb = vb.pp("shortcut").pp("conv").pp("conv");
            let inner = match norm {
                Some(Norm::WeightNorm) => {
                    Conv1d::load_weight_norm(&conv_vb, dim, dim, 1, 1, 0, 1, 1, true)?
                }
                _ => Conv1d::load(&conv_vb, dim, dim, 1, 1, 0, 1, 1, true)?,
            };
            Some(StreamableConv1d::new(inner, causal, pad_mode, 1, 1, 1))
        };

        Ok(Self { block, shortcut, activation })
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
    #[tracing::instrument(name = "seanet-resnet-block", skip_all)]
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

    fn reset_state(&mut self) {
        for conv in &mut self.block {
            conv.reset_state();
        }
        if let Some(shortcut) = &mut self.shortcut {
            shortcut.reset_state();
        }
    }
}

/// Encoder layer (residuals + downsample).
struct EncoderLayer<T: WithDTypeF, B: Backend> {
    residuals: Vec<SeaNetResnetBlock<T, B>>,
    downsample: StreamableConv1d<T, B>,
}

/// SeaNet encoder - audio -> latent representation.
pub struct SeaNetEncoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamableConv1d<T, B>,
    layers: Vec<EncoderLayer<T, B>>,
    final_conv: StreamableConv1d<T, B>,
    activation: Activation,
}

impl<T: WithDTypeF, B: Backend> SeaNetEncoder<T, B> {
    pub fn load(vb: &Path<B>, cfg: &SeaNetConfig) -> Result<Self> {
        if cfg.lstm > 0 {
            crate::bail!("LSTM in SeaNet is not supported");
        }

        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1usize;
        let mut layer_idx = 0;
        let vb = vb.pp("model");

        // Initial convolution
        let init_norm = if cfg.disable_norm_outer_blocks >= 1 { None } else { Some(cfg.norm) };
        let init_conv_vb = vb.pp(layer_idx).pp("conv").pp("conv");
        let init_inner = match init_norm {
            Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                &init_conv_vb,
                cfg.channels,
                mult * cfg.n_filters,
                cfg.kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
            _ => Conv1d::load(
                &init_conv_vb,
                cfg.channels,
                mult * cfg.n_filters,
                cfg.kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
        };
        let init_conv =
            StreamableConv1d::new(init_inner, cfg.causal, cfg.pad_mode, cfg.kernel_size, 1, 1);
        layer_idx += 1;

        // Encoder layers
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().rev().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks >= i + 2 { None } else { Some(cfg.norm) };

            // Residual blocks
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                let block = SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * cfg.n_filters,
                    &[(cfg.residual_kernel_size, dilation), (1, 1)],
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

            // Downsample
            let k_size = ratio * 2;
            let down_conv_vb = vb.pp(layer_idx + 1).pp("conv").pp("conv");
            let down_inner = match norm {
                Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                    &down_conv_vb,
                    mult * cfg.n_filters,
                    mult * cfg.n_filters * 2,
                    k_size,
                    ratio,
                    0,
                    1,
                    1,
                    true,
                )?,
                _ => Conv1d::load(
                    &down_conv_vb,
                    mult * cfg.n_filters,
                    mult * cfg.n_filters * 2,
                    k_size,
                    ratio,
                    0,
                    1,
                    1,
                    true,
                )?,
            };
            let downsample =
                StreamableConv1d::new(down_inner, true, cfg.pad_mode, k_size, ratio, 1);
            layer_idx += 2;

            layers.push(EncoderLayer { residuals, downsample });
            mult *= 2;
        }

        // Final convolution
        let final_norm =
            if cfg.disable_norm_outer_blocks >= n_blocks { None } else { Some(cfg.norm) };
        let final_conv_vb = vb.pp(layer_idx + 1).pp("conv").pp("conv");
        let final_inner = match final_norm {
            Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                &final_conv_vb,
                mult * cfg.n_filters,
                cfg.dimension,
                cfg.last_kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
            _ => Conv1d::load(
                &final_conv_vb,
                mult * cfg.n_filters,
                cfg.dimension,
                cfg.last_kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
        };
        let final_conv = StreamableConv1d::new(
            final_inner,
            cfg.causal,
            cfg.pad_mode,
            cfg.last_kernel_size,
            1,
            1,
        );

        Ok(Self { init_conv, layers, final_conv, activation: cfg.activation })
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
    #[tracing::instrument(name = "seanet-encoder", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        let mut xs = self.init_conv.step(xs, mask)?;
        for layer in &mut self.layers {
            for residual in &mut layer.residuals {
                xs = residual.step(&xs, mask)?;
            }
            if let Some(x) = xs.as_option() {
                let x = self.activation.apply(x)?;
                xs = layer.downsample.step(&StreamTensor::from_tensor(x), mask)?;
            }
        }
        if let Some(x) = xs.as_option() {
            let x = self.activation.apply(x)?;
            self.final_conv.step(&StreamTensor::from_tensor(x), mask)
        } else {
            Ok(StreamTensor::empty())
        }
    }

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
}

/// Decoder layer (upsample + residuals).
struct DecoderLayer<T: WithDTypeF, B: Backend> {
    upsample: StreamableConvTranspose1d<T, B>,
    residuals: Vec<SeaNetResnetBlock<T, B>>,
}

/// SeaNet decoder - latent representation -> audio.
pub struct SeaNetDecoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamableConv1d<T, B>,
    layers: Vec<DecoderLayer<T, B>>,
    final_conv: StreamableConv1d<T, B>,
    activation: Activation,
    final_activation: Option<Activation>,
}

impl<T: WithDTypeF, B: Backend> SeaNetDecoder<T, B> {
    pub fn load(vb: &Path<B>, cfg: &SeaNetConfig) -> Result<Self> {
        if cfg.lstm > 0 {
            crate::bail!("LSTM in SeaNet is not supported");
        }

        let n_blocks = 2 + cfg.ratios.len();
        let mut mult = 1 << cfg.ratios.len();
        let mut layer_idx = 0;
        let vb = vb.pp("model");

        // Initial convolution
        let init_norm =
            if cfg.disable_norm_outer_blocks == n_blocks { None } else { Some(cfg.norm) };
        let init_conv_vb = vb.pp(layer_idx).pp("conv").pp("conv");
        let init_inner = match init_norm {
            Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                &init_conv_vb,
                cfg.dimension,
                mult * cfg.n_filters,
                cfg.kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
            _ => Conv1d::load(
                &init_conv_vb,
                cfg.dimension,
                mult * cfg.n_filters,
                cfg.kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
        };
        let init_conv =
            StreamableConv1d::new(init_inner, cfg.causal, cfg.pad_mode, cfg.kernel_size, 1, 1);
        layer_idx += 1;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks + i + 1 >= n_blocks {
                None
            } else {
                Some(cfg.norm)
            };

            // Upsample
            let k_size = ratio * 2;
            let up_conv_vb = vb.pp(layer_idx + 1).pp("convtr").pp("convtr");
            let up_inner = ConvTranspose1d::load(
                &up_conv_vb,
                mult * cfg.n_filters,
                mult * cfg.n_filters / 2,
                k_size,
                ratio,
                0,
                0,
                1,
                true,
            )?;
            let upsample = StreamableConvTranspose1d::new(up_inner, true, k_size, ratio);
            layer_idx += 2;

            // Residual blocks
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                let block = SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * cfg.n_filters / 2,
                    &[(cfg.residual_kernel_size, dilation), (1, 1)],
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

            layers.push(DecoderLayer { upsample, residuals });
            mult /= 2;
        }

        // Final convolution
        let final_norm = if cfg.disable_norm_outer_blocks >= 1 { None } else { Some(cfg.norm) };
        let final_conv_vb = vb.pp(layer_idx + 1).pp("conv").pp("conv");
        let final_inner = match final_norm {
            Some(Norm::WeightNorm) => Conv1d::load_weight_norm(
                &final_conv_vb,
                cfg.n_filters,
                cfg.channels,
                cfg.last_kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
            _ => Conv1d::load(
                &final_conv_vb,
                cfg.n_filters,
                cfg.channels,
                cfg.last_kernel_size,
                1,
                0,
                1,
                1,
                true,
            )?,
        };
        let final_conv = StreamableConv1d::new(
            final_inner,
            cfg.causal,
            cfg.pad_mode,
            cfg.last_kernel_size,
            1,
            1,
        );

        Ok(Self {
            init_conv,
            layers,
            final_conv,
            activation: cfg.activation,
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
    #[tracing::instrument(name = "seanet-decoder", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        let mut xs = self.init_conv.step(xs, mask)?;
        for layer in &mut self.layers {
            if let Some(x) = xs.as_option() {
                let x = self.activation.apply(x)?;
                xs = layer.upsample.step(&StreamTensor::from_tensor(x), mask)?;
            }
            for residual in &mut layer.residuals {
                xs = residual.step(&xs, mask)?;
            }
        }
        if let Some(x) = xs.as_option() {
            let mut x = self.activation.apply(x)?;
            let result = self.final_conv.step(&StreamTensor::from_tensor(x.clone()), mask)?;
            if let (Some(r), Some(act)) = (result.as_option(), &self.final_activation) {
                x = act.apply(r)?;
                return Ok(StreamTensor::from_tensor(x));
            }
            return Ok(result);
        }
        Ok(StreamTensor::empty())
    }

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
}

// ============================================================================
// Transformer
// ============================================================================

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
}

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f64>,
    pub context: usize,
    pub conv_kernel_size: usize,
    pub use_conv_bias: bool,
    pub use_conv_block: bool,
    pub max_period: usize,
    pub gating: Option<Activation>,
    pub norm: NormType,
    pub positional_embedding: PositionalEmbedding,
    pub dim_feedforward: usize,
    pub kv_repeat: usize,
    pub conv_layout: bool,
    pub max_seq_len: usize,
}

// KV Cache for streaming attention
struct KvCache<T: WithDTypeF, B: Backend> {
    k: Option<Tensor<T, B>>,
    v: Option<Tensor<T, B>>,
    max_seq_len: usize,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    fn new(max_seq_len: usize) -> Self {
        Self { k: None, v: None, max_seq_len }
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    fn current_seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dims()[2], // k shape: [b, h, seq, d]
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
                // Concatenate along sequence dimension (dim 2)
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        // Trim if exceeds max_seq_len
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

// Rotary position embeddings
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

        // Precompute cos/sin for all positions up to max_seq_len
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

// Layer scale (multiply by learned scale)
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

// Normalization layer (supports both LayerNorm and RmsNorm)
enum TransformerNorm<T: WithDTypeF, B: Backend> {
    LayerNorm { weight: Tensor<T, B>, bias: Tensor<T, B>, eps: f32 },
    RmsNorm { alpha: Tensor<T, B>, eps: f32 },
}

impl<T: WithDTypeF, B: Backend> TransformerNorm<T, B> {
    fn load(vb: &Path<B>, d_model: usize, norm_type: NormType) -> Result<Self> {
        match norm_type {
            NormType::LayerNorm => {
                let weight = if vb.contains("alpha") {
                    vb.tensor("alpha", (1, 1, d_model))?.reshape((d_model,))?
                } else {
                    vb.tensor("weight", (d_model,))?
                };
                let bias = vb.tensor("bias", (d_model,))?;
                Ok(Self::LayerNorm { weight, bias, eps: 1e-5 })
            }
            NormType::RmsNorm => {
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

// MLP (feed-forward network)
struct Mlp<T: WithDTypeF, B: Backend> {
    linear1_weight: Tensor<T, B>,
    linear1_bias: Option<Tensor<T, B>>,
    linear2_weight: Tensor<T, B>,
    linear2_bias: Option<Tensor<T, B>>,
}

impl<T: WithDTypeF, B: Backend> Mlp<T, B> {
    fn load(vb: &Path<B>, d_model: usize, dim_feedforward: usize, bias: bool) -> Result<Self> {
        let linear1_weight = vb.pp("linear1").tensor("weight", (dim_feedforward, d_model))?;
        let linear1_bias =
            if bias { Some(vb.pp("linear1").tensor("bias", (dim_feedforward,))?) } else { None };
        let linear2_weight = vb.pp("linear2").tensor("weight", (d_model, dim_feedforward))?;
        let linear2_bias =
            if bias { Some(vb.pp("linear2").tensor("bias", (d_model,))?) } else { None };
        Ok(Self { linear1_weight, linear1_bias, linear2_weight, linear2_bias })
    }

    #[tracing::instrument(name = "mlp-forward", skip_all)]
    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // xs: [b, t, d_model]
        let mut xs = xs.matmul_t(&self.linear1_weight)?;
        if let Some(bias) = &self.linear1_bias {
            xs = xs.broadcast_add(bias)?;
        }
        xs = xs.gelu_erf()?;
        xs = xs.matmul_t(&self.linear2_weight)?;
        if let Some(bias) = &self.linear2_bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }
}

// Streaming multi-head self-attention
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
    fn load(vb: &Path<B>, cfg: &TransformerConfig) -> Result<Self> {
        let d_model = cfg.d_model;
        let num_heads = cfg.num_heads;
        let head_dim = d_model / num_heads;
        let num_kv = num_heads / cfg.kv_repeat;
        let out_dim = d_model + 2 * num_kv * head_dim;

        let in_proj_weight = vb.pp("self_attn").tensor("in_proj_weight", (out_dim, d_model))?;
        let in_proj_bias = if cfg.bias_attn {
            Some(vb.pp("self_attn").tensor("in_proj_bias", (out_dim,))?)
        } else {
            None
        };

        let out_proj_weight =
            vb.pp("self_attn").pp("out_proj").tensor("weight", (d_model, d_model))?;
        let out_proj_bias = if cfg.bias_attn {
            Some(vb.pp("self_attn").pp("out_proj").tensor("bias", (d_model,))?)
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

        // Project to QKV
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

        // Append to KV cache and get full K, V
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Attention: Q @ K^T / sqrt(head_dim)
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn_weights = q.matmul_t(&k)?.scale(scale)?;

        // Apply causal mask
        let attn_weights = attn_weights.apply_causality_mask(offset)?;

        // Softmax
        let attn_weights = attn_weights.softmax()?;

        // Attention output: weights @ V
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [b, num_heads, t, head_dim] -> [b, t, num_heads, head_dim] -> [b, t, d_model]
        let attn_output =
            attn_output.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;

        // Output projection
        let mut out = crate::ops::matmul_t(&attn_output, &self.out_proj_weight)?;
        if let Some(bias) = &self.out_proj_bias {
            out = out.broadcast_add(bias)?;
        }

        Ok(out)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// Streaming transformer layer
struct StreamingTransformerLayer<T: WithDTypeF, B: Backend> {
    self_attn: StreamingMultiheadAttention<T, B>,
    mlp: Mlp<T, B>,
    norm1: TransformerNorm<T, B>,
    norm2: TransformerNorm<T, B>,
    layer_scale_1: Option<LayerScale<T, B>>,
    layer_scale_2: Option<LayerScale<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformerLayer<T, B> {
    fn load(vb: &Path<B>, cfg: &TransformerConfig) -> Result<Self> {
        let self_attn = StreamingMultiheadAttention::load(vb, cfg)?;
        let mlp = Mlp::load(vb, cfg.d_model, cfg.dim_feedforward, cfg.bias_ff)?;
        let norm1 = TransformerNorm::load(&vb.pp("norm1"), cfg.d_model, cfg.norm)?;
        let norm2 = TransformerNorm::load(&vb.pp("norm2"), cfg.d_model, cfg.norm)?;

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
        &mut self,
        xs: &Tensor<T, B>,
        rope: Option<&RotaryEmbedding<T, B>>,
        offset: usize,
    ) -> Result<Tensor<T, B>> {
        // Pre-norm architecture (norm_first = true)
        // xs + layer_scale_1(self_attn(norm1(xs)))
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

// Streaming transformer (stack of layers)
struct StreamingTransformer<T: WithDTypeF, B: Backend> {
    layers: Vec<StreamingTransformerLayer<T, B>>,
    rope: Option<RotaryEmbedding<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformer<T, B> {
    fn load(vb: &Path<B>, cfg: &TransformerConfig, device: &B) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(StreamingTransformerLayer::load(&vb_layers.pp(i), cfg)?);
        }

        let rope = if cfg.positional_embedding == PositionalEmbedding::Rope {
            let head_dim = cfg.d_model / cfg.num_heads;
            Some(RotaryEmbedding::new(head_dim, cfg.max_seq_len, cfg.max_period as f32, device)?)
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
        if self.layers.is_empty() { 0 } else { self.layers[0].self_attn.kv_cache.current_seq_len() }
    }

    fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_kv_cache();
        }
    }
}

/// Projected transformer with input/output projections.
pub struct Transformer<T: WithDTypeF, B: Backend> {
    input_proj: Option<Tensor<T, B>>,
    output_proj: Option<Tensor<T, B>>,
    transformer: StreamingTransformer<T, B>,
    conv_layout: bool,
}

impl<T: WithDTypeF, B: Backend> Transformer<T, B> {
    pub fn load(
        vb: &Path<B>,
        input_dim: usize,
        cfg: &TransformerConfig,
        device: &B,
    ) -> Result<Self> {
        let input_proj = if input_dim != cfg.d_model {
            Some(vb.pp("input_proj").tensor("weight", (cfg.d_model, input_dim))?)
        } else {
            None
        };

        let output_proj = if input_dim != cfg.d_model {
            Some(vb.pp("output_projs").pp(0).tensor("weight", (input_dim, cfg.d_model))?)
        } else {
            None
        };

        let transformer = StreamingTransformer::load(&vb.pp("transformer"), cfg, device)?;

        Ok(Self { input_proj, output_proj, transformer, conv_layout: cfg.conv_layout })
    }

    pub fn forward(&mut self, xs: &Tensor<T, B>) -> Result<Vec<Tensor<T, B>>> {
        // Apply conv_layout transpose if needed
        let xs = if self.conv_layout { xs.transpose(1, 2)?.contiguous()? } else { xs.clone() };

        // Apply input projection
        let xs = match &self.input_proj {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };

        // Transformer layers
        let xs = self.transformer.forward(&xs)?;

        // Apply output projection
        let ys = match &self.output_proj {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };

        // Transpose back if conv_layout
        let ys = if self.conv_layout { ys.transpose(1, 2)?.contiguous()? } else { ys };

        Ok(vec![ys])
    }

    pub fn reset_state(&mut self) {
        self.transformer.reset_state();
    }
}

impl<T: WithDTypeF, B: Backend> StreamingModule<T, B> for Transformer<T, B> {
    #[tracing::instrument(name = "transformer", skip_all)]
    fn step(&mut self, xs: &StreamTensor<T, B>, _mask: &StreamMask) -> Result<StreamTensor<T, B>> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                let results = self.forward(xs)?;
                Ok(StreamTensor::from_tensor(results.into_iter().next().unwrap()))
            }
        }
    }

    fn reset_state(&mut self) {
        Transformer::reset_state(self);
    }
}

// ============================================================================
// Vector Quantization
// ============================================================================

/// Euclidean codebook for vector quantization.
#[allow(dead_code)]
pub struct EuclideanCodebook<T: WithDTypeF, B: Backend> {
    embedding: Tensor<T, B>,
    c2: Tensor<T, B>, // Precomputed: (embedding * embedding).sum(dim=-1) / 2.0
    dim: usize,
}

impl<T: WithDTypeF, B: Backend> EuclideanCodebook<T, B> {
    pub fn load(vb: &Path<B>, dim: usize, codebook_size: usize, epsilon: f64) -> Result<Self> {
        let cluster_usage = vb.tensor("cluster_usage", (codebook_size,))?;
        let embedding_sum = vb.tensor("embedding_sum", (codebook_size, dim))?;

        // embedding = embedding_sum / max(cluster_usage, epsilon)
        let epsilon_t =
            Tensor::full(T::from_f32(epsilon as f32), (codebook_size,), cluster_usage.device())?;
        let cluster_usage = cluster_usage.maximum(&epsilon_t)?;
        let cluster_usage = cluster_usage.unsqueeze(1)?;
        let embedding = embedding_sum.broadcast_div(&cluster_usage)?;

        // Precompute c2 = (embedding * embedding).sum(dim=-1) / 2.0
        // This is used for efficient distance computation: dist = c2 - dot_prod
        let c2 = embedding.sqr()?.sum_keepdim(vec![1])?.scale(T::from_f32(0.5))?;
        let c2 = c2.reshape((codebook_size,))?;

        Ok(Self { embedding, c2, dim })
    }

    #[tracing::instrument(name = "ec-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        // Save target shape (all dims except the last)
        let mut target_shape: Vec<usize> = xs.dims().to_vec();
        target_shape.pop();

        // Flatten to 2D: [*, dim] -> [N, dim]
        let xs = xs.flatten(0, xs.rank().saturating_sub(2))?;

        // Compute distances using precomputed c2:
        // ||x - e||^2 / 2 = ||e||^2/2 - x*e^T + ||x||^2/2
        // We only need relative distances, so: dist = c2 - dot_prod
        let dot_prod = xs.matmul_t(&self.embedding)?; // [N, codebook_size]
        let dists = self.c2.broadcast_sub(&dot_prod)?; // [N, codebook_size]

        // Argmin to get indices, then reshape to target_shape
        let codes = dists.argmin(1)?; // [N]
        if target_shape.is_empty() { Ok(codes) } else { codes.reshape(target_shape) }
    }

    #[tracing::instrument(name = "ec-decode", skip_all)]
    pub fn decode(&self, indices: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        // Save final dims: indices.dims() + [dim]
        let mut final_dims = indices.dims().to_vec();
        final_dims.push(self.dim);

        // Flatten indices
        let flat_indices = indices.flatten(0, indices.rank().saturating_sub(1))?;

        // Convert i64 indices to u32 and index into embedding
        let indices_vec = flat_indices.to_vec()?;
        let indices_u32: Vec<u32> = indices_vec.iter().map(|&x| x as u32).collect();
        let values = self.embedding.index_select(&indices_u32, 0)?;

        // Reshape to final_dims
        values.reshape(final_dims)
    }
}

/// Vector quantization layer.
pub struct VectorQuantization<T: WithDTypeF, B: Backend> {
    project_in: Option<Tensor<T, B>>,
    project_out: Option<Tensor<T, B>>,
    codebook: EuclideanCodebook<T, B>,
}

impl<T: WithDTypeF, B: Backend> VectorQuantization<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
    ) -> Result<Self> {
        let codebook_dim = codebook_dim.unwrap_or(dim);
        let (project_in, project_out) = if codebook_dim == dim {
            (None, None)
        } else {
            let p_in = vb.pp("project_in").tensor("weight", (codebook_dim, dim))?;
            let p_out = vb.pp("project_out").tensor("weight", (dim, codebook_dim))?;
            (Some(p_in), Some(p_out))
        };
        let codebook =
            EuclideanCodebook::load(&vb.pp("_codebook"), codebook_dim, codebook_size, 1e-5)?;
        Ok(Self { project_in, project_out, codebook })
    }

    #[tracing::instrument(name = "vq-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let xs = xs.t()?.contiguous()?; // [B, C, T] -> [B, T, C]
        let xs = match &self.project_in {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };
        self.codebook.encode(&xs)
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let quantized = self.codebook.decode(codes)?;
        let quantized = match &self.project_out {
            Some(proj) => quantized.matmul_t(proj)?,
            None => quantized,
        };
        quantized.t()?.contiguous()
    }
}

/// Residual vector quantization.
pub struct ResidualVectorQuantization<T: WithDTypeF, B: Backend> {
    layers: Vec<VectorQuantization<T, B>>,
}

impl<T: WithDTypeF, B: Backend> ResidualVectorQuantization<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_q: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
    ) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let layer =
                VectorQuantization::load(&layers_vb.pp(i), dim, codebook_size, codebook_dim)?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let mut codes = Vec::with_capacity(self.layers.len());
        let mut residual = xs.clone();
        for layer in &self.layers {
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            residual = residual.sub(&quantized)?;
            codes.push(indices);
        }
        // Stack codes: [n_q, B, T]
        let codes_refs: Vec<&Tensor<i64, B>> = codes.iter().collect();
        Tensor::stack(&codes_refs, 0)
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        if self.layers.is_empty() {
            crate::bail!("empty layers in ResidualVectorQuantization");
        }

        let inner_shape: Vec<usize> = codes.dims()[1..].to_vec();
        let mut quantized = self.layers[0]
            .decode(&codes.narrow(0, ..1)?.contiguous()?.reshape(inner_shape.clone())?)?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            let layer_codes =
                codes.narrow(0, i..i + 1)?.contiguous()?.reshape(inner_shape.clone())?;
            quantized = quantized.add(&layer.decode(&layer_codes)?)?;
        }
        Ok(quantized)
    }
}

/// Residual vector quantizer with input/output projections.
pub struct ResidualVectorQuantizer<T: WithDTypeF, B: Backend> {
    vq: ResidualVectorQuantization<T, B>,
    input_proj: Option<Tensor<T, B>>,
    output_proj: Option<Tensor<T, B>>,
}

impl<T: WithDTypeF, B: Backend> ResidualVectorQuantizer<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        force_projection: bool,
    ) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let output_dim = output_dim.unwrap_or(dim);

        let input_proj = if input_dim != dim || force_projection {
            Some(vb.pp("input_proj").tensor("weight", (dim, input_dim, 1))?)
        } else {
            None
        };

        let output_proj = if output_dim != dim || force_projection {
            Some(vb.pp("output_proj").tensor("weight", (output_dim, dim, 1))?)
        } else {
            None
        };

        let vq = ResidualVectorQuantization::load(&vb.pp("vq"), n_q, dim, bins, None)?;
        Ok(Self { vq, input_proj, output_proj })
    }

    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        // xs: [B, C, T]
        let xs = match &self.input_proj {
            Some(proj) => {
                // proj is stored as [out_dim, in_dim, 1] - conv1d weight format
                xs.conv1d(proj, None, 1, 0, 1, 1)?
            }
            None => xs.clone(),
        };
        let codes = self.vq.encode(&xs)?;
        codes.transpose(0, 1)?.contiguous() // [n_q, B, T] -> [B, n_q, T]
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let codes = codes.transpose(0, 1)?.contiguous()?; // [B, n_q, T] -> [n_q, B, T]
        let quantized = self.vq.decode(&codes)?;
        match &self.output_proj {
            Some(proj) => {
                // proj is stored as [out_dim, in_dim, 1] - conv1d weight format
                quantized.conv1d(proj, None, 1, 0, 1, 1)
            }
            None => Ok(quantized),
        }
    }
}

/// Split residual vector quantizer (semantic + acoustic).
pub struct SplitResidualVectorQuantizer<T: WithDTypeF, B: Backend> {
    rvq_first: ResidualVectorQuantizer<T, B>,
    rvq_rest: ResidualVectorQuantizer<T, B>,
    n_q: usize,
}

impl<T: WithDTypeF, B: Backend> SplitResidualVectorQuantizer<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
    ) -> Result<Self> {
        let rvq_first = ResidualVectorQuantizer::load(
            &vb.pp("rvq_first"),
            dim,
            input_dim,
            output_dim,
            1,
            bins,
            true,
        )?;
        let rvq_rest = ResidualVectorQuantizer::load(
            &vb.pp("rvq_rest"),
            dim,
            input_dim,
            output_dim,
            n_q - 1,
            bins,
            true,
        )?;
        Ok(Self { rvq_first, rvq_rest, n_q })
    }

    #[tracing::instrument(name = "rvq-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let codes = self.rvq_first.encode(xs)?;
        if self.n_q > 1 {
            // Encode again (not residual - semantic + acoustic split)
            let rest_codes = self.rvq_rest.encode(xs)?;
            Tensor::cat(&[&codes, &rest_codes], 1)
        } else {
            Ok(codes)
        }
    }

    #[tracing::instrument(name = "rvq-decode", skip_all)]
    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let first_codes = codes.narrow(1, ..1)?.contiguous()?;
        let quantized = self.rvq_first.decode(&first_codes)?;
        if self.n_q > 1 {
            let rest_codes = codes.narrow(1, 1..self.n_q)?.contiguous()?;
            quantized.add(&self.rvq_rest.decode(&rest_codes)?)
        } else {
            Ok(quantized)
        }
    }
}

// ============================================================================
// Mimi Model
// ============================================================================

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

/// Mimi configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: SeaNetConfig,
    pub transformer: TransformerConfig,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    /// Default configuration for Mimi v0.1.
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        Self::v0_1_inner(num_codebooks, Norm::WeightNorm)
    }

    /// Configuration for Mimi v0.1 without weight normalization.
    /// Use this for models that have regular conv weights instead of weight_g/weight_v.
    pub fn v0_1_no_weight_norm(num_codebooks: Option<usize>) -> Self {
        Self::v0_1_inner(num_codebooks, Norm::TimeGroupNorm)
    }

    fn v0_1_inner(num_codebooks: Option<usize>, norm: Norm) -> Self {
        let seanet_cfg = SeaNetConfig {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: 0,
            norm,
            pad_mode: PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = TransformerConfig {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            conv_kernel_size: 5,
            use_conv_bias: true,
            use_conv_block: false,
            max_period: 10000,
            gating: None,
            norm: NormType::LayerNorm,
            positional_embedding: PositionalEmbedding::Rope,
            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true,
            max_seq_len: 8192,
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

/// Mimi audio tokenizer model.
pub struct Mimi<T: WithDTypeF, B: Backend> {
    encoder: SeaNetEncoder<T, B>,
    decoder: SeaNetDecoder<T, B>,
    encoder_transformer: Transformer<T, B>,
    decoder_transformer: Transformer<T, B>,
    downsample: ConvDownsample1d<T, B>,
    upsample: ConvTrUpsample1d<T, B>,
    quantizer: SplitResidualVectorQuantizer<T, B>,
    config: Config,
}

impl<T: WithDTypeF, B: Backend> Mimi<T, B> {
    /// Load a Mimi model from weights.
    pub fn load(vb: &Path<B>, cfg: Config, device: &B) -> Result<Self> {
        let dim = cfg.seanet.dimension;

        let encoder = SeaNetEncoder::load(&vb.pp("encoder"), &cfg.seanet)?;
        let decoder = SeaNetDecoder::load(&vb.pp("decoder"), &cfg.seanet)?;

        let encoder_transformer =
            Transformer::load(&vb.pp("encoder_transformer"), dim, &cfg.transformer, device)?;
        let decoder_transformer =
            Transformer::load(&vb.pp("decoder_transformer"), dim, &cfg.transformer, device)?;

        let quantizer = SplitResidualVectorQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dim,
            Some(dim),
            Some(dim),
            cfg.quantizer_n_q,
            cfg.quantizer_bins,
        )?;

        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;

        let downsample =
            ConvDownsample1d::load(&vb.pp("downsample"), downsample_stride, dim, true)?;
        let upsample = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, dim, true)?;

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            config: cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Encode audio to codes (non-streaming).
    pub fn encode(&mut self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let xs = self.encoder.forward(xs)?;
        self.encoder_transformer.reset_state();
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = &xs[0];
        let xs = self.downsample.forward(xs)?;
        self.quantizer.encode(&xs)
    }

    /// Decode codes to audio (non-streaming).
    pub fn decode(&mut self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let emb = self.quantizer.decode(codes)?;
        let emb = self.upsample.forward(&emb)?;
        self.decoder_transformer.reset_state();
        let outs = self.decoder_transformer.forward(&emb)?;
        self.decoder.forward(&outs[0])
    }

    /// Encode audio step (streaming).
    pub fn encode_step(
        &mut self,
        xs: &StreamTensor<T, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<i64, B>> {
        let xs = self.encoder.step(xs, mask)?;
        let xs = self.encoder_transformer.step(&xs, mask)?;
        let xs = self.downsample.step(&xs, mask)?;
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.quantizer.encode(xs)?)),
        }
    }

    /// Decode codes step (streaming).
    pub fn decode_step(
        &mut self,
        codes: &StreamTensor<i64, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let emb: StreamTensor<T, B> = match codes.as_option() {
            Some(codes) => StreamTensor::from_tensor(self.quantizer.decode(codes)?),
            None => StreamTensor::empty(),
        };
        let emb = self.upsample.step(&emb, mask)?;
        let out = self.decoder_transformer.step(&emb, mask)?;
        self.decoder.step(&out, mask)
    }

    /// Reset all streaming state.
    pub fn reset_state(&mut self) {
        self.encoder.reset_state();
        self.decoder.reset_state();
        self.encoder_transformer.reset_state();
        self.decoder_transformer.reset_state();
        self.downsample.reset_state();
        self.upsample.reset_state();
    }
}
