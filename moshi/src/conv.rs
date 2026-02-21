use crate::streaming::{StreamMask, StreamTensor};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Norm {
    WeightNorm,
    SpectralNorm,
    TimeGroupNorm,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

// ============================================================================
// Streaming State Types
// ============================================================================

pub struct Conv1dState<T: WithDTypeF, B: Backend> {
    pub prev_xs: Option<Tensor<T, B>>,
    pub left_pad_applied: bool,
}

pub struct ConvTr1dState<T: WithDTypeF, B: Backend> {
    pub prev_ys: Option<Tensor<T, B>>,
}

// ============================================================================
// NormConv1d
// ============================================================================

pub struct NormConv1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Option<Tensor<T, B>>,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl<T: WithDTypeF, B: Backend> NormConv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        _causal: bool,
        norm: Option<Norm>,
        bias: bool,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let vb = vb.pp("conv").pp("conv");
        let weight = match norm {
            Some(Norm::WeightNorm) => {
                if vb.contains("weight") {
                    vb.tensor("weight", (out_c, in_c / groups, k_size))?
                } else {
                    let weight_g: Tensor<T, B> = vb.tensor("weight_g", (out_c, 1, 1))?;
                    let weight_v: Tensor<T, B> =
                        vb.tensor("weight_v", (out_c, in_c / groups, k_size))?;
                    let norm_v = weight_v.sqr()?.sum_keepdim(vec![1, 2])?.sqrt()?;
                    weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?
                }
            }
            Some(Norm::SpectralNorm) => {
                xn::bail!("SpectralNorm is not supported yet")
            }
            Some(Norm::TimeGroupNorm) => {
                xn::bail!("TimeGroupNorm requires GroupNorm which is not available in xn")
            }
            None => vb.tensor("weight", (out_c, in_c / groups, k_size))?,
        };
        let bias = if bias {
            Some(vb.tensor("bias", (out_c,))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            dilation,
            groups,
        })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
    }

    pub fn kernel_size(&self) -> usize {
        self.weight.dims()[2]
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn dilation(&self) -> usize {
        self.dilation
    }
}

// ============================================================================
// NormConvTranspose1d
// ============================================================================

pub struct NormConvTranspose1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    pub(crate) bias: Option<Tensor<T, B>>,
    pub(crate) k_size: usize,
    pub(crate) stride: usize,
    groups: usize,
}

impl<T: WithDTypeF, B: Backend> NormConvTranspose1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        _causal: bool,
        norm: Option<Norm>,
        bias: bool,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let vb = vb.pp("convtr").pp("convtr");
        let weight = match norm {
            Some(Norm::WeightNorm) => {
                if vb.contains("weight") {
                    vb.tensor("weight", (in_c, out_c / groups, k_size))?
                } else {
                    let weight_g: Tensor<T, B> = vb.tensor("weight_g", (in_c, 1, 1))?;
                    let weight_v: Tensor<T, B> =
                        vb.tensor("weight_v", (in_c, out_c / groups, k_size))?;
                    let norm_v = weight_v.sqr()?.sum_keepdim(vec![1, 2])?.sqrt()?;
                    weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?
                }
            }
            Some(Norm::SpectralNorm) => {
                xn::bail!("SpectralNorm is not supported yet")
            }
            Some(Norm::TimeGroupNorm) => {
                xn::bail!("TimeGroupNorm requires GroupNorm which is not available in xn")
            }
            None => vb.tensor("weight", (in_c, out_c / groups, k_size))?,
        };
        let bias = if bias {
            Some(vb.tensor("bias", (out_c,))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            k_size,
            stride,
            groups,
        })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.conv_transpose1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            0,
            0,
            self.groups,
        )
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn get_extra_padding_for_conv1d<T: WithDTypeF, B: Backend>(
    xs: &Tensor<T, B>,
    k_size: usize,
    stride: usize,
    padding_total: usize,
) -> Result<usize> {
    let len = xs.dim(2usize)?;
    let n_frames = (len + padding_total).saturating_sub(k_size) as f64 / stride as f64 + 1.0;
    let ideal_len =
        ((n_frames.ceil() as usize - 1) * stride + k_size).saturating_sub(padding_total);
    Ok(ideal_len.saturating_sub(len))
}

fn pad1d<T: WithDTypeF, B: Backend>(
    xs: &Tensor<T, B>,
    pad_l: usize,
    pad_r: usize,
    mode: PadMode,
) -> Result<Tensor<T, B>> {
    match mode {
        PadMode::Constant => xs.pad_with_zeros(2usize, pad_l, pad_r),
        PadMode::Reflect => xn::bail!("pad-mode 'reflect' is not supported"),
        PadMode::Replicate => xs.pad_with_same(2usize, pad_l, pad_r),
    }
}

fn unpad1d<T: WithDTypeF, B: Backend>(
    xs: &Tensor<T, B>,
    unpad_l: usize,
    unpad_r: usize,
) -> Result<Tensor<T, B>> {
    let len = xs.dim(2usize)?;
    if len < unpad_l + unpad_r {
        xn::bail!("unpad1d: tensor len {len} is too low, {unpad_l} + {unpad_r}");
    }
    xs.narrow(2, unpad_l..len - unpad_r)?.contiguous()
}

// ============================================================================
// StreamableConv1d
// ============================================================================

pub struct StreamableConv1d<T: WithDTypeF, B: Backend> {
    conv: NormConv1d<T, B>,
    causal: bool,
    pad_mode: PadMode,
    kernel_size: usize,
}

impl<T: WithDTypeF, B: Backend> StreamableConv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        norm: Option<Norm>,
        pad_mode: PadMode,
    ) -> Result<Self> {
        let conv = NormConv1d::load(
            vb, in_c, out_c, k_size, causal, norm, bias, stride, dilation, groups,
        )?;
        if k_size < stride {
            xn::bail!("kernel-size {k_size} is smaller than stride {stride}");
        }
        Ok(Self {
            conv,
            causal,
            pad_mode,
            kernel_size: k_size,
        })
    }

    pub fn init_state(&self) -> Conv1dState<T, B> {
        Conv1dState {
            prev_xs: None,
            left_pad_applied: false,
        }
    }

    fn padding_total(&self) -> usize {
        let k_size = (self.kernel_size - 1) * self.conv.dilation() + 1;
        k_size - self.conv.stride()
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let k_size = self.conv.kernel_size();
        let stride = self.conv.stride();
        let dilation = self.conv.dilation();
        let k_size_eff = (k_size - 1) * dilation + 1;
        let padding_total = k_size_eff - stride;
        let extra_padding = get_extra_padding_for_conv1d(xs, k_size_eff, stride, padding_total)?;
        let xs = if self.causal {
            pad1d(xs, padding_total, extra_padding, self.pad_mode)?
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            pad1d(
                xs,
                padding_left,
                padding_right + extra_padding,
                self.pad_mode,
            )?
        };
        self.conv.forward(&xs)
    }

    #[tracing::instrument(name = "streamable-conv1d", skip_all)]
    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut Conv1dState<T, B>,
        _mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs.clone(),
        };
        let xs = if state.left_pad_applied {
            xs
        } else {
            state.left_pad_applied = true;
            pad1d(&xs, self.padding_total(), 0, self.pad_mode)?
        };

        let stride = self.conv.stride();
        let dilation = self.conv.dilation();
        let kernel = (self.kernel_size - 1) * dilation + 1;

        let xs = match &state.prev_xs {
            None => xs,
            Some(prev) => Tensor::cat(&[prev, &xs], 2)?,
        };

        let seq_len = xs.dim(2usize)?;
        let num_frames = (seq_len + stride).saturating_sub(kernel) / stride;

        if num_frames > 0 {
            let offset = num_frames * stride;
            if seq_len > offset {
                state.prev_xs = Some(xs.narrow(2, offset..seq_len)?.contiguous()?);
            } else {
                state.prev_xs = None;
            }
            let in_l = (num_frames - 1) * stride + kernel;
            let xs = xs.narrow(2, ..in_l)?.contiguous()?;
            Ok(StreamTensor::from_tensor(self.conv.forward(&xs)?))
        } else {
            state.prev_xs = Some(xs);
            Ok(StreamTensor::empty())
        }
    }
}

// ============================================================================
// StreamableConvTranspose1d
// ============================================================================

pub struct StreamableConvTranspose1d<T: WithDTypeF, B: Backend> {
    pub(crate) convtr: NormConvTranspose1d<T, B>,
    causal: bool,
    kernel_size: usize,
}

impl<T: WithDTypeF, B: Backend> StreamableConvTranspose1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_c: usize,
        out_c: usize,
        k_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        norm: Option<Norm>,
    ) -> Result<Self> {
        let convtr =
            NormConvTranspose1d::load(vb, in_c, out_c, k_size, causal, norm, bias, stride, groups)?;
        Ok(Self {
            convtr,
            causal,
            kernel_size: k_size,
        })
    }

    pub fn init_state(&self) -> ConvTr1dState<T, B> {
        ConvTr1dState { prev_ys: None }
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let padding_total = self.convtr.k_size.saturating_sub(self.convtr.stride);
        let xs = self.convtr.forward(xs)?;
        if self.causal {
            unpad1d(&xs, 0, padding_total)
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            unpad1d(&xs, padding_left, padding_right)
        }
    }

    #[tracing::instrument(name = "streamable-convtr1d", skip_all)]
    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut ConvTr1dState<T, B>,
        _mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let xs = match xs.as_option() {
            Some(xs) => xs,
            None => return Ok(StreamTensor::empty()),
        };
        let stride = self.convtr.stride;

        let ys = self.convtr.forward(xs)?;
        let ot = ys.dim(2usize)?;

        let ys = match &state.prev_ys {
            None => ys,
            Some(prev_ys) => {
                let pt = prev_ys.dim(2usize)?;
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

        let invalid_steps = self.kernel_size - stride;
        let valid_len = ot.saturating_sub(invalid_steps);
        if valid_len > 0 {
            let valid = ys.narrow(2, ..valid_len)?.contiguous()?;
            if ot > valid_len {
                state.prev_ys = Some(ys.narrow(2, valid_len..ot)?.contiguous()?);
            } else {
                state.prev_ys = None;
            }
            Ok(StreamTensor::from_tensor(valid))
        } else {
            state.prev_ys = Some(ys);
            Ok(StreamTensor::empty())
        }
    }
}

// ============================================================================
// ConvDownsample1d / ConvTrUpsample1d
// ============================================================================

pub struct ConvDownsample1d<T: WithDTypeF, B: Backend> {
    conv: StreamableConv1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvDownsample1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        stride: usize,
        dim: usize,
        causal: bool,
        _learnt: bool,
    ) -> Result<Self> {
        let conv = StreamableConv1d::load(
            &vb.pp("conv"),
            dim,
            dim,
            /* k_size */ 2 * stride,
            /* stride */ stride,
            /* dilation */ 1,
            /* groups */ 1,
            /* bias */ false,
            /* causal */ causal,
            /* norm */ None,
            /* pad_mode */ PadMode::Replicate,
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(&self) -> Conv1dState<T, B> {
        self.conv.init_state()
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.conv.forward(xs)
    }

    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut Conv1dState<T, B>,
        m: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        self.conv.step(xs, state, m)
    }
}

pub struct ConvTrUpsample1d<T: WithDTypeF, B: Backend> {
    convtr: StreamableConvTranspose1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> ConvTrUpsample1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        stride: usize,
        dim: usize,
        causal: bool,
        _learnt: bool,
    ) -> Result<Self> {
        let convtr = StreamableConvTranspose1d::load(
            &vb.pp("convtr"),
            dim,
            dim,
            /* k_size */ 2 * stride,
            /* stride */ stride,
            /* groups */ dim, // depthwise
            /* bias */ false,
            /* causal */ causal,
            /* norm */ None,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self) -> ConvTr1dState<T, B> {
        self.convtr.init_state()
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        self.convtr.forward(xs)
    }

    pub fn step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut ConvTr1dState<T, B>,
        m: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        self.convtr.step(xs, state, m)
    }
}
