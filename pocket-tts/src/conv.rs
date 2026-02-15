use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

/// Pad input so that a convolution covers the full input.
pub fn pad_for_conv1d<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
) -> Result<Tensor<T, B>> {
    let length = x.dim(2usize)?;
    let n_frames = (length as f64 - kernel_size as f64) / stride as f64 + 1.0;
    let ideal_length = (n_frames.ceil() as usize - 1) * stride + kernel_size;
    let extra = ideal_length.saturating_sub(length);
    if extra > 0 { x.pad_with_zeros(2usize, 0, extra) } else { Ok(x.clone()) }
}

/// Conv1d wrapper with weight and optional bias.
pub struct Conv1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    bias: Option<Tensor<T, B>>,
    pub stride: usize,
    pub dilation: usize,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl<T: WithDTypeF, B: Backend> Conv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (out_channels, in_channels / groups, kernel_size))?;
        let bias = if bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, dilation, kernel_size, in_channels, out_channels, groups })
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.conv1d(&self.weight, self.bias.as_ref(), self.stride, 0, self.dilation, self.groups)
    }
}

/// ConvTranspose1d wrapper.
pub struct ConvTranspose1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    pub bias: Option<Tensor<T, B>>,
    pub stride: usize,
    pub kernel_size: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl<T: WithDTypeF, B: Backend> ConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (in_channels, out_channels / groups, kernel_size))?;
        let bias = if bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, kernel_size, out_channels, groups })
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.conv_transpose1d(&self.weight, self.bias.as_ref(), self.stride, 0, 0, self.groups)
    }
}

/// Streaming state for StreamingConv1d.
pub struct StreamingConv1dState<T: WithDTypeF, B: Backend> {
    pub previous: Tensor<T, B>,
    pub first: bool,
}

/// Streaming Conv1d with causal padding.
pub struct StreamingConv1d<T: WithDTypeF, B: Backend> {
    pub conv: Conv1d<T, B>,
    pad_mode: PadMode,
}

#[derive(Clone, Copy)]
pub enum PadMode {
    Constant,
    Replicate,
}

impl<T: WithDTypeF, B: Backend> StreamingConv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = Conv1d::load(
            &vb.pp("conv"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
        )?;
        Ok(Self { conv, pad_mode })
    }

    fn effective_kernel_size(&self) -> usize {
        (self.conv.kernel_size - 1) * self.conv.dilation + 1
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConv1dState<T, B>> {
        let kernel = self.effective_kernel_size();
        let prev_len = kernel - self.conv.stride;
        let previous = Tensor::zeros(
            (batch_size, self.conv.in_channels, prev_len),
            self.conv.weight.device(),
        )?;
        Ok(StreamingConv1dState { previous, first: true })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConv1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let tp = state.previous.dim(2usize)?;

        // On first call with replicate padding, fill previous with first sample
        if tp > 0 && matches!(self.pad_mode, PadMode::Replicate) && state.first {
            let init = x.narrow(2, 0..1)?.contiguous()?;
            // Broadcast init to fill previous
            let mut fills = Vec::with_capacity(tp);
            for _ in 0..tp {
                fills.push(&init);
            }
            // Actually just repeat using cat
            state.previous = if tp == 1 {
                init
            } else {
                let refs: Vec<&Tensor<T, B>> = (0..tp).map(|_| &init).collect();
                Tensor::cat(&refs, 2)?
            };
        }

        // Prepend previous state
        let x = if tp > 0 { Tensor::cat(&[&state.previous, x], 2)? } else { x.clone() };

        // Run convolution
        let y = self.conv.forward(&x)?;

        // Update state
        if tp > 0 {
            let xlen = x.dim(2usize)?;
            state.previous = x.narrow(2, xlen - tp..xlen)?.contiguous()?;
            if matches!(self.pad_mode, PadMode::Replicate) {
                state.first = false;
            }
        }

        Ok(y)
    }
}

/// Streaming state for StreamingConvTranspose1d.
pub struct StreamingConvTr1dState<T: WithDTypeF, B: Backend> {
    pub partial: Tensor<T, B>,
}

/// Streaming ConvTranspose1d.
pub struct StreamingConvTranspose1d<T: WithDTypeF, B: Backend> {
    pub convtr: ConvTranspose1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> StreamingConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let convtr = ConvTranspose1d::load(
            &vb.pp("convtr"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConvTr1dState<T, B>> {
        let pt = self.convtr.kernel_size - self.convtr.stride;
        let partial =
            Tensor::zeros((batch_size, self.convtr.out_channels, pt), self.convtr.weight.device())?;
        Ok(StreamingConvTr1dState { partial })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConvTr1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut y = self.convtr.forward(x)?;
        let pt = state.partial.dim(2usize)?;

        if pt > 0 {
            // Add overlap from previous
            let y_start = y.narrow(2, 0..pt)?.contiguous()?;
            let y_start = y_start.add(&state.partial)?;
            y.slice_set(&y_start, 2usize, 0)?;

            // Save new partial (subtract bias if present)
            let y_len = y.dim(2usize)?;
            let mut for_partial = y.narrow(2, y_len - pt..y_len)?.contiguous()?;
            if let Some(bias) = &self.convtr.bias {
                let bias = bias.reshape((1, bias.elem_count(), 1))?;
                for_partial = for_partial.broadcast_sub(&bias)?;
            }
            state.partial = for_partial;

            // Return without the partial tail
            y = y.narrow(2, 0..y_len - pt)?.contiguous()?;
        }

        Ok(y)
    }
}
