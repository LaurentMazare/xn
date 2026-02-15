#![allow(clippy::too_many_arguments)]

use crate::error::Context;
use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct Config {
    pub chin: usize,
    pub chout: usize,
    pub hidden: usize,
    pub depth: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub causal: bool,
    pub resample: usize,
    pub growth: f32,
    pub max_hidden: usize,
    pub normalize: bool,
    pub glu: bool,
    pub floor: f32,
    pub bias: bool,
    pub sample_rate: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chin: 2,
            chout: 1,
            hidden: 48,
            depth: 5,
            kernel_size: 8,
            stride: 4,
            causal: true,
            resample: 4,
            growth: 2.0,
            max_hidden: 10_000,
            normalize: true,
            glu: true,
            floor: 1e-3,
            bias: true,
            sample_rate: 16_000,
        }
    }
}

impl Config {
    pub fn total_stride(&self) -> usize {
        self.stride.pow(self.depth as u32) / self.resample
    }

    /// Matches Python: math.ceil(length * resample) then encode/decode loops
    pub fn valid_length(&self, length: usize) -> usize {
        // Python: length = math.ceil(length * self.resample)
        let mut length = (length * self.resample).div_ceil(self.resample);
        length *= self.resample;

        // Encoder passes
        for _ in 0..self.depth {
            // Python: length = math.ceil((length - self.kernel_size) / self.stride) + 1
            let num = length.saturating_sub(self.kernel_size);
            length = num.div_ceil(self.stride) + 1;
            length = length.max(1);
        }

        // Decoder passes
        for _ in 0..self.depth {
            length = (length - 1) * self.stride + self.kernel_size;
        }

        // Python: int(math.ceil(length / self.resample))
        length.div_ceil(self.resample)
    }
}

// ============================================================================
// Sinc resampling
// ============================================================================

#[tracing::instrument(skip_all)]
fn hann_window<T: WithDTypeF, B: Backend>(size: usize, device: &B) -> Result<Tensor<T, B>> {
    let mut data = Vec::with_capacity(size);
    let pi = std::f32::consts::PI;
    for i in 0..size {
        // periodic=False means symmetric window
        let v = 0.5 * (1.0 - (2.0 * pi * i as f32 / (size - 1) as f32).cos());
        data.push(T::from_f32(v));
    }
    Tensor::from_vec(data, (size,), device)
}

/// sinc(t) = sin(t)/t for t != 0, 1 for t == 0
fn compute_sinc(t: f32) -> f32 {
    if t.abs() < 1e-10 { 1.0 } else { t.sin() / t }
}

#[tracing::instrument(skip_all)]
pub fn kernel_upsample2<T: WithDTypeF, B: Backend>(
    zeros: usize,
    device: &B,
) -> Result<Tensor<T, B>> {
    let win = hann_window::<T, B>(4 * zeros + 1, device)?;
    let win_vec = win.to_vec()?;
    // winodd = win[1::2]
    let winodd: Vec<T> = win_vec.iter().skip(1).step_by(2).copied().collect();

    // t = linspace(-zeros + 0.5, zeros - 0.5, 2*zeros) * pi
    let pi = std::f32::consts::PI;
    let mut kernel_data = Vec::with_capacity(2 * zeros);
    for (i, winodd) in winodd.iter().enumerate().take(2 * zeros) {
        let t = (-(zeros as f32) + 0.5 + i as f32) * pi;
        let sinc_val = compute_sinc(t);
        let v = sinc_val * WithDTypeF::to_f32(*winodd);
        kernel_data.push(T::from_f32(v));
    }

    Tensor::from_vec(kernel_data, (1, 1, 2 * zeros), device)
}

pub fn kernel_downsample2<T: WithDTypeF, B: Backend>(
    zeros: usize,
    device: &B,
) -> Result<Tensor<T, B>> {
    kernel_upsample2(zeros, device)
}

#[tracing::instrument(skip_all)]
pub fn upsample2<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    zeros: usize,
) -> Result<Tensor<T, B>> {
    let dims = x.dims();
    let time = dims[dims.len() - 1];
    let kernel = kernel_upsample2::<T, B>(zeros, x.device())?;

    // Reshape to (-1, 1, time)
    let batch_channels: usize = dims[..dims.len() - 1].iter().product();
    let x_flat = x.reshape((batch_channels, 1, time))?;

    // conv1d with padding=zeros, then [1:] to remove first element
    let out = x_flat.conv1d(&kernel, None, 1, zeros, 1, 1)?;
    let out_len = out.dim(2)?;
    let out = out.narrow(2, 1..out_len)?.contiguous()?;
    let out = out.reshape([&dims[..dims.len() - 1], &[time]].concat())?;

    // Interleave: stack([x, out], dim=-1).view(*other, -1)
    let x_unsq = x.unsqueeze(dims.len())?;
    let out_unsq = out.unsqueeze(dims.len())?;
    let y = Tensor::cat(&[&x_unsq, &out_unsq], dims.len())?;
    let mut new_dims = dims.to_vec();
    new_dims[dims.len() - 1] = time * 2;
    y.reshape(new_dims)
}

#[tracing::instrument(skip_all)]
pub fn downsample2<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    zeros: usize,
) -> Result<Tensor<T, B>> {
    let dims = x.dims();
    let mut time = dims[dims.len() - 1];

    // Pad if odd
    let x = if !time.is_multiple_of(2) {
        time += 1;
        x.pad_with_zeros(dims.len() - 1, 0, 1)?
    } else {
        x.clone()
    };

    // Extract even and odd samples
    let x_vec = x.to_vec()?;
    let inner_size: usize = dims[..dims.len() - 1].iter().product();
    let half_time = time / 2;

    let mut xeven_data = Vec::with_capacity(inner_size * half_time);
    let mut xodd_data = Vec::with_capacity(inner_size * half_time);

    for batch_idx in 0..inner_size {
        let offset = batch_idx * time;
        for t in 0..half_time {
            xeven_data.push(x_vec[offset + t * 2]);
            xodd_data.push(x_vec[offset + t * 2 + 1]);
        }
    }

    let mut new_dims = dims.to_vec();
    new_dims[dims.len() - 1] = half_time;

    let xeven = Tensor::from_vec(xeven_data, new_dims.clone(), x.device())?;
    let xodd = Tensor::from_vec(xodd_data, new_dims.clone(), x.device())?;

    let kernel = kernel_downsample2::<T, B>(zeros, x.device())?;

    // conv1d on xodd, then [:-1]
    let batch_channels: usize = new_dims[..new_dims.len() - 1].iter().product();
    let xodd_flat = xodd.reshape((batch_channels, 1, half_time))?;
    let conv_out = xodd_flat.conv1d(&kernel, None, 1, zeros, 1, 1)?;
    let conv_len = conv_out.dim(2)?;
    let conv_out = conv_out.narrow(2, ..conv_len - 1)?.contiguous()?;
    let conv_out = conv_out.reshape(new_dims)?;

    // out = (xeven + conv_out) * 0.5
    let sum = xeven.add(&conv_out)?;
    sum.scale(T::from_f32(0.5))
}

// ============================================================================
// LSTM
// ============================================================================

pub struct LstmCell<T: WithDTypeF, B: Backend> {
    weight_ih: Tensor<T, B>,
    weight_hh: Tensor<T, B>,
    sum_bias: Tensor<T, B>,
    hidden_size: usize,
}

impl<T: WithDTypeF, B: Backend> LstmCell<T, B> {
    pub fn load_layer(
        vb: &Path<B>,
        layer: usize,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let weight_ih = vb.tensor(&format!("weight_ih_l{layer}"), (4 * hidden_size, input_size))?;
        let weight_hh =
            vb.tensor(&format!("weight_hh_l{layer}"), (4 * hidden_size, hidden_size))?;
        let bias_ih = vb.tensor(&format!("bias_ih_l{layer}"), (4 * hidden_size,))?;
        let bias_hh = vb.tensor(&format!("bias_hh_l{layer}"), (4 * hidden_size,))?;
        let sum_bias = bias_ih.add(&bias_hh)?.reshape((1, 4 * hidden_size))?;
        Ok(Self { weight_ih, weight_hh, sum_bias, hidden_size })
    }

    pub fn load_layer_reverse(
        vb: &Path<B>,
        layer: usize,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let weight_ih =
            vb.tensor(&format!("weight_ih_l{layer}_reverse"), (4 * hidden_size, input_size))?;
        let weight_hh =
            vb.tensor(&format!("weight_hh_l{layer}_reverse"), (4 * hidden_size, hidden_size))?;
        let bias_ih = vb.tensor(&format!("bias_ih_l{layer}_reverse"), (4 * hidden_size,))?;
        let bias_hh = vb.tensor(&format!("bias_hh_l{layer}_reverse"), (4 * hidden_size,))?;
        let sum_bias = bias_ih.add(&bias_hh)?.reshape((1, 4 * hidden_size))?;
        Ok(Self { weight_ih, weight_hh, sum_bias, hidden_size })
    }

    /// x: (batch, input), h: (batch, hidden), c: (batch, hidden)
    #[tracing::instrument(skip_all)]
    pub fn forward_step(
        &self,
        x: &Tensor<T, B>,
        h: &Tensor<T, B>,
        c: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let gates_ih = x.matmul_t(&self.weight_ih)?;
        let gates_hh = h.matmul_t(&self.weight_hh)?;
        let gates = gates_ih.add(&gates_hh)?.broadcast_add(&self.sum_bias)?;

        let i = gates.narrow(1, ..self.hidden_size)?.sigmoid()?;
        let f = gates.narrow(1, self.hidden_size..2 * self.hidden_size)?.sigmoid()?;
        let g = gates.narrow(1, 2 * self.hidden_size..3 * self.hidden_size)?.tanh()?;
        let o = gates.narrow(1, 3 * self.hidden_size..4 * self.hidden_size)?.sigmoid()?;

        let c_new = f.mul(c)?.add(&i.mul(&g)?)?;
        let h_new = o.mul(&c_new.tanh()?)?;

        Ok((h_new, c_new))
    }
}

pub struct Lstm<T: WithDTypeF, B: Backend> {
    layers: Vec<LstmCell<T, B>>,
    hidden_size: usize,
}

impl<T: WithDTypeF, B: Backend> Lstm<T, B> {
    pub fn load(
        vb: &Path<B>,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            layers.push(LstmCell::load_layer(vb, layer, in_size, hidden_size)?);
        }
        Ok(Self { layers, hidden_size })
    }

    /// x: (seq_len, batch, input), state: Option<(h, c)> where h,c are (num_layers, batch, hidden)
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(name = "lstm_forward", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: Option<(Tensor<T, B>, Tensor<T, B>)>,
    ) -> Result<(Tensor<T, B>, (Tensor<T, B>, Tensor<T, B>))> {
        let (seq_len, batch, _input) = x.dims3()?;

        let num_layers = self.layers.len();

        let (mut h, mut c) = match state {
            Some((h, c)) => (h, c),
            None => {
                let h = Tensor::zeros((num_layers, batch, self.hidden_size), x.device())?;
                let c = Tensor::zeros((num_layers, batch, self.hidden_size), x.device())?;
                (h, c)
            }
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut x_t = x.narrow(0, t..t + 1)?.contiguous()?.reshape((batch, x.dim(2)?))?;

            let mut h_new_layers = Vec::with_capacity(num_layers);
            let mut c_new_layers = Vec::with_capacity(num_layers);

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let h_l = h
                    .narrow(0, layer_idx..layer_idx + 1)?
                    .contiguous()?
                    .reshape((batch, self.hidden_size))?;
                let c_l = c
                    .narrow(0, layer_idx..layer_idx + 1)?
                    .contiguous()?
                    .reshape((batch, self.hidden_size))?;
                let (h_new, c_new) = layer.forward_step(&x_t, &h_l, &c_l)?;
                h_new_layers.push(h_new.unsqueeze(0)?);
                c_new_layers.push(c_new.unsqueeze(0)?);
                x_t = h_new_layers
                    .last()
                    .context("no last layer")?
                    .reshape((batch, self.hidden_size))?;
            }

            let h_refs: Vec<_> = h_new_layers.iter().collect();
            let c_refs: Vec<_> = c_new_layers.iter().collect();
            h = Tensor::cat(&h_refs, 0)?;
            c = Tensor::cat(&c_refs, 0)?;

            outputs.push(x_t.unsqueeze(0)?);
        }

        let out_refs: Vec<_> = outputs.iter().collect();
        let output = Tensor::cat(&out_refs, 0)?;
        Ok((output, (h, c)))
    }
}

pub struct BiLstm<T: WithDTypeF, B: Backend> {
    forward_layers: Vec<LstmCell<T, B>>,
    backward_layers: Vec<LstmCell<T, B>>,
    hidden_size: usize,
}

impl<T: WithDTypeF, B: Backend> BiLstm<T, B> {
    pub fn load(
        vb: &Path<B>,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut forward_layers = Vec::with_capacity(num_layers);
        let mut backward_layers = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { 2 * hidden_size };
            forward_layers.push(LstmCell::load_layer(vb, layer, in_size, hidden_size)?);
            backward_layers.push(LstmCell::load_layer_reverse(vb, layer, in_size, hidden_size)?);
        }
        Ok(Self { forward_layers, backward_layers, hidden_size })
    }

    #[tracing::instrument(skip_all)]
    #[tracing::instrument(name = "bilstm_forward", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let (seq_len, batch, _input) = x.dims3()?;

        let num_layers = self.forward_layers.len();
        let mut layer_input = x.clone();

        for layer_idx in 0..num_layers {
            let input_size = layer_input.dim(2)?;

            let mut h_f = Tensor::zeros((batch, self.hidden_size), x.device())?;
            let mut c_f = Tensor::zeros((batch, self.hidden_size), x.device())?;
            let mut forward_outputs = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let x_t =
                    layer_input.narrow(0, t..t + 1)?.contiguous()?.reshape((batch, input_size))?;
                let (h_new, c_new) =
                    self.forward_layers[layer_idx].forward_step(&x_t, &h_f, &c_f)?;
                forward_outputs.push(h_new.unsqueeze(0)?);
                h_f = h_new;
                c_f = c_new;
            }

            let mut h_b = Tensor::zeros((batch, self.hidden_size), x.device())?;
            let mut c_b = Tensor::zeros((batch, self.hidden_size), x.device())?;
            let mut backward_outputs = Vec::with_capacity(seq_len);

            for t in (0..seq_len).rev() {
                let x_t =
                    layer_input.narrow(0, t..t + 1)?.contiguous()?.reshape((batch, input_size))?;
                let (h_new, c_new) =
                    self.backward_layers[layer_idx].forward_step(&x_t, &h_b, &c_b)?;
                backward_outputs.push(h_new.unsqueeze(0)?);
                h_b = h_new;
                c_b = c_new;
            }
            backward_outputs.reverse();

            let fwd_refs: Vec<_> = forward_outputs.iter().collect();
            let bwd_refs: Vec<_> = backward_outputs.iter().collect();
            let fwd_out = Tensor::cat(&fwd_refs, 0)?;
            let bwd_out = Tensor::cat(&bwd_refs, 0)?;
            layer_input = Tensor::cat(&[&fwd_out, &bwd_out], 2)?;
        }

        Ok(layer_input)
    }
}

// ============================================================================
// BLSTM wrapper (matches Python class)
// ============================================================================

pub enum BlstmInner<T: WithDTypeF, B: Backend> {
    Bidirectional { lstm: BiLstm<T, B>, linear: crate::nn::Linear<T, B> },
    Unidirectional { lstm: Lstm<T, B> },
}

pub struct Blstm<T: WithDTypeF, B: Backend> {
    inner: BlstmInner<T, B>,
    hidden_size: usize,
}

impl<T: WithDTypeF, B: Backend> Blstm<T, B> {
    pub fn load(vb: &Path<B>, dim: usize, layers: usize, bidirectional: bool) -> Result<Self> {
        let inner = if bidirectional {
            let lstm = BiLstm::load(&vb.pp("lstm"), dim, dim, layers)?;
            let linear = crate::nn::Linear::load(&vb.pp("linear"), 2 * dim, dim)?;
            BlstmInner::Bidirectional { lstm, linear }
        } else {
            let lstm = Lstm::load(&vb.pp("lstm"), dim, dim, layers)?;
            BlstmInner::Unidirectional { lstm }
        };
        Ok(Self { inner, hidden_size: dim })
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(name = "blstm_forward", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: Option<(Tensor<T, B>, Tensor<T, B>)>,
    ) -> Result<(Tensor<T, B>, Option<(Tensor<T, B>, Tensor<T, B>)>)> {
        match &self.inner {
            BlstmInner::Bidirectional { lstm, linear } => {
                let y = lstm.forward(x)?;
                let out = linear.forward(&y)?;
                Ok((out, None))
            }
            BlstmInner::Unidirectional { lstm } => {
                let (out, new_state) = lstm.forward(x, state)?;
                Ok((out, Some(new_state)))
            }
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ============================================================================
// Conv layers
// ============================================================================

pub struct Conv1d<T: WithDTypeF, B: Backend> {
    pub weight: Tensor<T, B>,
    pub bias: Option<Tensor<T, B>>,
    pub stride: usize,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_size: usize,
}

impl<T: WithDTypeF, B: Backend> Conv1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (out_channels, in_channels, kernel_size))?;
        let bias = if use_bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride, out_channels, in_channels, kernel_size })
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.conv1d(&self.weight, self.bias.as_ref(), self.stride, 0, 1, 1)
    }
}

pub struct ConvTranspose1d<T: WithDTypeF, B: Backend> {
    pub weight: Tensor<T, B>,
    pub bias: Option<Tensor<T, B>>,
    pub stride: usize,
}

impl<T: WithDTypeF, B: Backend> ConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (in_channels, out_channels, kernel_size))?;
        let bias = if use_bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, bias, stride })
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.conv_transpose1d(&self.weight, self.bias.as_ref(), self.stride, 0, 0, 1)
    }
}

// ============================================================================
// GLU activation
// ============================================================================

fn glu<T: WithDTypeF, B: Backend>(x: &Tensor<T, B>, dim: usize) -> Result<Tensor<T, B>> {
    let size = x.dim(dim)?;
    let half = size / 2;
    let a = x.narrow(dim, ..half)?.contiguous()?;
    let b = x.narrow(dim, half..)?.contiguous()?;
    a.mul(&b.sigmoid()?)
}

// ============================================================================
// Encoder/Decoder blocks matching Python structure
// ============================================================================

/// Encoder block: Conv1d -> ReLU -> Conv1d(1x1) -> GLU/ReLU
pub struct EncoderBlock<T: WithDTypeF, B: Backend> {
    pub conv0: Conv1d<T, B>, // encode[0]: Conv1d(chin, hidden, kernel_size, stride)
    pub conv2: Conv1d<T, B>, // encode[2]: Conv1d(hidden, hidden*ch_scale, 1)
    pub glu: bool,
}

impl<T: WithDTypeF, B: Backend> EncoderBlock<T, B> {
    pub fn load(
        vb: &Path<B>,
        chin: usize,
        hidden: usize,
        kernel_size: usize,
        stride: usize,
        glu: bool,
        use_bias: bool,
    ) -> Result<Self> {
        let ch_scale = if glu { 2 } else { 1 };
        let conv0 = Conv1d::load(&vb.pp("0"), chin, hidden, kernel_size, stride, use_bias)?;
        let conv2 = Conv1d::load(&vb.pp("2"), hidden, hidden * ch_scale, 1, 1, use_bias)?;
        Ok(Self { conv0, conv2, glu })
    }

    #[tracing::instrument(name = "encoder_forward", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let x = self.conv0.forward(x)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        if self.glu { glu(&x, 1) } else { x.relu() }
    }
}

/// Decoder block: Conv1d(1x1) -> GLU/ReLU -> ConvTranspose1d -> [ReLU]
pub struct DecoderBlock<T: WithDTypeF, B: Backend> {
    pub conv0: Conv1d<T, B>, // decode[0]: Conv1d(hidden, hidden*ch_scale, 1)
    pub convtr: ConvTranspose1d<T, B>, // decode[2]: ConvTranspose1d(hidden, chout, kernel_size, stride)
    pub glu: bool,
    pub has_relu: bool, // decode[3] exists if index > 0
}

impl<T: WithDTypeF, B: Backend> DecoderBlock<T, B> {
    pub fn load(
        vb: &Path<B>,
        hidden: usize,
        chout: usize,
        kernel_size: usize,
        stride: usize,
        glu: bool,
        use_bias: bool,
        has_relu: bool,
    ) -> Result<Self> {
        let ch_scale = if glu { 2 } else { 1 };
        let conv0 = Conv1d::load(&vb.pp("0"), hidden, hidden * ch_scale, 1, 1, use_bias)?;
        let convtr =
            ConvTranspose1d::load(&vb.pp("2"), hidden, chout, kernel_size, stride, use_bias)?;
        Ok(Self { conv0, convtr, glu, has_relu })
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let x = self.conv0.forward(x)?;
        let x = if self.glu { glu(&x, 1)? } else { x.relu()? };
        let x = self.convtr.forward(&x)?;
        if self.has_relu { x.relu() } else { Ok(x) }
    }
}

// ============================================================================
// Fast conv for streaming (matches Python fast_conv)
// ============================================================================

#[tracing::instrument(skip_all)]
fn fast_conv<T: WithDTypeF, B: Backend>(
    conv: &Conv1d<T, B>,
    x: &Tensor<T, B>,
) -> Result<Tensor<T, B>> {
    let (batch, _ch_in, length) = x.dims3()?;
    let kernel = conv.kernel_size;

    if batch != 1 {
        return conv.forward(x);
    }

    if kernel == 1 {
        // x: (1, chin, length) -> (chin, length)
        let x = x.reshape((conv.in_channels, length))?;
        // weight: (chout, chin, 1) -> (chout, chin)
        let w = conv.weight.reshape((conv.out_channels, conv.in_channels))?;
        // out = bias + w @ x
        let out = w.matmul(&x)?;
        let out = if let Some(ref bias) = conv.bias {
            let bias = bias.reshape((conv.out_channels, 1))?;
            out.broadcast_add(&bias)?
        } else {
            out
        };
        out.reshape((1, conv.out_channels, length))
    } else if length == kernel {
        // x: (1, chin, kernel) -> (chin*kernel, 1)
        let x = x.reshape((conv.in_channels * kernel, 1))?;
        // weight: (chout, chin, kernel) -> (chout, chin*kernel)
        let w = conv.weight.reshape((conv.out_channels, conv.in_channels * kernel))?;
        let out = w.matmul(&x)?;
        let out = if let Some(ref bias) = conv.bias {
            let bias = bias.reshape((conv.out_channels, 1))?;
            out.broadcast_add(&bias)?
        } else {
            out
        };
        out.reshape((1, conv.out_channels, 1))
    } else {
        conv.forward(x)
    }
}

// ============================================================================
// Demucs model
// ============================================================================

pub struct Demucs<T: WithDTypeF, B: Backend> {
    pub config: Config,
    pub encoder: Vec<EncoderBlock<T, B>>,
    pub decoder: Vec<DecoderBlock<T, B>>,
    pub lstm: Blstm<T, B>,
}

impl<T: WithDTypeF, B: Backend> Demucs<T, B> {
    pub fn load(vb: &Path<B>, config: Config) -> Result<Self> {
        let mut encoder = Vec::with_capacity(config.depth);
        let mut decoder = Vec::with_capacity(config.depth);

        let mut chin = config.chin;
        let mut chout = config.chout;
        let mut hidden = config.hidden;

        for index in 0..config.depth {
            let enc_vb = vb.pp(format!("encoder.{index}"));
            encoder.push(EncoderBlock::load(
                &enc_vb,
                chin,
                hidden,
                config.kernel_size,
                config.stride,
                config.glu,
                config.bias,
            )?);

            // decoder is inserted at front, so decoder[0] corresponds to last encoder
            let dec_vb = vb.pp(format!("decoder.{}", config.depth - 1 - index));
            let has_relu = index > 0;
            decoder.insert(
                0,
                DecoderBlock::load(
                    &dec_vb,
                    hidden,
                    chout,
                    config.kernel_size,
                    config.stride,
                    config.glu,
                    config.bias,
                    has_relu,
                )?,
            );

            chout = hidden;
            chin = hidden;
            hidden = ((config.growth * hidden as f32).round() as usize).min(config.max_hidden);
        }

        // BLSTM: bi = not causal
        let lstm = Blstm::load(&vb.pp("lstm"), chin, 2, !config.causal)?;

        Ok(Self { config, encoder, decoder, lstm })
    }

    /// Non-streaming forward pass
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(name = "demucs_forward", skip_all)]
    pub fn forward(
        &self,
        mix: &Tensor<T, B>,
        lstm_state: Option<(Tensor<T, B>, Tensor<T, B>)>,
    ) -> Result<(Tensor<T, B>, Option<(Tensor<T, B>, Tensor<T, B>)>)> {
        let (std_val, mix) = if self.config.normalize {
            // Python: std = mix.std(dim=-1, keepdim=True)
            // std = sqrt(E[x^2] - E[x]^2)
            let last_dim = mix.rank() - 1;
            let n = T::from_f32(mix.dim(last_dim)? as f32);

            let mean = mix.sum_keepdim(vec![last_dim])?.scale(T::from_f32(1.0) / n)?;
            let sq = mix.sqr()?;
            let mean_sq = sq.sum_keepdim(vec![last_dim])?.scale(T::from_f32(1.0) / n)?;
            // var = mean_sq - mean^2, implemented as mean_sq + (-1 * mean^2)
            let mean_sq_neg = mean.sqr()?.scale(T::from_f32(-1.0))?;
            let var = mean_sq.add(&mean_sq_neg)?;
            let std = var.sqrt()?;
            let denom = std.add_scalar(T::from_f32(self.config.floor))?;
            let mix_norm = mix.broadcast_div(&denom)?;
            let std_out = std.narrow(1, ..1)?.contiguous()?;
            (Some(std_out), mix_norm)
        } else {
            (None, mix.clone())
        };

        let length = mix.dim(mix.rank() - 1)?;
        let valid_len = self.config.valid_length(length);
        let pad_amount = valid_len.saturating_sub(length);
        let x =
            if pad_amount > 0 { mix.pad_with_zeros(mix.rank() - 1, 0, pad_amount)? } else { mix };

        // Upsample
        let x = match self.config.resample {
            4 => upsample2(&upsample2(&x, 56)?, 56)?,
            2 => upsample2(&x, 56)?,
            _ => x,
        };

        // Encoder
        let mut skips = Vec::with_capacity(self.config.depth);
        let mut x = x;
        for enc in &self.encoder {
            x = enc.forward(&x)?;
            skips.push(x.clone());
        }

        // LSTM: (batch, channels, time) -> (time, batch, channels)
        let x = x.transpose(0, 2)?.transpose(1, 2)?.contiguous()?;
        let (x, new_lstm_state) = self.lstm.forward(&x, lstm_state)?;
        let x = x.transpose(1, 2)?.transpose(0, 2)?.contiguous()?;

        // Decoder with skip connections
        let mut x = x;
        for dec in &self.decoder {
            let skip = skips.pop().context("empty skips")?;
            let x_len = x.dim(2)?;
            let skip = skip.narrow(2, ..x_len.min(skip.dim(2)?))?.contiguous()?;
            x = x.narrow(2, ..skip.dim(2)?)?.contiguous()?.add(&skip)?;
            x = dec.forward(&x)?;
        }

        // Downsample
        let x = match self.config.resample {
            4 => downsample2(&downsample2(&x, 56)?, 56)?,
            2 => downsample2(&x, 56)?,
            _ => x,
        };

        // Trim and denormalize
        let x = x.narrow(2, ..length)?.contiguous()?;
        let x = if let Some(std_val) = std_val { x.broadcast_mul(&std_val)? } else { x };

        Ok((x, new_lstm_state))
    }
}

// ============================================================================
// Streaming implementation (matches Python DemucsStreamer)
// ============================================================================

pub struct DemucsStreamer<T: WithDTypeF, B: Backend> {
    pub demucs: Demucs<T, B>,
    pub lstm_state: Option<(Tensor<T, B>, Tensor<T, B>)>,
    pub conv_state: Option<Vec<Tensor<T, B>>>,
    pub resample_in: Tensor<T, B>,
    pub resample_out: Tensor<T, B>,
    pub mean_variance: Tensor<T, B>,
    pub mean_total: Tensor<T, B>,
    pub mean_decay: f32,
    pub pending: Tensor<T, B>,
    pub resample_lookahead: usize,
    pub resample_buffer: usize,
    pub total_length: usize,
    pub initial_frame_length: usize,
    pub stride: usize,
    pub dry: f32,
}

impl<T: WithDTypeF, B: Backend> DemucsStreamer<T, B> {
    pub fn new(
        demucs: Demucs<T, B>,
        device: &B,
        num_frames: usize,
        resample_lookahead: usize,
        resample_buffer: usize,
        mean_decay_duration: f32,
        dry: f32,
    ) -> Result<Self> {
        let config = &demucs.config;
        let total_stride = config.total_stride();
        let resample_buffer = resample_buffer.min(total_stride);
        let total_length = config.valid_length(1) + resample_lookahead;
        let initial_frame_length = total_length + total_stride * (num_frames - 1);
        let stride = total_stride * num_frames;

        let resample_in = Tensor::zeros((config.chin, resample_buffer), device)?;
        let resample_out = Tensor::zeros((config.chout, resample_buffer), device)?;
        let mean_variance = Tensor::zeros((config.chin, 1), device)?;
        let mean_total = Tensor::zeros((1, 1), device)?;

        let mean_receptive_field_in_samples = mean_decay_duration * config.sample_rate as f32;
        let mean_receptive_field_in_frames = mean_receptive_field_in_samples / total_stride as f32;
        let mean_decay = 1.0 - 1.0 / mean_receptive_field_in_frames;

        let pending = Tensor::zeros((config.chin, 0), device)?;

        Ok(Self {
            demucs,
            lstm_state: None,
            conv_state: None,
            resample_in,
            resample_out,
            mean_variance,
            mean_total,
            mean_decay,
            pending,
            resample_lookahead,
            resample_buffer,
            total_length,
            initial_frame_length,
            stride,
            dry,
        })
    }

    pub fn current_frame_length(&self) -> usize {
        if self.conv_state.is_none() { self.initial_frame_length } else { self.stride }
    }

    fn variance(&self) -> Result<Tensor<T, B>> {
        self.mean_variance.broadcast_div(&self.mean_total)
    }

    /// Feed audio and get processed output. wav: (chin, time)
    #[tracing::instrument(skip_all)]
    pub fn feed(&mut self, wav: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let config = self.demucs.config.clone();
        let resample = config.resample;

        // Append to pending
        self.pending = if self.pending.dim(1)? == 0 {
            wav.clone()
        } else {
            Tensor::cat(&[&self.pending, wav], 1)?
        };

        let mut outs = Vec::new();

        while self.pending.dim(1)? >= self.initial_frame_length {
            let frame = self.pending.narrow(1, ..self.initial_frame_length)?.contiguous()?;

            // Online normalization (matches Python streaming)
            let frame = if config.normalize {
                // variance = (frame**2).mean(dim=-1, keepdim=True)
                let sq = frame.sqr()?;
                let n = T::from_f32(frame.dim(1)? as f32);
                let variance = sq.sum_keepdim(vec![1])?.scale(T::from_f32(1.0) / n)?;

                // Update running stats
                let decay = T::from_f32(self.mean_decay);
                let one_minus_decay = T::from_f32(1.0 - self.mean_decay);
                self.mean_variance =
                    self.mean_variance.scale(decay)?.add(&variance.scale(one_minus_decay)?)?;
                self.mean_total = self.mean_total.scale_add(decay, one_minus_decay)?;
                // frame = frame / (floor + sqrt(variance))
                let running_var = self.variance()?;
                let std = running_var.sqrt()?;
                frame.broadcast_div(&std.add_scalar(T::from_f32(config.floor))?)?
            } else {
                frame
            };

            // Prepend resample buffer
            let padded_frame = Tensor::cat(&[&self.resample_in, &frame], 1)?;
            // Save end of frame for next iteration
            self.resample_in =
                frame.narrow(1, self.stride - self.resample_buffer..self.stride)?.contiguous()?;

            // Upsample
            let frame = match resample {
                4 => upsample2(&upsample2(&padded_frame, 56)?, 56)?,
                2 => upsample2(&padded_frame, 56)?,
                _ => padded_frame,
            };

            // Remove pre-sampling buffer, trim to expected length
            let frame = frame
                .narrow(
                    1,
                    resample * self.resample_buffer
                        ..resample * (self.resample_buffer + self.initial_frame_length),
                )?
                .contiguous()?;

            // Process frame through streaming encoder/decoder
            let (out, extra) = self.separate_frame(&frame)?;

            // Downsample with buffer
            let padded_out = Tensor::cat(&[&self.resample_out, &out, &extra], 1)?;
            self.resample_out =
                out.narrow(1, out.dim(1)? - self.resample_buffer..)?.contiguous()?;

            let out = match resample {
                4 => downsample2(&downsample2(&padded_out, 56)?, 56)?,
                2 => downsample2(&padded_out, 56)?,
                _ => padded_out,
            };

            let out = out
                .narrow(
                    1,
                    self.resample_buffer / resample..self.resample_buffer / resample + self.stride,
                )?
                .contiguous()?;

            // Denormalize
            let out = if config.normalize {
                let std = self.variance()?.sqrt()?;
                let std = std.narrow(0, ..1)?.contiguous()?;
                out.broadcast_mul(&std)?
            } else {
                out
            };

            // Mix dry signal (Python: dry * dry_signal[:chout] + (1-dry) * out)
            // dry_signal would be frame[:, :stride] but we need to handle resample
            // For simplicity, if dry > 0, mix with input
            let out = if self.dry > T::from_f32(0.0).to_f32() {
                // This is approximate - proper implementation would track the dry signal
                out.scale(T::from_f32(1.0 - self.dry))?
            } else {
                out
            };

            outs.push(out);
            self.pending = self.pending.narrow(1, self.stride..)?.contiguous()?;
        }

        if outs.is_empty() {
            Tensor::zeros((config.chout, 0), wav.device())
        } else {
            let refs: Vec<_> = outs.iter().collect();
            Tensor::cat(&refs, 1)
        }
    }

    /// Core streaming separation (matches Python _separate_frame)
    #[tracing::instrument(skip_all)]
    fn separate_frame(&mut self, frame: &Tensor<T, B>) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let config = &self.demucs.config;
        let depth = config.depth;
        let kernel_size = config.kernel_size;
        let conv_stride = config.stride;
        let resample = config.resample;

        let mut skips = Vec::new();
        let mut next_state = Vec::new();
        let mut stride = self.stride * resample;

        // Add batch dimension
        let mut x = frame.unsqueeze(0)?;

        // Encoder
        for (idx, encode) in self.demucs.encoder.iter().enumerate() {
            stride /= conv_stride;
            let length = x.dim(2)?;

            if idx == depth - 1 {
                // Last encoder layer: use fast_conv
                x = fast_conv(&encode.conv0, &x)?;
                x = x.relu()?;
                x = fast_conv(&encode.conv2, &x)?;
                x = if encode.glu { glu(&x, 1)? } else { x.relu()? };
            } else {
                // Use conv state for overlap
                let x_new = if let Some(ref mut conv_state) = self.conv_state {
                    let prev = conv_state.remove(0);
                    let prev = prev.narrow(2, stride..)?.contiguous()?;
                    let tgt = (length - kernel_size) / conv_stride + 1;
                    let missing = tgt.saturating_sub(prev.dim(2)?);
                    if missing > 0 {
                        let offset = length - kernel_size - conv_stride * (missing - 1);
                        let x_slice = x.narrow(2, offset..length)?.contiguous()?;
                        let x_enc = encode.conv0.forward(&x_slice)?;
                        let x_enc = x_enc.relu()?;
                        let x_enc = fast_conv(&encode.conv2, &x_enc)?;
                        let x_enc = if encode.glu { glu(&x_enc, 1)? } else { x_enc.relu()? };
                        Tensor::cat(&[&prev, &x_enc], 2)?
                    } else {
                        prev
                    }
                } else {
                    let x_enc = encode.conv0.forward(&x)?;
                    let x_enc = x_enc.relu()?;
                    let x_enc = fast_conv(&encode.conv2, &x_enc)?;
                    if encode.glu { glu(&x_enc, 1)? } else { x_enc.relu()? }
                };
                next_state.push(x_new.clone());
                x = x_new;
            }
            skips.push(x.clone());
        }

        // LSTM
        let x = x.transpose(0, 2)?.transpose(1, 2)?.contiguous()?;
        let (x, new_lstm_state) = self.demucs.lstm.forward(&x, self.lstm_state.take())?;
        self.lstm_state = new_lstm_state;
        let mut x = x.transpose(1, 2)?.transpose(0, 2)?.contiguous()?;

        // Decoder
        let mut extra: Option<Tensor<T, B>> = None;

        for (idx, decode) in self.demucs.decoder.iter().enumerate() {
            let skip = skips.pop().context("empty skips")?;
            let x_len = x.dim(2)?;

            // Add skip connection
            let skip_slice = skip.narrow(2, ..x_len.min(skip.dim(2)?))?.contiguous()?;
            x = x.narrow(2, ..skip_slice.dim(2)?)?.contiguous()?.add(&skip_slice)?;

            // decode[0] + decode[1]
            x = fast_conv(&decode.conv0, &x)?;
            x = if decode.glu { glu(&x, 1)? } else { x.relu()? };

            // Handle extra for better resampling
            if let Some(ref mut e) = extra {
                let skip_rest = skip.narrow(2, x_len..)?.contiguous()?;
                let e_len = e.dim(2)?;
                let skip_rest =
                    skip_rest.narrow(2, ..e_len.min(skip_rest.dim(2)?))?.contiguous()?;
                *e = e.narrow(2, ..skip_rest.dim(2)?)?.contiguous()?.add(&skip_rest)?;
                // Apply decode[0], decode[1], decode[2] to extra
                let e_conv = fast_conv(&decode.conv0, e)?;
                let e_act = if decode.glu { glu(&e_conv, 1)? } else { e_conv.relu()? };
                *e = decode.convtr.forward(&e_act)?;
            }

            // decode[2]: ConvTranspose1d
            x = decode.convtr.forward(&x)?;

            // Save state and compute extra
            let x_len = x.dim(2)?;
            let state_entry = if let Some(ref bias) = decode.convtr.bias {
                // state = x[..., -stride:] - bias
                let bias_neg = bias.reshape((1, bias.elem_count(), 1))?;
                x.narrow(2, x_len - conv_stride..)?.contiguous()?.broadcast_sub(&bias_neg)?
            } else {
                x.narrow(2, x_len - conv_stride..)?.contiguous()?
            };
            next_state.push(state_entry.clone());

            let new_extra = match extra {
                None => x.narrow(2, x_len - conv_stride..)?.contiguous()?,
                Some(e) => {
                    let e_slice = e.narrow(2, ..conv_stride.min(e.dim(2)?))?.contiguous()?;
                    let state_slice = state_entry.narrow(2, ..e_slice.dim(2)?)?.contiguous()?;
                    let e_new = e_slice.add(&state_slice)?;
                    if e.dim(2)? > conv_stride {
                        let e_rest = e.narrow(2, conv_stride..)?.contiguous()?;
                        Tensor::cat(&[&e_new, &e_rest], 2)?
                    } else {
                        e_new
                    }
                }
            };
            extra = Some(new_extra);

            x = x.narrow(2, ..x_len - conv_stride)?.contiguous()?;

            // Add previous decoder state
            if let Some(ref mut conv_state) = self.conv_state
                && !conv_state.is_empty()
            {
                let prev = conv_state.remove(0);
                let prev_len = prev.dim(2)?;
                let x_slice = x.narrow(2, ..prev_len.min(x.dim(2)?))?.contiguous()?;
                let prev_slice = prev.narrow(2, ..x_slice.dim(2)?)?.contiguous()?;
                let x_updated = x_slice.add(&prev_slice)?;
                if x.dim(2)? > prev_len {
                    let x_rest = x.narrow(2, prev_len..)?.contiguous()?;
                    x = Tensor::cat(&[&x_updated, &x_rest], 2)?;
                } else {
                    x = x_updated;
                }
            }

            // Final ReLU if not last decoder
            if idx != depth - 1 {
                x = x.relu()?;
                if let Some(ref mut e) = extra {
                    *e = e.relu()?;
                }
            }
        }

        self.conv_state = Some(next_state);

        // Remove batch dimension
        let x = x.reshape((config.chout, x.dim(2)?))?;
        let extra = extra.unwrap();
        let extra_len = extra.dim(2)?;
        let extra = extra.reshape((config.chout, extra_len))?;

        Ok((x, extra))
    }

    pub fn flush(&mut self) -> Result<Tensor<T, B>> {
        let pending_length = self.pending.dim(1)?;
        let config = &self.demucs.config;
        let padding = Tensor::zeros((config.chin, self.total_length), self.pending.device())?;
        let out = self.feed(&padding)?;
        let out_len = out.dim(1)?;
        if out_len > pending_length && pending_length > 0 {
            out.narrow(1, ..pending_length)?.contiguous()
        } else {
            Ok(out)
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        let config = &self.demucs.config;
        let device = self.pending.device().clone();
        self.lstm_state = None;
        self.conv_state = None;
        self.resample_in = Tensor::zeros((config.chin, self.resample_buffer), &device)?;
        self.resample_out = Tensor::zeros((config.chout, self.resample_buffer), &device)?;
        self.mean_variance = Tensor::zeros((config.chin, 1), &device)?;
        self.mean_total = Tensor::zeros((1, 1), &device)?;
        self.pending = Tensor::zeros((config.chin, 0), &device)?;
        Ok(())
    }
}
