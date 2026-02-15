use crate::conv::{
    PadMode, StreamingConv1d, StreamingConv1dState, StreamingConvTr1dState,
    StreamingConvTranspose1d,
};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

// ---- SEANetResnetBlock ----

pub struct SEANetResnetBlock<T: WithDTypeF, B: Backend> {
    convs: Vec<StreamingConv1d<T, B>>,
}

pub struct SEANetResnetBlockState<T: WithDTypeF, B: Backend> {
    conv_states: Vec<StreamingConv1dState<T, B>>,
}

impl<T: WithDTypeF, B: Backend> SEANetResnetBlock<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        pad_mode: PadMode,
        compress: usize,
    ) -> Result<Self> {
        let hidden = dim / compress;
        let mut convs = Vec::new();
        for (i, (&ks, &dil)) in kernel_sizes.iter().zip(dilations.iter()).enumerate() {
            let in_c = if i == 0 { dim } else { hidden };
            let out_c = if i == kernel_sizes.len() - 1 { dim } else { hidden };
            // block.{2*i+1} in Python (ELU at even indices, Conv at odd)
            let conv = StreamingConv1d::load(
                &vb.pp("block").pp(2 * i + 1),
                in_c,
                out_c,
                ks,
                1,
                dil,
                pad_mode,
                1,
                true,
            )?;
            convs.push(conv);
        }
        Ok(Self { convs })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<SEANetResnetBlockState<T, B>> {
        let conv_states =
            self.convs.iter().map(|c| c.init_state(batch_size)).collect::<Result<Vec<_>>>()?;
        Ok(SEANetResnetBlockState { conv_states })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut SEANetResnetBlockState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut v = x.clone();
        for (conv, cs) in self.convs.iter().zip(state.conv_states.iter_mut()) {
            v = v.elu(1.0)?;
            v = conv.forward(&v, cs)?;
        }
        x.add(&v)
    }
}

// ---- SEANetEncoder ----

struct EncoderLayer<T: WithDTypeF, B: Backend> {
    residuals: Vec<SEANetResnetBlock<T, B>>,
    downsample: StreamingConv1d<T, B>,
}

pub struct SEANetEncoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamingConv1d<T, B>,
    layers: Vec<EncoderLayer<T, B>>,
    final_conv: StreamingConv1d<T, B>,
    pub hop_length: usize,
    pub dimension: usize,
}

type EncoderLayerState<T, B> = (Vec<SEANetResnetBlockState<T, B>>, StreamingConv1dState<T, B>);

pub struct SEANetEncoderState<T: WithDTypeF, B: Backend> {
    init_conv_state: StreamingConv1dState<T, B>,
    layer_states: Vec<EncoderLayerState<T, B>>,
    final_conv_state: StreamingConv1dState<T, B>,
}

impl<T: WithDTypeF, B: Backend> SEANetEncoder<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: &[usize],
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: PadMode,
        compress: usize,
    ) -> Result<Self> {
        // Ratios are reversed for encoder
        let ratios: Vec<usize> = ratios.iter().rev().copied().collect();
        let hop_length: usize = ratios.iter().product();

        let mut mult = 1usize;
        let init_conv = StreamingConv1d::load(
            &vb.pp("model").pp(0),
            channels,
            mult * n_filters,
            kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        let mut layers = Vec::new();
        let mut layer_idx = 1usize;

        for &ratio in &ratios {
            let mut residuals = Vec::new();
            for j in 0..n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                let block = SEANetResnetBlock::load(
                    &vb.pp("model").pp(layer_idx),
                    mult * n_filters,
                    &[residual_kernel_size, 1],
                    &[dilation, 1],
                    pad_mode,
                    compress,
                )?;
                residuals.push(block);
                layer_idx += 1;
            }

            // ELU at layer_idx, downsample conv at layer_idx+1
            let downsample = StreamingConv1d::load(
                &vb.pp("model").pp(layer_idx + 1),
                mult * n_filters,
                mult * n_filters * 2,
                ratio * 2,
                ratio,
                1,
                pad_mode,
                1,
                true,
            )?;
            layer_idx += 2;
            layers.push(EncoderLayer { residuals, downsample });
            mult *= 2;
        }

        // ELU at layer_idx, final conv at layer_idx+1
        let final_conv = StreamingConv1d::load(
            &vb.pp("model").pp(layer_idx + 1),
            mult * n_filters,
            dimension,
            last_kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        Ok(Self { init_conv, layers, final_conv, hop_length, dimension })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<SEANetEncoderState<T, B>> {
        let init_conv_state = self.init_conv.init_state(batch_size)?;
        let layer_states = self
            .layers
            .iter()
            .map(|l| {
                let res = l
                    .residuals
                    .iter()
                    .map(|r| r.init_state(batch_size))
                    .collect::<Result<Vec<_>>>()?;
                let down = l.downsample.init_state(batch_size)?;
                Ok((res, down))
            })
            .collect::<Result<Vec<_>>>()?;
        let final_conv_state = self.final_conv.init_state(batch_size)?;
        Ok(SEANetEncoderState { init_conv_state, layer_states, final_conv_state })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut SEANetEncoderState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut x = self.init_conv.forward(x, &mut state.init_conv_state)?;
        for (layer, (res_states, down_state)) in
            self.layers.iter().zip(state.layer_states.iter_mut())
        {
            for (res, rs) in layer.residuals.iter().zip(res_states.iter_mut()) {
                x = res.forward(&x, rs)?;
            }
            x = x.elu(1.0)?;
            x = layer.downsample.forward(&x, down_state)?;
        }
        x = x.elu(1.0)?;
        self.final_conv.forward(&x, &mut state.final_conv_state)
    }
}

// ---- SEANetDecoder ----

struct DecoderLayer<T: WithDTypeF, B: Backend> {
    upsample: StreamingConvTranspose1d<T, B>,
    residuals: Vec<SEANetResnetBlock<T, B>>,
}

pub struct SEANetDecoder<T: WithDTypeF, B: Backend> {
    init_conv: StreamingConv1d<T, B>,
    layers: Vec<DecoderLayer<T, B>>,
    final_conv: StreamingConv1d<T, B>,
}

type DecoderLayerState<T, B> = (StreamingConvTr1dState<T, B>, Vec<SEANetResnetBlockState<T, B>>);

pub struct SEANetDecoderState<T: WithDTypeF, B: Backend> {
    init_conv_state: StreamingConv1dState<T, B>,
    layer_states: Vec<DecoderLayerState<T, B>>,
    final_conv_state: StreamingConv1dState<T, B>,
}

impl<T: WithDTypeF, B: Backend> SEANetDecoder<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: &[usize],
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: PadMode,
        compress: usize,
    ) -> Result<Self> {
        let mut mult = 1 << ratios.len();

        let init_conv = StreamingConv1d::load(
            &vb.pp("model").pp(0),
            dimension,
            mult * n_filters,
            kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        let mut layers = Vec::new();
        let mut layer_idx = 1usize;

        for &ratio in ratios {
            // ELU at layer_idx, upsample at layer_idx+1
            let upsample = StreamingConvTranspose1d::load(
                &vb.pp("model").pp(layer_idx + 1),
                mult * n_filters,
                mult * n_filters / 2,
                ratio * 2,
                ratio,
                1,
                true,
            )?;
            layer_idx += 2;

            let mut residuals = Vec::new();
            for j in 0..n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                let block = SEANetResnetBlock::load(
                    &vb.pp("model").pp(layer_idx),
                    mult * n_filters / 2,
                    &[residual_kernel_size, 1],
                    &[dilation, 1],
                    pad_mode,
                    compress,
                )?;
                residuals.push(block);
                layer_idx += 1;
            }

            layers.push(DecoderLayer { upsample, residuals });
            mult /= 2;
        }

        // ELU at layer_idx, final conv at layer_idx+1
        let final_conv = StreamingConv1d::load(
            &vb.pp("model").pp(layer_idx + 1),
            n_filters,
            channels,
            last_kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        Ok(Self { init_conv, layers, final_conv })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<SEANetDecoderState<T, B>> {
        let init_conv_state = self.init_conv.init_state(batch_size)?;
        let layer_states = self
            .layers
            .iter()
            .map(|l| {
                let up = l.upsample.init_state(batch_size)?;
                let res = l
                    .residuals
                    .iter()
                    .map(|r| r.init_state(batch_size))
                    .collect::<Result<Vec<_>>>()?;
                Ok((up, res))
            })
            .collect::<Result<Vec<_>>>()?;
        let final_conv_state = self.final_conv.init_state(batch_size)?;
        Ok(SEANetDecoderState { init_conv_state, layer_states, final_conv_state })
    }

    pub fn forward(
        &self,
        z: &Tensor<T, B>,
        state: &mut SEANetDecoderState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut z = self.init_conv.forward(z, &mut state.init_conv_state)?;
        for (layer, (up_state, res_states)) in self.layers.iter().zip(state.layer_states.iter_mut())
        {
            z = z.elu(1.0)?;
            z = layer.upsample.forward(&z, up_state)?;
            for (res, rs) in layer.residuals.iter().zip(res_states.iter_mut()) {
                z = res.forward(&z, rs)?;
            }
        }
        z = z.elu(1.0)?;
        self.final_conv.forward(&z, &mut state.final_conv_state)
    }
}
