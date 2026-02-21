use crate::streaming::{StreamMask, StreamTensor};
use crate::{batched_transformer as bt, conv, quantization, seanet, transformer};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: seanet::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: 0,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
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
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
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

// ============================================================================
// Streaming State Types
// ============================================================================

pub struct MimiEncodeState<T: WithDTypeF, B: Backend> {
    pub encoder: seanet::EncoderState<T, B>,
    pub encoder_transformer: bt::BatchedTransformerState<T, B>,
    pub downsample: conv::Conv1dState<T, B>,
}

pub struct MimiDecodeState<T: WithDTypeF, B: Backend> {
    pub upsample: conv::ConvTr1dState<T, B>,
    pub decoder_transformer: bt::BatchedTransformerState<T, B>,
    pub decoder: seanet::DecoderState<T, B>,
}

// ============================================================================
// Mimi
// ============================================================================

pub struct Mimi<T: WithDTypeF, B: Backend> {
    encoder: seanet::SeaNetEncoder<T, B>,
    decoder: seanet::SeaNetDecoder<T, B>,
    encoder_transformer: bt::BatchedProjectedTransformer<T, B>,
    decoder_transformer: bt::BatchedProjectedTransformer<T, B>,
    downsample: conv::ConvDownsample1d<T, B>,
    upsample: conv::ConvTrUpsample1d<T, B>,
    quantizer: quantization::SplitResidualVectorQuantizer<T, B>,
    config: Config,
}

impl<T: WithDTypeF, B: Backend> Mimi<T, B> {
    pub fn load(vb: &Path<B>, cfg: Config) -> Result<Self> {
        let dim = cfg.seanet.dimension;

        let encoder = seanet::SeaNetEncoder::load(&vb.pp("encoder"), &cfg.seanet)?;
        let decoder = seanet::SeaNetDecoder::load(&vb.pp("decoder"), &cfg.seanet)?;

        let encoder_transformer = bt::BatchedProjectedTransformer::load(
            &vb.pp("encoder_transformer"),
            dim,
            &cfg.transformer,
        )?;
        let decoder_transformer = bt::BatchedProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            dim,
            &cfg.transformer,
        )?;

        let quantizer = quantization::SplitResidualVectorQuantizer::load(
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

        let downsample = conv::ConvDownsample1d::load(
            &vb.pp("downsample"),
            downsample_stride,
            dim,
            /* causal */ true,
            /* learnt */ true,
        )?;
        let upsample = conv::ConvTrUpsample1d::load(
            &vb.pp("upsample"),
            downsample_stride,
            dim,
            /* causal */ true,
            /* learnt */ true,
        )?;

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

    pub fn init_encode_state(&self, batch_size: usize) -> Result<MimiEncodeState<T, B>> {
        let state = MimiEncodeState {
            encoder: self.encoder.init_state(),
            encoder_transformer: self.encoder_transformer.init_state(batch_size)?,
            downsample: self.downsample.init_state(),
        };
        Ok(state)
    }

    pub fn init_decode_state(&self, batch_size: usize) -> Result<MimiDecodeState<T, B>> {
        let state = MimiDecodeState {
            upsample: self.upsample.init_state(),
            decoder_transformer: self.decoder_transformer.init_state(batch_size)?,
            decoder: self.decoder.init_state(),
        };
        Ok(state)
    }

    /// Encode audio to codes (non-streaming).
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let batch_size = xs.dim(0)?;
        let xs = self.encoder.forward(xs)?;
        let mut tf_state = self.encoder_transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);
        let xs = self
            .encoder_transformer
            .forward(&xs, &mut tf_state, &mask)?;
        let xs = &xs[0];
        let xs = self.downsample.forward(xs)?;
        self.quantizer.encode(&xs)
    }

    /// Decode codes to audio (non-streaming).
    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let batch_size = codes.dim(0)?;
        let emb = self.quantizer.decode(codes)?;
        let emb = self.upsample.forward(&emb)?;
        let mut tf_state = self.decoder_transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);
        let outs = self
            .decoder_transformer
            .forward(&emb, &mut tf_state, &mask)?;
        self.decoder.forward(&outs[0])
    }

    /// Encode audio step (streaming).
    pub fn encode_step(
        &self,
        xs: &StreamTensor<T, B>,
        state: &mut MimiEncodeState<T, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<i64, B>> {
        let xs = self.encoder.step(xs, &mut state.encoder, mask)?;
        let xs = self
            .encoder_transformer
            .step(&xs, &mut state.encoder_transformer, mask)?;
        let xs = self.downsample.step(&xs, &mut state.downsample, mask)?;
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.quantizer.encode(xs)?)),
        }
    }

    /// Decode codes step (streaming).
    pub fn decode_step(
        &self,
        codes: &StreamTensor<i64, B>,
        state: &mut MimiDecodeState<T, B>,
        mask: &StreamMask,
    ) -> Result<StreamTensor<T, B>> {
        let emb: StreamTensor<T, B> = match codes.as_option() {
            Some(codes) => StreamTensor::from_tensor(self.quantizer.decode(codes)?),
            None => StreamTensor::empty(),
        };
        let emb = self.upsample.step(&emb, &mut state.upsample, mask)?;
        let out = self
            .decoder_transformer
            .step(&emb, &mut state.decoder_transformer, mask)?;
        self.decoder.step(&out, &mut state.decoder, mask)
    }
}
