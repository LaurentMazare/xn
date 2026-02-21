use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use crate::{conv, quantization, seanet, transformer};
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

pub struct Mimi<T: WithDTypeF, B: Backend> {
    encoder: seanet::SeaNetEncoder<T, B>,
    decoder: seanet::SeaNetDecoder<T, B>,
    encoder_transformer: transformer::ProjectedTransformer<T, B>,
    decoder_transformer: transformer::ProjectedTransformer<T, B>,
    downsample: conv::ConvDownsample1d<T, B>,
    upsample: conv::ConvTrUpsample1d<T, B>,
    quantizer: quantization::SplitResidualVectorQuantizer<T, B>,
    config: Config,
}

impl<T: WithDTypeF, B: Backend> Mimi<T, B> {
    pub fn load(vb: &Path<B>, cfg: Config, device: &B) -> Result<Self> {
        let dim = cfg.seanet.dimension;

        let encoder = seanet::SeaNetEncoder::load(&vb.pp("encoder"), &cfg.seanet)?;
        let decoder = seanet::SeaNetDecoder::load(&vb.pp("decoder"), &cfg.seanet)?;

        let encoder_transformer = transformer::ProjectedTransformer::load(
            &vb.pp("encoder_transformer"),
            dim,
            &cfg.transformer,
            device,
        )?;
        let decoder_transformer = transformer::ProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            dim,
            &cfg.transformer,
            device,
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
