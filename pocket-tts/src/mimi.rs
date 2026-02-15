use crate::conv::pad_for_conv1d;
use crate::conv::{StreamingConv1dState, StreamingConvTr1dState};
use crate::dummy_quantizer::DummyQuantizer;
use crate::mimi_transformer::{ProjectedTransformer, StreamingTransformerState};
use crate::resample::{ConvDownsample1d, ConvTrUpsample1d};
use crate::seanet::{SEANetDecoder, SEANetDecoderState, SEANetEncoder, SEANetEncoderState};
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: usize,
    pub dimension: usize,
    pub quantizer_dimension: usize,
    pub quantizer_output_dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    // Transformer params
    pub transformer_d_model: usize,
    pub transformer_num_heads: usize,
    pub transformer_num_layers: usize,
    pub transformer_layer_scale: f64,
    pub transformer_context: usize,
    pub transformer_max_period: f32,
    pub transformer_dim_feedforward: usize,
}

pub struct MimiModel<T: WithDTypeF, B: Backend> {
    encoder: SEANetEncoder<T, B>,
    decoder: SEANetDecoder<T, B>,
    encoder_transformer: ProjectedTransformer<T, B>,
    decoder_transformer: ProjectedTransformer<T, B>,
    pub quantizer: DummyQuantizer<T, B>,
    downsample: Option<ConvDownsample1d<T, B>>,
    upsample: Option<ConvTrUpsample1d<T, B>>,
    frame_rate: usize,
    _encoder_frame_rate: f64,
    pub sample_rate: usize,
    _dimension: usize,
}

pub struct MimiState<T: WithDTypeF, B: Backend> {
    _encoder_state: SEANetEncoderState<T, B>,
    decoder_state: SEANetDecoderState<T, B>,
    _encoder_transformer_state: StreamingTransformerState<T, B>,
    decoder_transformer_state: StreamingTransformerState<T, B>,
    _downsample_state: Option<StreamingConv1dState<T, B>>,
    upsample_state: Option<StreamingConvTr1dState<T, B>>,
}

impl<T: WithDTypeF, B: Backend> MimiModel<T, B> {
    pub fn load(vb: &Path<B>, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = crate::conv::PadMode::Constant;

        let encoder = SEANetEncoder::load(
            &vb.pp("encoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let decoder = SEANetDecoder::load(
            &vb.pp("decoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let output_dimensions = vec![cfg.dimension];
        let encoder_transformer = ProjectedTransformer::load(
            &vb.pp("encoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;

        let decoder_transformer = ProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;

        let quantizer = DummyQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dimension,
            cfg.quantizer_output_dimension,
        )?;

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let (downsample, upsample) = if (encoder_frame_rate - cfg.frame_rate as f64).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate as f64) as usize;
            let ds =
                ConvDownsample1d::load(&vb.pp("downsample"), downsample_stride, cfg.dimension)?;
            let us = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, cfg.dimension)?;
            (Some(ds), Some(us))
        } else {
            (None, None)
        };

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            frame_rate: cfg.frame_rate,
            _encoder_frame_rate: encoder_frame_rate,
            sample_rate: cfg.sample_rate,
            _dimension: cfg.dimension,
        })
    }

    pub fn frame_size(&self) -> usize {
        self.sample_rate / self.frame_rate
    }

    pub fn init_state(&self, batch_size: usize, sequence_length: usize) -> Result<MimiState<T, B>> {
        let upsample_state = match &self.upsample {
            Some(us) => Some(us.init_state(batch_size)?),
            None => None,
        };
        let _downsample_state = match &self.downsample {
            Some(ds) => Some(ds.init_state(batch_size)?),
            None => None,
        };
        let s = MimiState {
            _encoder_state: self.encoder.init_state(batch_size)?,
            decoder_state: self.decoder.init_state(batch_size)?,
            _encoder_transformer_state: self
                .encoder_transformer
                .init_state(batch_size, sequence_length)?,
            decoder_transformer_state: self
                .decoder_transformer
                .init_state(batch_size, sequence_length)?,
            _downsample_state,
            upsample_state,
        };
        Ok(s)
    }

    /// Encode audio to latent (non-streaming). Returns [B, C, T'].
    pub fn encode_to_latent(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let frame_size = self.frame_size();
        let x = pad_for_conv1d(x, frame_size, frame_size)?;

        let mut enc_state = self.encoder.init_state(x.dim(0usize)?)?;
        let emb = self.encoder.forward(&x, &mut enc_state)?;

        let mut et_state = self.encoder_transformer.init_state(x.dim(0usize)?, 8192)?;
        let outs = self.encoder_transformer.forward(&emb, &mut et_state)?;
        let emb = &outs[0];

        // Downsample to frame rate
        match &self.downsample {
            Some(ds) => ds.forward_no_state(emb),
            None => Ok(emb.clone()),
        }
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    pub fn decode_from_latent(
        &self,
        latent: &Tensor<T, B>,
        state: &mut MimiState<T, B>,
    ) -> Result<Tensor<T, B>> {
        // Upsample to encoder frame rate
        let emb = match (&self.upsample, &mut state.upsample_state) {
            (Some(us), Some(us_state)) => us.forward(latent, us_state)?,
            _ => latent.clone(),
        };

        let outs = self.decoder_transformer.forward(&emb, &mut state.decoder_transformer_state)?;
        self.decoder.forward(&outs[0], &mut state.decoder_state)
    }
}
