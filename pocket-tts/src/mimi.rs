use crate::conv::pad_for_conv1d;
use crate::conv::{StreamingConv1dState, StreamingConvTr1dState};
use crate::dummy_quantizer::DummyQuantizer;
use crate::resample::{ConvDownsample1d, ConvTrUpsample1d};
use crate::seanet::{SEANetDecoder, SEANetDecoderState, SEANetEncoder, SEANetEncoderState};
use crate::transformer::{ProjectedTransformer, StreamingTransformerState};
use xn::nn::var_builder::Path;
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: f64,
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
    #[serde(default)]
    pub downsample_channel_wise: bool,
}

pub struct MimiEncoder<Q: BackendQ> {
    encoder: SEANetEncoder<Q::T, Q::B>,
    encoder_transformer: ProjectedTransformer<Q>,
    downsample: Option<ConvDownsample1d<Q::T, Q::B>>,
    frame_size: usize,
}

pub struct MimiDecoder<Q: BackendQ> {
    decoder: SEANetDecoder<Q::T, Q::B>,
    decoder_transformer: ProjectedTransformer<Q>,
    upsample: Option<ConvTrUpsample1d<Q::T, Q::B>>,
    pub quantizer: DummyQuantizer<Q::T, Q::B>,
    pub sample_rate: usize,
    pub frame_rate: f64,
}

pub struct MimiModel<Q: BackendQ> {
    encoder: MimiEncoder<Q>,
    decoder: MimiDecoder<Q>,
    pub sample_rate: usize,
}

#[derive(Debug, Clone)]
pub struct MimiEncoderState<T: WithDTypeF, B: Backend> {
    encoder_state: SEANetEncoderState<T, B>,
    encoder_transformer_state: StreamingTransformerState<T, B>,
    downsample_state: Option<StreamingConv1dState<T, B>>,
}

#[derive(Debug, Clone)]
pub struct MimiDecoderState<T: WithDTypeF, B: Backend> {
    decoder_state: SEANetDecoderState<T, B>,
    decoder_transformer_state: StreamingTransformerState<T, B>,
    upsample_state: Option<StreamingConvTr1dState<T, B>>,
}

#[derive(Debug, Clone)]
pub struct MimiState<T: WithDTypeF, B: Backend> {
    encoder_state: MimiEncoderState<T, B>,
    decoder_state: MimiDecoderState<T, B>,
}

impl<Q: BackendQ> MimiEncoder<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &MimiConfig) -> Result<Self> {
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

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let downsample = if (encoder_frame_rate - cfg.frame_rate).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
            let ds = ConvDownsample1d::load(
                &vb.pp("downsample"),
                downsample_stride,
                cfg.dimension,
                cfg.downsample_channel_wise,
            )?;
            Some(ds)
        } else {
            None
        };
        let frame_size = (cfg.sample_rate as f64 / cfg.frame_rate).round() as usize;
        Ok(Self { encoder, encoder_transformer, downsample, frame_size })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<MimiEncoderState<Q::T, Q::B>> {
        let downsample_state = match &self.downsample {
            Some(ds) => Some(ds.init_state(batch_size)?),
            None => None,
        };
        let s = MimiEncoderState {
            encoder_state: self.encoder.init_state(batch_size)?,
            encoder_transformer_state: self
                .encoder_transformer
                .init_state(batch_size, sequence_length)?,
            downsample_state,
        };
        Ok(s)
    }

    /// Encode audio to latent (non-streaming). Returns [B, C, T'].
    pub fn encode_to_latent(&self, x: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let x = pad_for_conv1d(x, self.frame_size, self.frame_size)?;
        let mut enc_state = self.encoder.init_state(x.dim(0usize)?)?;
        let emb = self.encoder.forward(&x, &mut enc_state)?;
        let mut et_state = self.encoder_transformer.init_state(x.dim(0usize)?, 8192)?;
        let mut outs = self.encoder_transformer.forward(&emb, &mut et_state)?;
        let emb = outs.swap_remove(0);
        // Downsample to frame rate
        match &self.downsample {
            Some(ds) => ds.forward_no_state(&emb),
            None => Ok(emb),
        }
    }

    pub fn encode_to_latent_step(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut MimiEncoderState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let x = pad_for_conv1d(x, self.frame_size, self.frame_size)?;
        let emb = self.encoder.forward(&x, &mut state.encoder_state)?;
        let mut outs =
            self.encoder_transformer.forward(&emb, &mut state.encoder_transformer_state)?;
        let emb = outs.swap_remove(0);
        // Downsample to frame rate
        match (&self.downsample, &mut state.downsample_state) {
            (Some(ds), Some(ds_state)) => ds.forward(&emb, ds_state),
            _ => Ok(emb),
        }
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
}

impl<Q: BackendQ> MimiDecoder<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = crate::conv::PadMode::Constant;

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

        let quantizer = DummyQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dimension,
            cfg.quantizer_output_dimension,
        )?;
        let output_dimensions = vec![cfg.dimension];
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
        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let upsample = if (encoder_frame_rate - cfg.frame_rate).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
            let us = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, cfg.dimension)?;
            Some(us)
        } else {
            None
        };
        Ok(Self {
            decoder,
            decoder_transformer,
            upsample,
            quantizer,
            sample_rate: cfg.sample_rate,
            frame_rate: cfg.frame_rate,
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<MimiDecoderState<Q::T, Q::B>> {
        let upsample_state = match &self.upsample {
            Some(us) => Some(us.init_state(batch_size)?),
            None => None,
        };
        let s = MimiDecoderState {
            decoder_state: self.decoder.init_state(batch_size)?,
            decoder_transformer_state: self
                .decoder_transformer
                .init_state(batch_size, sequence_length)?,
            upsample_state,
        };
        Ok(s)
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    pub fn decode_from_latent_step(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        state: &mut MimiDecoderState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        // Upsample to encoder frame rate
        let emb = match (&self.upsample, &mut state.upsample_state) {
            (Some(us), Some(us_state)) => us.forward(latent, us_state)?,
            _ => latent.clone(),
        };

        let outs = self.decoder_transformer.forward(&emb, &mut state.decoder_transformer_state)?;
        self.decoder.forward(&outs[0], &mut state.decoder_state)
    }

    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate).round() as usize
    }
}

impl<Q: BackendQ> MimiModel<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &MimiConfig) -> Result<Self> {
        let encoder = MimiEncoder::load(vb, cfg)?;
        let decoder = MimiDecoder::load(vb, cfg)?;

        Ok(Self { encoder, decoder, sample_rate: cfg.sample_rate })
    }

    pub fn frame_size(&self) -> usize {
        self.encoder.frame_size
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<MimiState<Q::T, Q::B>> {
        let s = MimiState {
            encoder_state: self.encoder.init_state(batch_size, sequence_length)?,
            decoder_state: self.decoder.init_state(batch_size, sequence_length)?,
        };
        Ok(s)
    }

    /// Encode audio to latent (non-streaming). Returns [B, C, T'].
    pub fn encode_to_latent(&self, x: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        self.encoder.encode_to_latent(x)
    }

    pub fn encode_to_latent_step(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut MimiState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        self.encoder.encode_to_latent_step(x, &mut state.encoder_state)
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    pub fn decode_from_latent_step(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        state: &mut MimiState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        self.decoder.decode_from_latent_step(latent, &mut state.decoder_state)
    }
}
