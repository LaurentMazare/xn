use crate::flow_lm::{FlowLM, FlowLMConfig, FlowLMState};
use crate::mimi::{MimiConfig, MimiDecoder, MimiDecoderState, MimiEncoder};
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FuserConfig {
    pub sum: Vec<String>,
    pub streaming_sum: Vec<String>,
    pub prepend: Vec<String>,
    pub cross: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LutConditioner {
    n_bins: usize,
    dim: usize,
    possible_values: Vec<String>,
    tokenizer: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerInnerConfig {
    Lut { lut: LutConditioner },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConditionerConfig {
    pub name: String,
    #[serde(flatten)]
    pub inner: ConditionerInnerConfig,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub fuser: FuserConfig,
    pub conditioners: Vec<ConditionerConfig>,
}

impl TTSConfig {
    pub fn v202601(temp: f32) -> Self {
        Self {
            flow_lm: FlowLMConfig {
                d_model: 1024,
                num_heads: 16,
                num_layers: 6,
                dim_feedforward: 4096,
                max_period: 10000.0,
                n_bins: 4000,
                lut_dim: 1024,
                flow_dim: 512,
                flow_depth: 6,
                ldim: 32,
            },
            mimi: MimiConfig {
                channels: 1,
                sample_rate: 24000,
                frame_rate: 12.5,
                dimension: 512,
                quantizer_dimension: 32,
                quantizer_output_dimension: 512,
                n_filters: 64,
                n_residual_layers: 1,
                ratios: vec![6, 5, 4],
                kernel_size: 7,
                last_kernel_size: 3,
                residual_kernel_size: 3,
                dilation_base: 2,
                compress: 2,
                transformer_d_model: 512,
                transformer_num_heads: 8,
                transformer_num_layers: 2,
                transformer_layer_scale: 0.01,
                transformer_context: 250,
                transformer_max_period: 10000.0,
                transformer_dim_feedforward: 2048,
                downsample_channel_wise: false,
            },
            temp,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
            conditioners: vec![],
            fuser: FuserConfig {
                sum: vec![],
                streaming_sum: vec![],
                prepend: vec![],
                cross: vec![],
            },
        }
    }
}

pub struct TTSModel<Q: BackendQ> {
    pub flow_lm: FlowLM<Q>,
    pub mimi: MimiDecoder<Unquantized<f32, Q::B>>,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

#[derive(Clone, Debug)]
pub struct TTSState<Q: BackendQ> {
    pub flow_lm_state: FlowLMState<Q>,
}

impl<Q: BackendQ> TTSModel<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, &cfg.flow_lm)?;
        let mimi = MimiDecoder::load(&vb.pp("mimi"), &cfg.mimi)?;

        Ok(Self {
            flow_lm,
            mimi,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
        })
    }

    pub fn with_eos_threshold(mut self, eos_threshold: f32) -> Self {
        self.eos_threshold = eos_threshold;
        self
    }

    pub fn sample_rate(&self) -> usize {
        self.mimi.sample_rate
    }

    /// Initialize flow LM state with the given sequence length budget.
    pub fn init_flow_lm_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<TTSState<Q>> {
        Ok(TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? })
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState<Q>, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text_with_padding(
        &self,
        state: &mut TTSState<Q>,
        text_tokens: &[u32],
        pad_to: usize,
    ) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let (batch_size, seq_len, dim) = text_embeddings.dims3()?;
        let padding_required = pad_to.saturating_sub(seq_len);
        let text_embeddings = if padding_required > 0
            && let Some(padding_embeds) = self.flow_lm.conditioner.learnt_padding()
        {
            let padding_embeds =
                padding_embeds.expand((batch_size, padding_required, dim))?.contiguous()?;
            Tensor::cat(&[&text_embeddings, &padding_embeds], 1)?
        } else {
            text_embeddings
        };
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    pub fn prompt_text_null(&self, state: &mut TTSState<Q>) -> Result<()> {
        let empty_text = match self.flow_lm.conditioner.learnt_padding() {
            None => xn::bail!("Model does not support null text prompt"),
            Some(p) => p,
        };
        let dev = empty_text.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, empty_text, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with audio conditioning. Increments state.
    pub fn prompt_audio(
        &self,
        state: &mut TTSState<Q>,
        audio_conditioning: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        let dev = audio_conditioning.device();
        let empty_text = Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), dev)?;
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        let text_embeddings = Tensor::cat(&[&empty_text, audio_conditioning], 1)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// Run one autoregressive generation step.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    #[allow(clippy::type_complexity)]
    pub fn generate_step(
        &self,
        state: &mut TTSState<Q>,
        backbone_input: &Tensor<Q::T, Q::B>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let dev = backbone_input.device();
        let empty_text = Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), dev)?;

        let (latent, is_eos) = self.flow_lm.sample_next_latent(
            backbone_input,
            &empty_text,
            &mut state.flow_lm_state,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )?;

        Ok((latent, is_eos))
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_step_cfg(
        &self,
        state: &mut TTSState<Q>,
        null_state: &mut TTSState<Q>,
        cfg_coef: f32,
        backbone_input: &Tensor<Q::T, Q::B>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let dev = backbone_input.device();
        let empty_text = Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), dev)?;

        let (latent, is_eos) = self.flow_lm.sample_next_latent_cfg(
            backbone_input,
            &empty_text,
            &mut state.flow_lm_state,
            &mut null_state.flow_lm_state,
            cfg_coef,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )?;

        Ok((latent, is_eos))
    }

    /// Decode latent to audio using mimi (streaming).
    pub fn decode_latent(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        mimi_state: &mut MimiDecoderState<f32, Q::B>,
    ) -> Result<Tensor<f32, Q::B>> {
        let denorm =
            latent.broadcast_mul(&self.flow_lm.emb_std)?.broadcast_add(&self.flow_lm.emb_mean)?;

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        // Convert from Q::T to f32 for mimi
        let f32_transposed = transposed.to()?;
        let quantized = self.mimi.quantizer.forward(&f32_transposed)?;
        self.mimi.decode_from_latent_step(&quantized, mimi_state)
    }

    /// Initialize mimi streaming state.
    pub fn init_mimi_state(
        &self,
        batch_size: usize,
        context: usize,
    ) -> Result<MimiDecoderState<f32, Q::B>> {
        self.mimi.init_state(batch_size, context)
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState<Q>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        backbone_input_latents: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        let input = self.flow_lm.input_linear.forward(backbone_input_latents)?;
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let _out =
            self.flow_lm.transformer.forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
    }

    pub fn device(&self) -> &Q::B {
        self.flow_lm.input_linear.device()
    }
}

pub struct MimiEnc<Q: BackendQ> {
    speaker_proj: Option<Linear<Q::T, Q::B>>,
    mimi: MimiEncoder<Unquantized<f32, Q::B>>,
}

impl<Q: BackendQ> MimiEnc<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &TTSConfig) -> Result<Self> {
        let mimi = MimiEncoder::load(&vb.pp("mimi"), &cfg.mimi)?;
        let speaker_proj = if vb.contains("flow_lm.speaker_proj_weight") {
            let weights = vb
                .tensor("flow_lm.speaker_proj_weight", (cfg.flow_lm.d_model, cfg.mimi.dimension))?;
            Some(Linear::new(weights))
        } else {
            None
        };
        Ok(Self { speaker_proj, mimi })
    }

    /// Encode audio for voice conditioning. Returns [1, T', dim].
    pub fn encode_audio(&self, audio: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let f32_audio = audio.to::<f32>()?;
        let encoded = self.mimi.encode_to_latent(&f32_audio)?;
        // [B, C, T] -> [B, T, C]
        let latents = encoded.transpose(1, 2)?.contiguous()?;
        let latents = latents.to::<Q::T>()?;
        match self.speaker_proj.as_ref() {
            Some(p) => p.forward(&latents),
            None => Ok(latents),
        }
    }
}

pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Split text into sentence-aligned chunks that fit within a token budget.
///
/// This mirrors the Python `split_into_best_sentences` function: it prepares the text,
/// tokenizes it, finds sentence boundaries (after `.`, `!`, `...`, `?` tokens), then
/// greedily groups sentences into chunks of at most `max_tokens` tokens each.
pub fn split_into_best_sentences(
    tokenizer: &dyn crate::Tokenizer,
    text: &str,
    max_tokens: Option<usize>,
) -> Vec<String> {
    let max_tokens = max_tokens.unwrap_or(MAX_TOKENS_PER_CHUNK);
    let (prepared, _) = prepare_text_prompt(text);
    let prepared = prepared.trim().to_string();
    let tokens = tokenizer.encode(&prepared);

    // Get end-of-sentence token ids by tokenizing ".!...?" and skipping the first token
    // (the first token includes the leading space marker from sentencepiece).
    let eos_marker_tokens = tokenizer.encode(".!...?");
    let eos_tokens =
        if eos_marker_tokens.len() > 1 { &eos_marker_tokens[1..] } else { &eos_marker_tokens[..] };

    // Find sentence boundary indices: positions where a non-EOS token follows one or more EOS tokens.
    let mut sentence_boundaries = vec![0usize];
    let mut prev_was_eos = false;

    for (idx, &token) in tokens.iter().enumerate() {
        if eos_tokens.contains(&token) {
            prev_was_eos = true;
        } else {
            if prev_was_eos {
                sentence_boundaries.push(idx);
            }
            prev_was_eos = false;
        }
    }
    sentence_boundaries.push(tokens.len());

    // Build (token_count, sentence_text) pairs by decoding each token sub-range.
    let mut sentences = Vec::new();
    for window in sentence_boundaries.windows(2) {
        let (start, end) = (window[0], window[1]);
        let text = tokenizer.decode(&tokens[start..end]);
        sentences.push((end - start, text));
    }

    // Greedily group sentences into chunks that stay under max_tokens.
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    for (nb_tokens, sentence) in sentences {
        if current_chunk.is_empty() {
            current_chunk = sentence;
            current_token_count = nb_tokens;
            continue;
        }

        if current_token_count + nb_tokens > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence;
            current_token_count = nb_tokens;
        } else {
            current_chunk.push(' ');
            current_chunk.push_str(&sentence);
            current_token_count += nb_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Prepare text for generation: capitalize, add punctuation, pad short text.
pub fn prepare_text_prompt(text: &str) -> (String, usize) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    let text = text.replace(['\n', '\r'], " ");
    let mut text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos = if number_of_words <= 4 { 3 } else { 1 };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().to_string() + chars.as_str();
    }
    if text.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        text.push('.');
    }
    if text.split_whitespace().count() < 5 {
        text = format!("        {text}");
    }
    (text, frames_after_eos)
}
