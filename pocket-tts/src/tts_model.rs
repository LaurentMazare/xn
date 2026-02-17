use crate::flow_lm::{FlowLM, FlowLMConfig, FlowLMState};
use crate::mimi::{MimiConfig, MimiModel, MimiState};
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
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
                frame_rate: 12,
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
            },
            temp,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
        }
    }
}

pub struct TTSModel<T: WithDTypeF, B: Backend> {
    pub flow_lm: FlowLM<T, B>,
    pub mimi: MimiModel<T, B>,
    speaker_proj: Option<Linear<T, B>>,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

#[derive(Clone, Debug)]
pub struct TTSState<T: WithDTypeF, B: Backend> {
    pub flow_lm_state: FlowLMState<T, B>,
}

impl<T: WithDTypeF, B: Backend> TTSModel<T, B> {
    pub fn load(
        vb: &Path<B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, &cfg.flow_lm)?;
        let mimi = MimiModel::load(&vb.pp("mimi"), &cfg.mimi)?;

        let speaker_proj = if vb.contains("flow_lm.speaker_proj_weight") {
            let weights = vb.tensor("flow_lm.speaker_proj_weight", (1024, 512))?;
            Some(Linear::new(weights))
        } else {
            None
        };

        Ok(Self {
            flow_lm,
            mimi,
            speaker_proj,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.mimi.sample_rate
    }

    /// Initialize flow LM state with the given sequence length budget.
    pub fn init_flow_lm_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<TTSState<T, B>> {
        Ok(TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? })
    }

    /// Encode audio for voice conditioning. Returns [1, T', dim].
    pub fn encode_audio(&self, audio: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let encoded = self.mimi.encode_to_latent(audio)?;
        // [B, C, T] -> [B, T, C]
        let latents = encoded.transpose(1, 2)?.contiguous()?;
        match self.speaker_proj.as_ref() {
            Some(p) => p.forward(&latents),
            None => Ok(latents),
        }
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState<T, B>, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with audio conditioning. Increments state.
    pub fn prompt_audio(
        &self,
        state: &mut TTSState<T, B>,
        audio_conditioning: &Tensor<T, B>,
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
    pub fn generate_step(
        &self,
        state: &mut TTSState<T, B>,
        backbone_input: &Tensor<T, B>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<T, B>, bool)> {
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

    /// Decode latent to audio using mimi (streaming).
    pub fn decode_latent(
        &self,
        latent: &Tensor<T, B>,
        mimi_state: &mut MimiState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let denorm =
            latent.broadcast_mul(&self.flow_lm.emb_std)?.broadcast_add(&self.flow_lm.emb_mean)?;

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        let quantized = self.mimi.quantizer.forward(&transposed)?;
        self.mimi.decode_from_latent(&quantized, mimi_state)
    }

    /// Initialize mimi streaming state.
    pub fn init_mimi_state(&self, batch_size: usize, context: usize) -> Result<MimiState<T, B>> {
        self.mimi.init_state(batch_size, context)
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState<T, B>,
        text_embeddings: &Tensor<T, B>,
        backbone_input_latents: &Tensor<T, B>,
    ) -> Result<()> {
        let input = self.flow_lm.input_linear.forward(backbone_input_latents)?;
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let _out =
            self.flow_lm.transformer.forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
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
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

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
