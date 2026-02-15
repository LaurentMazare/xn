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

pub struct TTSModel<T: WithDTypeF, B: Backend> {
    pub flow_lm: FlowLM<T, B>,
    pub mimi: MimiModel<T, B>,
    speaker_proj: Option<Linear<T, B>>,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

pub struct TTSState<T: WithDTypeF, B: Backend> {
    pub flow_lm_state: FlowLMState<T, B>,
}

impl<T: WithDTypeF, B: Backend> TTSModel<T, B> {
    pub fn load(vb: &Path<B>, cfg: &TTSConfig) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), &cfg.flow_lm)?;
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
