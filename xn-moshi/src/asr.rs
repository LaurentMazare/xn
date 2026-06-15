use crate::conditioners::Conditioners;
use crate::lm::{LmModel, LmState};
use crate::mimi::{Mimi, MimiEncodeState};
use crate::moshi;
use crate::sampling::gumbel_max;
use xn::streaming::{StreamMask, StreamTensor};
use xn::{BackendQ, Result, Tensor, WithDTypeF};

const TOKEN_EOP: u32 = 0;
const TOKEN_PAD: u32 = 3;
const TOKEN_SILENCE_PAD: u32 = 4;

pub trait TokenBooster: 'static + Send + Sync {
    fn on_text_token(&mut self, text_token: u32);
    fn text_biases(&self) -> Vec<(u32, f32)>;
}

// ============================================================================
// Messages
// ============================================================================

#[derive(Debug, Clone)]
pub enum AsrWord {
    Word { tokens: Vec<u32>, start_time: f64, batch_idx: usize },
    EndWord { stop_time: f64, batch_idx: usize },
}

impl AsrWord {
    pub fn batch_idx(&self) -> usize {
        match self {
            AsrWord::Word { batch_idx, .. } | AsrWord::EndWord { batch_idx, .. } => *batch_idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub words: Vec<AsrWord>,
    pub prs: Vec<Vec<f32>>,
    // For the audio tokens, first dimension is batch, second is codebooks.
    pub audio_tokens: Vec<Vec<u32>>,
    pub text_tokens: Vec<u32>,
}

mod item_state {
    use super::*;
    pub struct ItemState {
        batch_idx: usize,
        step_idx: usize,
        text_token: u32,
        word_tokens: Vec<u32>,
        unended_word: bool,
        last_stop_time: f64,
        token_booster: Option<Box<dyn TokenBooster>>,
    }

    impl ItemState {
        pub fn new(batch_idx: usize, text_token: u32) -> Self {
            Self {
                batch_idx,
                step_idx: 0,
                text_token,
                word_tokens: vec![],
                unended_word: false,
                last_stop_time: 0.,
                token_booster: None,
            }
        }

        pub fn reset(&mut self, token_booster: Option<Box<dyn TokenBooster>>) {
            self.step_idx = 0;
            self.text_token = 0;
            self.word_tokens.clear();
            self.unended_word = false;
            self.last_stop_time = 0.;
            self.token_booster = token_booster;
        }

        pub fn text_token(&self) -> u32 {
            self.text_token
        }

        pub fn is_first_step(&self) -> bool {
            self.step_idx == 0
        }

        pub fn flush_tokens(&mut self) -> Option<AsrWord> {
            if !self.word_tokens.is_empty() {
                let mut tokens = vec![];
                std::mem::swap(&mut self.word_tokens, &mut tokens);
                let word = AsrWord::Word {
                    tokens,
                    start_time: self.last_stop_time,
                    batch_idx: self.batch_idx,
                };
                self.unended_word = true;
                Some(word)
            } else {
                None
            }
        }

        pub fn on_text_token(
            &mut self,
            text_token: u32,
            asr_delay_in_tokens: usize,
            words: &mut Vec<AsrWord>,
        ) {
            self.text_token = text_token;
            self.step_idx += 1;
            if self.step_idx >= asr_delay_in_tokens {
                if let Some(tb) = self.token_booster.as_mut() {
                    tb.on_text_token(text_token);
                }
                if text_token == TOKEN_PAD
                    || text_token == TOKEN_EOP
                    || text_token == TOKEN_SILENCE_PAD
                {
                    if let Some(word) = self.flush_tokens() {
                        words.push(word)
                    }
                } else {
                    self.word_tokens.push(self.text_token);
                }
                if self.text_token == TOKEN_EOP {
                    let stop_time = (self.step_idx - asr_delay_in_tokens) as f64 / 12.5;
                    if self.unended_word {
                        self.unended_word = false;
                        words.push(AsrWord::EndWord { stop_time, batch_idx: self.batch_idx });
                    }
                    self.last_stop_time = stop_time;
                }
            }
        }

        pub fn with_text_bias(&self) -> bool {
            self.token_booster.is_some()
        }

        pub fn text_biases(&self, asr_delay_in_tokens: usize) -> Vec<(u32, f32)> {
            if self.step_idx < asr_delay_in_tokens {
                return vec![];
            }
            match &self.token_booster {
                None => vec![],
                Some(tb) => tb.text_biases(),
            }
        }
    }
}
use item_state::ItemState;

pub struct AsrState<Q: BackendQ> {
    model: Asr<Q>,
    pub lm: LmState<Q>,
    pub audio_tokenizer: MimiEncodeState<f32, Q::B>,
    pub batch: Vec<ItemState>,
    model_step_idx: usize,
    temperature: Tensor<f32, Q::B>,
    condition: Option<Tensor<Q::T, Q::B>>,
}

#[derive(Clone)]
pub struct Asr<Q: BackendQ> {
    asr_delay_in_tokens: usize,
    default_temperature: f64,
    lm: std::sync::Arc<LmModel<Q>>,
    audio_tokenizer: std::sync::Arc<Mimi<f32, Q::B>>,
    conditioners: Option<std::sync::Arc<Conditioners<Q::T, Q::B>>>,
    default_condition: Option<Tensor<Q::T, Q::B>>,
}

impl<Q: BackendQ> Asr<Q> {
    pub fn new(
        asr_delay_in_tokens: usize,
        default_temperature: f64,
        audio_tokenizer: Mimi<f32, Q::B>,
        lm: LmModel<Q>,
    ) -> Self {
        Self {
            asr_delay_in_tokens,
            default_temperature,
            lm: std::sync::Arc::new(lm),
            audio_tokenizer: std::sync::Arc::new(audio_tokenizer),
            conditioners: None,
            default_condition: None,
        }
    }

    pub fn conditioners(&self) -> Option<&Conditioners<Q::T, Q::B>> {
        self.conditioners.as_deref()
    }

    pub fn init_state(&self, batch_size: usize) -> Result<AsrState<Q>> {
        let text_start_token = self.lm.text_start_token();
        let batch =
            (0..batch_size).map(|batch_idx| ItemState::new(batch_idx, text_start_token)).collect();
        let temperature =
            Tensor::full(self.default_temperature as f32, (batch_size, 1), self.device())?;
        let condition = match self.default_condition.as_ref() {
            None => None,
            Some(c) => {
                let c = c.expand((batch_size, 1, c.dim(xn::D::Minus1)?))?;
                // TODO(laurent): add to xn a contiguous function that ensures a copy is made.
                Some(c.contiguous()?.copy()?)
            }
        };
        Ok(AsrState {
            model: self.clone(),
            lm: self.lm.init_state(batch_size)?,
            audio_tokenizer: self.audio_tokenizer.init_encode_state(batch_size)?,
            batch,
            model_step_idx: 0,
            temperature,
            condition,
        })
    }

    pub fn device(&self) -> &Q::B {
        self.lm.device()
    }

    pub fn asr_delay_in_tokens(&self) -> usize {
        self.asr_delay_in_tokens
    }

    pub fn warmup(&self, frame_size: usize) -> Result<()> {
        let mut state = self.init_state(1)?;
        let audio: Tensor<f32, _> = Tensor::zeros((1, 1, frame_size), self.device())?;
        let pcm = StreamTensor::from_tensor(audio);
        let mask = StreamMask::all_active(1);
        for _ in 0..3 {
            let _ = state.step_pcm(&pcm, &mask, |_, _, _| {})?;
        }
        Ok(())
    }
}

/// Loading from weight files.
impl<Q: BackendQ> Asr<Q> {
    pub fn load(
        mimi_weight: &str,
        lm_weight: &str,
        config: Option<&str>,
        asr_delay_in_tokens: usize,
        default_temperature: f64,
        dev: Q::B,
    ) -> Result<Self> {
        Self::load_with_mimi_config(
            mimi_weight,
            lm_weight,
            config,
            crate::mimi::Config::v0_1(Some(32)),
            asr_delay_in_tokens,
            default_temperature,
            dev,
        )
    }

    pub fn load_with_mimi_config(
        mimi_weight: &str,
        lm_weight: &str,
        config: Option<&str>,
        mimi_config: crate::mimi::Config,
        asr_delay_in_tokens: usize,
        default_temperature: f64,
        dev: Q::B,
    ) -> Result<Self> {
        use crate::lm;
        use xn::nn::VB;

        let moshi_config = match config {
            Some(c) => {
                let c = std::fs::read_to_string(c)
                    .map_err(|e| xn::Error::Msg(format!("reading config {c}: {e}")))?;
                let config = serde_json::from_str::<moshi::Config>(&c)
                    .map_err(|e| xn::Error::Msg(format!("parsing config: {e}")))?;
                Some(config)
            }
            None => None,
        };
        let lm_config = match &moshi_config {
            Some(c) => c.to_lm_config(),
            None => lm::Config::stt_2_6b(),
        };

        let mimi_vb = VB::load(&[mimi_weight], dev.clone())?.root();
        let mimi_model: Mimi<f32, Q::B> = Mimi::load(&mimi_vb, mimi_config)?;
        mimi_vb.check_all_used_with_ignore(|s| {
            s.ends_with("_codebook._initialized")
                || s.ends_with("_codebook.cluster_usage")
                || s.ends_with("_codebook.embedding_sum")
        })?;

        let lm_vb = VB::load(&[lm_weight], dev)?.root();
        let lm_model: LmModel<Q> = LmModel::load(&lm_vb, &lm_config)?;
        let conditioners = match &moshi_config {
            Some(c) => Some(c.load_conditioners::<Q::T, _>(&lm_vb)?),
            None => None,
        };
        lm_vb.check_all_used_with_ignore(|s| s.starts_with("linears."))?;

        let default_condition = match conditioners.as_ref() {
            Some(conds) => {
                let delay = -0.08 * asr_delay_in_tokens as f64;
                let values = std::collections::HashMap::from([("delay".to_string(), delay.into())]);
                conds.condition_sum(&values)?
            }
            None => None,
        };

        Ok(Self {
            asr_delay_in_tokens,
            default_temperature,
            lm: std::sync::Arc::new(lm_model),
            audio_tokenizer: std::sync::Arc::new(mimi_model),
            conditioners: conditioners.map(std::sync::Arc::new),
            default_condition,
        })
    }
}

impl<Q: BackendQ> AsrState<Q> {
    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

    pub fn flush_tokens(&mut self, batch_idx: usize) -> Result<Option<AsrWord>> {
        if batch_idx >= self.batch.len() {
            xn::bail!("unexpected batch idx {batch_idx}")
        }
        Ok(self.batch[batch_idx].flush_tokens())
    }

    pub fn reset_state(&mut self) -> Result<()> {
        self.batch.iter_mut().for_each(|s| s.reset(None));
        self.model_step_idx = 0;
        let batch_size = self.batch.len();
        self.lm = self.model.lm.init_state(batch_size)?;
        self.audio_tokenizer = self.model.audio_tokenizer.init_encode_state(batch_size)?;
        Ok(())
    }

    pub fn device(&self) -> &Q::B {
        self.model.device()
    }

    pub fn condition_sum(
        &self,
        lang: Option<&str>,
        target_lang: Option<&str>,
        delay: f64,
    ) -> Result<Option<Tensor<Q::T, Q::B>>> {
        let conds = match self.model.conditioners.as_ref() {
            Some(conds) => {
                let mut values =
                    std::collections::HashMap::from([("delay".to_string(), delay.into())]);
                if let Some(lang) = lang {
                    values.insert("lang".to_string(), lang.into());
                    values.insert("languages_in_segment".to_string(), lang.into());
                }
                if let Some(target_lang) = target_lang {
                    values.insert("target_language".to_string(), target_lang.into());
                }
                conds.condition_sum(&values)?
            }
            None => None,
        };
        Ok(conds)
    }

    pub fn step_pcm<F>(
        &mut self,
        pcm: &StreamTensor<f32, Q::B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<StepResult>>
    where
        F: Fn(&[ItemState], &[u32], &[Vec<u32>]),
    {
        let audio_tokens = self.audio_tokenizer.encode_step(pcm, mask)?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, mask, f)
        } else {
            Ok(vec![])
        }
    }

    fn text_tokens(&self) -> Vec<u32> {
        let text_start_token = self.model.lm.text_start_token();
        self.batch
            .iter()
            .map(|s| if s.is_first_step() { text_start_token } else { s.text_token() })
            .collect()
    }

    fn text_bias(&self, asr_delay_in_tokens: usize) -> Result<Option<Tensor<Q::T, Q::B>>> {
        let with_text_bias = self.batch.iter().any(|s| s.with_text_bias());
        if !with_text_bias {
            return Ok(None);
        }
        let batch_size = self.batch.len();
        let vocab_size = self.model.lm.text_out_vocab_size();
        let mut bias = vec![Q::T::from_f32(0.0); batch_size * vocab_size];
        for (batch_idx, item) in self.batch.iter().enumerate() {
            for (token_id, token_bias) in item.text_biases(asr_delay_in_tokens) {
                let token_id = token_id as usize;
                if token_id < vocab_size {
                    bias[batch_idx * vocab_size + token_id] = Q::T::from_f32(token_bias);
                }
            }
        }
        let bias = Tensor::from_vec(bias, (batch_size, vocab_size), self.device())?;
        Ok(Some(bias))
    }

    /// Process audio tokens (shape: batch, codebooks, steps as i64) and return ASR messages.
    pub fn step_tokens<F>(
        &mut self,
        audio_tokens: &Tensor<i64, Q::B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<StepResult>>
    where
        F: Fn(&[ItemState], &[u32], &[Vec<u32>]),
    {
        let (batch_size, codebooks, steps) = audio_tokens.dims3()?;
        if batch_size != self.batch.len() {
            xn::bail!("batch size mismatch: {batch_size} != {}", self.batch.len());
        }

        // Pull all audio tokens to CPU once.
        let all_audio_tokens: Vec<i64> = audio_tokens.to_vec()?;
        // Layout: [batch][codebook][step] in row-major = batch * codebooks * steps

        let mut step_results = vec![];
        for step in 0..steps {
            // Extract tokens for this step: audio_tokens[:, :, step]
            let audio_tokens: Vec<Vec<u32>> = (0..batch_size)
                .map(|b| {
                    (0..codebooks)
                        .map(|cb| {
                            all_audio_tokens[b * codebooks * steps + cb * steps + step] as u32
                        })
                        .collect()
                })
                .collect();

            // Build per-codebook token vectors with next_token logic
            let audio_ids: Vec<Vec<u32>> = (0..codebooks)
                .map(|codebook_idx| {
                    audio_tokens
                        .iter()
                        .zip(self.batch.iter_mut())
                        .enumerate()
                        .map(|(batch_idx, (tokens, item))| {
                            if !mask.is_active(batch_idx) {
                                0u32
                            } else if item.is_first_step() {
                                // The first slice is dropped and replace with pad tokens. Note
                                // that we do not shift the audio slices and just discard this
                                // first slice.
                                self.model.lm.audio_pad_token()
                            } else {
                                tokens[codebook_idx]
                            }
                        })
                        .collect()
                })
                .collect();

            let text_tokens = self.text_tokens();

            f(self.batch.as_slice(), &text_tokens, &audio_ids);

            // Build audio_ids as slices for the LM forward pass
            let audio_id_refs: Vec<Option<&[u32]>> =
                audio_ids.iter().map(|ids| Some(ids.as_slice())).collect();

            let (text_logits, transformer_out) = self.lm.forward(
                Some(&text_tokens),
                &audio_id_refs,
                mask,
                self.condition.as_ref(),
            )?;
            self.model_step_idx += 1;
            // We compute self.text_bias as early as possible so that the CPU
            // computations take place while the cuda async computations are running.
            let text_bias = self.text_bias(self.model.asr_delay_in_tokens)?;

            // Extra heads
            let extra_heads = self.lm.extra_heads(&transformer_out)?;
            let mut prs = vec![];
            for extra_head in extra_heads.iter() {
                // softmax on last dim, shape (batch, 1, dim) -> take (:, 0, 0)
                let eh = extra_head.softmax()?;
                let eh_data: Vec<Q::T> = eh.to_vec()?;
                let eh_dims = eh.dims();
                let dim = eh_dims[2];
                // Extract first element per batch (index 0 of seq=0)
                let prs_: Vec<f32> = (0..batch_size)
                    .map(|b| <Q::T as WithDTypeF>::to_f32(eh_data[b * dim]))
                    .collect();
                prs.push(prs_);
            }

            // Sample text tokens
            // text_logits shape: (batch, 1, text_out_vocab_size)
            let (batch_size, _one, vocab_size) = text_logits.dims3()?;
            let logits_2d = text_logits.reshape((batch_size, vocab_size))?;
            let logits_2d = if let Some(text_bias) = text_bias.as_ref() {
                logits_2d.add(text_bias)?
            } else {
                logits_2d
            };
            let sampled_tokens = gumbel_max(&logits_2d, &self.temperature)?;
            let mut words = vec![];
            for (batch_idx, (text_token, item)) in
                sampled_tokens.to_vec()?.into_iter().zip(self.batch.iter_mut()).enumerate()
            {
                if !mask.is_active(batch_idx) {
                    continue;
                }
                item.on_text_token(text_token as u32, self.model.asr_delay_in_tokens, &mut words);
            }
            let step_result = StepResult { words, prs, audio_tokens, text_tokens };
            step_results.push(step_result);
        }
        Ok(step_results)
    }

    pub fn reset_batch_idx(
        &mut self,
        batch_idx: usize,
        temp: Option<f64>,
        cond: Option<&Tensor<Q::T, Q::B>>,
        token_booster: Option<Box<dyn TokenBooster>>,
    ) -> Result<()> {
        if batch_idx >= self.batch.len() {
            xn::bail!("batch index out of range: {batch_idx} >= {}", self.batch.len());
        }
        self.batch[batch_idx].reset(token_booster);
        self.lm.reset_batch_idx(batch_idx)?;
        self.audio_tokenizer.reset_batch_idx(batch_idx)?;
        let temp = temp.unwrap_or(self.model.default_temperature) as f32;
        let temp = Tensor::full(temp, (1, 1), self.device())?;
        self.temperature.slice_set(&temp, 0, batch_idx)?;
        let cond = match cond {
            None => self.model.default_condition.as_ref(),
            Some(_) => cond,
        };
        if let Some(batch_cond) = self.condition.as_mut()
            && let Some(c) = cond
        {
            batch_cond.slice_set(c, 0, batch_idx)?;
        }
        Ok(())
    }
}
