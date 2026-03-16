use crate::lm::{LmModel, LmState};
use crate::mimi::{Mimi, MimiEncodeState};
use xn::streaming::{StreamMask, StreamTensor};
use xn::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// Messages
// ============================================================================

#[derive(Debug, Clone)]
pub enum AsrMsg {
    Step {
        step_idx: usize,
        prs: Vec<Vec<f32>>,
    },
    Word {
        tokens: Vec<u32>,
        start_time: f64,
        batch_idx: usize,
    },
    EndWord {
        stop_time: f64,
        batch_idx: usize,
    },
}

// ============================================================================
// Per-batch-element state
// ============================================================================

#[derive(Debug, Clone)]
pub struct ItemState {
    step_idx: usize,
    text_token: u32,
    word_tokens: Vec<u32>,
    unended_word: bool,
    last_stop_time: f64,
    audio_pad_token: u32,
    next_codebooks: Vec<u32>,
}

impl ItemState {
    fn reset(&mut self) {
        self.step_idx = 0;
        self.text_token = 0;
        self.word_tokens.clear();
        self.unended_word = false;
        self.last_stop_time = 0.;
        self.next_codebooks.fill(self.audio_pad_token);
    }

    pub fn text_token(&self) -> u32 {
        self.text_token
    }

    pub fn is_first_step(&self) -> bool {
        self.step_idx == 0
    }

    pub fn next_token(&mut self, codebook_idx: usize, token: u32) -> u32 {
        let v = self.next_codebooks[codebook_idx];
        self.next_codebooks[codebook_idx] = token;
        if self.is_first_step() {
            self.audio_pad_token
        } else {
            v
        }
    }
}

// ============================================================================
// ASR State
// ============================================================================

pub struct AsrState<MimiT: WithDTypeF, LmT: WithDTypeF, B: Backend> {
    model: Asr<MimiT, LmT, B>,
    pub lm: LmState<LmT, B>,
    pub audio_tokenizer: MimiEncodeState<MimiT, B>,
    pub batch: Vec<ItemState>,
    model_step_idx: usize,
}

#[derive(Clone)]
pub struct Asr<MimiT: WithDTypeF, LmT: WithDTypeF, B: Backend> {
    asr_delay_in_tokens: usize,
    temperature: f64,
    lm: std::sync::Arc<LmModel<LmT, B>>,
    audio_tokenizer: std::sync::Arc<Mimi<MimiT, B>>,
}

impl<MimiT: WithDTypeF, LmT: WithDTypeF, B: Backend> Asr<MimiT, LmT, B> {
    pub fn new(
        asr_delay_in_tokens: usize,
        temperature: f64,
        audio_tokenizer: Mimi<MimiT, B>,
        lm: LmModel<LmT, B>,
    ) -> Self {
        Self {
            asr_delay_in_tokens,
            temperature,
            lm: std::sync::Arc::new(lm),
            audio_tokenizer: std::sync::Arc::new(audio_tokenizer),
        }
    }

    pub fn init_state(&self, batch_size: usize) -> Result<AsrState<MimiT, LmT, B>> {
        let text_start_token = self.lm.text_start_token();
        let audio_pad_token = self.lm.audio_pad_token();
        let in_audio_codebooks = self.lm.in_audio_codebooks();

        let item_state = ItemState {
            text_token: text_start_token,
            word_tokens: vec![],
            unended_word: false,
            step_idx: 0,
            last_stop_time: 0.,
            audio_pad_token,
            next_codebooks: vec![audio_pad_token; in_audio_codebooks],
        };

        Ok(AsrState {
            model: self.clone(),
            lm: self.lm.init_state(batch_size)?,
            audio_tokenizer: self.audio_tokenizer.init_encode_state(batch_size)?,
            batch: vec![item_state; batch_size],
            model_step_idx: 0,
        })
    }

    pub fn device(&self) -> &B {
        self.lm.device()
    }

    pub fn asr_delay_in_tokens(&self) -> usize {
        self.asr_delay_in_tokens
    }
}

impl<MimiT: WithDTypeF, LmT: WithDTypeF, B: Backend> AsrState<MimiT, LmT, B> {
    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

    pub fn reset_state(&mut self) -> Result<()> {
        self.batch.iter_mut().for_each(|s| s.reset());
        self.model_step_idx = 0;
        let batch_size = self.batch.len();
        self.lm = self.model.lm.init_state(batch_size)?;
        self.audio_tokenizer = self.model.audio_tokenizer.init_encode_state(batch_size)?;
        Ok(())
    }

    pub fn step_pcm<F>(
        &mut self,
        pcm: &StreamTensor<MimiT, B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
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
            .map(|s| {
                if s.is_first_step() {
                    text_start_token
                } else {
                    s.text_token()
                }
            })
            .collect()
    }

    /// Process audio tokens (shape: batch, codebooks, steps as i64) and return ASR messages.
    pub fn step_tokens<F>(
        &mut self,
        audio_tokens: &Tensor<i64, B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], &[u32], &[Vec<u32>]),
    {
        let dims = audio_tokens.dims();
        let (batch_size, codebooks, steps) = (dims[0], dims[1], dims[2]);
        if batch_size != self.batch.len() {
            xn::bail!("batch size mismatch: {batch_size} != {}", self.batch.len());
        }

        // Pull all audio tokens to CPU once.
        let all_audio_tokens: Vec<i64> = audio_tokens.to_vec()?;
        // Layout: [batch][codebook][step] in row-major = batch * codebooks * steps

        let mut words = vec![];
        for step in 0..steps {
            // Extract tokens for this step: audio_tokens[:, :, step]
            let audio_tokens_step: Vec<Vec<u32>> = (0..batch_size)
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
                    audio_tokens_step
                        .iter()
                        .zip(self.batch.iter_mut())
                        .enumerate()
                        .map(|(batch_idx, (tokens, item))| {
                            if !mask.is_active(batch_idx) {
                                0u32
                            } else {
                                item.next_token(codebook_idx, tokens[codebook_idx])
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

            let (text_logits, transformer_out) =
                self.lm.forward(Some(&text_tokens), &audio_id_refs, mask)?;

            self.model_step_idx += 1;

            // Extra heads
            let extra_heads = self.lm.extra_heads(&transformer_out)?;
            let mut prs = vec![];
            for extra_head in extra_heads.iter() {
                // softmax on last dim, shape (batch, 1, dim) -> take (:, 0, 0)
                let eh = extra_head.softmax()?;
                let eh_data: Vec<LmT> = eh.to_vec()?;
                let eh_dims = eh.dims();
                let dim = eh_dims[2];
                // Extract first element per batch (index 0 of seq=0)
                let prs_: Vec<f32> = (0..batch_size)
                    .map(|b| <LmT as WithDTypeF>::to_f32(eh_data[b * dim]))
                    .collect();
                prs.push(prs_);
            }
            if !prs.is_empty() {
                words.push(AsrMsg::Step {
                    step_idx: self.model_step_idx,
                    prs,
                });
            }

            // Sample text tokens
            // text_logits shape: (batch, 1, text_out_vocab_size)
            let (batch_size, _one, vocab_size) = text_logits.dims3()?;
            let logits_2d = text_logits.reshape((batch_size, vocab_size))?;
            let sampled_tokens = if self.model.temperature <= 0.0 {
                logits_2d.argmax(1)?
            } else {
                xn::nn::sampling::gumbel_softmax(
                    &logits_2d,
                    self.model.temperature as f32,
                    xn::D::Minus1,
                )?
            };

            for (batch_idx, (text_token, item)) in sampled_tokens
                .to_vec()?
                .into_iter()
                .zip(self.batch.iter_mut())
                .enumerate()
            {
                if !mask.is_active(batch_idx) {
                    continue;
                }
                item.text_token = text_token as u32;
                item.step_idx += 1;
                if item.step_idx >= self.model.asr_delay_in_tokens {
                    if text_token == 3 || text_token == 0 {
                        if !item.word_tokens.is_empty() {
                            let mut tokens = vec![];
                            std::mem::swap(&mut item.word_tokens, &mut tokens);
                            words.push(AsrMsg::Word {
                                tokens,
                                start_time: item.last_stop_time,
                                batch_idx,
                            });
                            item.unended_word = true;
                        }
                    } else {
                        item.word_tokens.push(item.text_token);
                    }
                    if item.text_token == 0 {
                        let stop_time =
                            (item.step_idx - self.model.asr_delay_in_tokens) as f64 / 12.5;
                        if item.unended_word {
                            item.unended_word = false;
                            words.push(AsrMsg::EndWord {
                                stop_time,
                                batch_idx,
                            });
                        }
                        item.last_stop_time = stop_time;
                    }
                }
            }
        }
        Ok(words)
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        if batch_idx >= self.batch.len() {
            xn::bail!(
                "batch index out of range: {batch_idx} >= {}",
                self.batch.len()
            );
        }
        self.batch[batch_idx].reset();
        self.lm.reset_batch_idx(batch_idx)?;
        Ok(())
    }
}
