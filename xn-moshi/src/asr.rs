use crate::lm::{LmModel, LmState};
use crate::mimi::{Mimi, MimiEncodeState};
use crate::streaming::{StreamMask, StreamTensor};
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

pub struct AsrState<T: WithDTypeF, B: Backend> {
    pub lm: LmState<T, B>,
    pub audio_tokenizer: MimiEncodeState<T, B>,
    pub batch: Vec<ItemState>,
    model_step_idx: usize,
}

pub struct Asr<T: WithDTypeF, B: Backend> {
    asr_delay_in_tokens: usize,
    temperature: f64,
    lm: LmModel<T, B>,
    audio_tokenizer: Mimi<T, B>,
}

impl<T: WithDTypeF, B: Backend> Asr<T, B> {
    pub fn new(
        asr_delay_in_tokens: usize,
        temperature: f64,
        audio_tokenizer: Mimi<T, B>,
        lm: LmModel<T, B>,
    ) -> Self {
        Self {
            asr_delay_in_tokens,
            temperature,
            lm,
            audio_tokenizer,
        }
    }

    pub fn init_state(&self, batch_size: usize) -> Result<AsrState<T, B>> {
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

    pub fn model_step_idx(&self, state: &AsrState<T, B>) -> usize {
        state.model_step_idx
    }

    pub fn reset_state(&self, state: &mut AsrState<T, B>) -> Result<()> {
        state.batch.iter_mut().for_each(|s| s.reset());
        state.model_step_idx = 0;
        let batch_size = state.batch.len();
        state.lm = self.lm.init_state(batch_size)?;
        state.audio_tokenizer = self.audio_tokenizer.init_encode_state(batch_size)?;
        Ok(())
    }

    pub fn step_pcm<F>(
        &self,
        pcm: &StreamTensor<T, B>,
        state: &mut AsrState<T, B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], &[u32], &[Vec<u32>]),
    {
        let audio_tokens =
            self.audio_tokenizer
                .encode_step(pcm, &mut state.audio_tokenizer, mask)?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, state, mask, f)
        } else {
            Ok(vec![])
        }
    }

    fn text_tokens(&self, state: &AsrState<T, B>) -> Vec<u32> {
        let text_start_token = self.lm.text_start_token();
        state
            .batch
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
        &self,
        audio_tokens: &Tensor<i64, B>,
        state: &mut AsrState<T, B>,
        mask: &StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], &[u32], &[Vec<u32>]),
    {
        let dims = audio_tokens.dims();
        let (batch_size, codebooks, steps) = (dims[0], dims[1], dims[2]);
        if batch_size != state.batch.len() {
            xn::bail!("batch size mismatch: {batch_size} != {}", state.batch.len());
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
                        .zip(state.batch.iter_mut())
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

            let text_tokens = self.text_tokens(state);

            f(state.batch.as_slice(), &text_tokens, &audio_ids);

            // Build audio_ids as slices for the LM forward pass
            let audio_id_refs: Vec<Option<&[u32]>> =
                audio_ids.iter().map(|ids| Some(ids.as_slice())).collect();

            let (text_logits, transformer_out) =
                self.lm
                    .forward(Some(&text_tokens), &audio_id_refs, &mut state.lm, mask)?;

            state.model_step_idx += 1;

            // Extra heads
            let extra_heads = self.lm.extra_heads(&transformer_out)?;
            let mut prs = vec![];
            for extra_head in extra_heads.iter() {
                // softmax on last dim, shape (batch, 1, dim) -> take (:, 0, 0)
                let eh = extra_head.softmax()?;
                let eh_data: Vec<T> = eh.to_vec()?;
                let eh_dims = eh.dims();
                let dim = eh_dims[2];
                // Extract first element per batch (index 0 of seq=0)
                let prs_: Vec<f32> = (0..batch_size)
                    .map(|b| <T as WithDTypeF>::to_f32(eh_data[b * dim]))
                    .collect();
                prs.push(prs_);
            }
            if !prs.is_empty() {
                words.push(AsrMsg::Step {
                    step_idx: state.model_step_idx,
                    prs,
                });
            }

            // Sample text tokens
            // text_logits shape: (batch, 1, text_out_vocab_size)
            let logits_dims = text_logits.dims();
            let vocab_size = logits_dims[2];
            let logits_2d = text_logits.reshape((logits_dims[0], vocab_size))?;
            let logits_data: Vec<T> = logits_2d.to_vec()?;

            let sampled_tokens = if self.temperature <= 0.0 {
                // Greedy: argmax over last dim on CPU
                (0..batch_size)
                    .map(|b| {
                        let start = b * vocab_size;
                        let slice = &logits_data[start..start + vocab_size];
                        argmax(slice)
                    })
                    .collect::<Vec<u32>>()
            } else {
                // Gumbel softmax sampling on CPU
                (0..batch_size)
                    .map(|b| {
                        let start = b * vocab_size;
                        let slice = &logits_data[start..start + vocab_size];
                        gumbel_argmax(slice, self.temperature)
                    })
                    .collect::<Vec<u32>>()
            };

            for (batch_idx, (text_token, item)) in sampled_tokens
                .into_iter()
                .zip(state.batch.iter_mut())
                .enumerate()
            {
                if !mask.is_active(batch_idx) {
                    continue;
                }
                item.text_token = text_token;
                item.step_idx += 1;
                if item.step_idx >= self.asr_delay_in_tokens {
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
                        let stop_time = (item.step_idx - self.asr_delay_in_tokens) as f64 / 12.5;
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

    pub fn reset_batch_idx(&self, state: &mut AsrState<T, B>, batch_idx: usize) -> Result<()> {
        if batch_idx >= state.batch.len() {
            xn::bail!(
                "batch index out of range: {batch_idx} >= {}",
                state.batch.len()
            );
        }
        state.batch[batch_idx].reset();
        self.lm.reset_batch_idx(&mut state.lm, batch_idx)?;
        Ok(())
    }
}

/// Greedy argmax.
fn argmax<T: WithDTypeF>(logits: &[T]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = <T as WithDTypeF>::to_f32(logits[0]);
    for (i, v) in logits.iter().enumerate().skip(1) {
        let vf = <T as WithDTypeF>::to_f32(*v);
        if vf > best_val {
            best_val = vf;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Gumbel-max trick for sampling from logits.
fn gumbel_argmax<T: WithDTypeF>(logits: &[T], temperature: f64) -> u32 {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut best_idx = 0u32;
    let mut best_val = f64::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        let u: f64 = loop {
            let u: f64 = rng.random();
            if u > 0.0 && u < 1.0 {
                break u;
            }
        };
        let gumbel = -(-u.ln()).ln();
        let score = <T as WithDTypeF>::to_f32(*v) as f64 / temperature + gumbel;
        if score > best_val {
            best_val = score;
            best_idx = i as u32;
        }
    }
    best_idx
}
