use crate::batched_transformer::{self as bt, BatchedTransformerState};
use crate::streaming::StreamMask;
use crate::transformer::{self, Config as TransformerConfig, Norm};
use xn::nn::{var_builder::Path, Embedding, Linear};
use xn::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone)]
pub struct Config {
    pub transformer: TransformerConfig,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
    pub extra_heads: Option<ExtraHeadsConfig>,
}

#[derive(Debug, Clone)]
pub struct ExtraHeadsConfig {
    pub num_heads: usize,
    pub dim: usize,
}

impl Config {
    pub fn asr_v0_1_1b() -> Self {
        let transformer = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 2048 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000,
            use_conv_block: false,
            conv_kernel_size: 3,
            use_conv_bias: true,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 8,
            extra_heads: None,
        }
    }

    pub fn stt_2_6b() -> Self {
        let transformer = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 8448, // 2048 * 4.125
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 375,
            max_period: 100_000,
            use_conv_block: false,
            conv_kernel_size: 3,
            use_conv_bias: true,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 4001,
            text_out_vocab_size: 4000,
            audio_codebooks: 32,
            extra_heads: None,
        }
    }

    pub fn asr_300m_202501() -> Self {
        let transformer = TransformerConfig {
            d_model: 1024,
            num_heads: 8,
            num_layers: 16,
            dim_feedforward: 1024 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000,
            use_conv_block: false,
            conv_kernel_size: 3,
            use_conv_bias: true,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 32,
            extra_heads: None,
        }
    }
}

// ============================================================================
// State
// ============================================================================

pub struct LmState<T: WithDTypeF, B: Backend> {
    pub transformer: BatchedTransformerState<T, B>,
}

// ============================================================================
// LmModel
// ============================================================================

pub struct LmModel<T: WithDTypeF, B: Backend> {
    transformer: bt::BatchedTransformer<T, B>,
    text_emb: Embedding<T, B>,        // (text_in_vocab_size, d_model)
    audio_embs: Vec<Embedding<T, B>>, // each (audio_vocab_size, d_model)
    text_linear: Linear<T, B>,        // (text_out_vocab_size, d_model)
    out_norm: Norm<T, B>,
    extra_heads: Vec<Linear<T, B>>, // each (dim, d_model)
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
}

impl<T: WithDTypeF, B: Backend> LmModel<T, B> {
    pub fn load(vb: &Path<B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.transformer.d_model;

        let text_emb = Embedding::load(vb.pp("text_emb"), cfg.text_in_vocab_size, d_model)?;
        let out_norm = Norm::load(vb.pp("out_norm"), d_model, cfg.transformer.norm)?;
        let text_linear = Linear::load(vb.pp("text_linear"), d_model, cfg.text_out_vocab_size)?;

        let transformer = bt::BatchedTransformer::load(&vb.pp("transformer"), &cfg.transformer)?;

        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            let emb = Embedding::load(vb_e.pp(i), cfg.audio_vocab_size, d_model)?;
            audio_embs.push(emb);
        }

        let mut extra_heads = vec![];
        if let Some(ExtraHeadsConfig { num_heads, dim }) = &cfg.extra_heads {
            for i in 0..*num_heads {
                let head = Linear::load(vb.pp("extra_heads").pp(i), d_model, *dim)?;
                extra_heads.push(head);
            }
        }

        Ok(Self {
            transformer,
            text_emb,
            audio_embs,
            text_linear,
            out_norm,
            extra_heads,
            audio_vocab_size: cfg.audio_vocab_size,
            text_in_vocab_size: cfg.text_in_vocab_size,
        })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<LmState<T, B>> {
        Ok(LmState {
            transformer: self.transformer.init_state(batch_size)?,
        })
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    pub fn device(&self) -> &B {
        self.text_emb.device()
    }

    /// Forward pass returning (text_logits, transformer_output).
    ///
    /// `text_ids`: token IDs per batch element (batch_size,), or None for zeros.
    /// `audio_ids`: per-codebook token IDs, each (batch_size,) or None to skip.
    pub fn forward(
        &self,
        text_ids: Option<&[u32]>,
        audio_ids: &[Option<&[u32]>],
        state: &mut LmState<T, B>,
        mask: &StreamMask,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        // Text embedding: forward gives (batch, d_model), unsqueeze to (batch, 1, d_model)
        let mut all_toks = vec![];
        let mut emb = match text_ids {
            Some(ids) => {
                let ids_t = Tensor::from_vec(
                    ids.iter().map(|&x| x as i64).collect(),
                    ids.len(),
                    self.device(),
                )?;
                all_toks.push(ids[0]);
                self.text_emb.forward(&ids_t)?.unsqueeze(1)?
            }
            None => {
                let d_model = self.text_emb.hidden_size();
                let batch_size = state.transformer.batch_size();
                Tensor::zeros((batch_size, 1, d_model), self.device())?
            }
        };

        // Audio embeddings
        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(ids) = audio_ids {
                let ids_t = Tensor::from_vec(
                    ids.iter().map(|&x| x as i64).collect(),
                    ids.len(),
                    self.device(),
                )?;
                let e = audio_emb.forward(&ids_t)?.unsqueeze(1)?;
                all_toks.push(ids[0]);
                emb = emb.add(&e)?;
            }
        }
        println!("{:?}", all_toks);

        // Transformer
        let ys = self
            .transformer
            .forward(&emb, &mut state.transformer, mask)?;
        let ys = self.out_norm.forward(&ys)?;
        let logits = self.text_linear.forward(&ys)?;
        println!("TL\n{logits}");
        Ok((logits, ys))
    }

    /// Compute extra head outputs from transformer output.
    pub fn extra_heads(&self, ys: &Tensor<T, B>) -> Result<Vec<Tensor<T, B>>> {
        let mut results = Vec::with_capacity(self.extra_heads.len());
        for head in &self.extra_heads {
            results.push(head.forward(ys)?);
        }
        Ok(results)
    }

    pub fn reset_batch_idx(&self, state: &mut LmState<T, B>, batch_idx: usize) -> Result<()> {
        state.transformer.reset_batch_idx(batch_idx)
    }
}
