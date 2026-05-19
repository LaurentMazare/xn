use crate::transformer_with_ca::CaSrc;
use xn::nn::var_builder::Path;
use xn::streaming::StreamMask;
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

pub fn add_sin_embeddings<T: WithDTypeF, B: Backend>(xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    let (_b, seq_len, dim) = xs.dims3()?;
    let device = xs.device();
    let half_dim = dim / 2;
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;
    let inv_freq: Vec<f32> =
        (0..half_dim).map(|i| 1f32 / 10000f32.powf(i as f32 / (half_dim - 1) as f32)).collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
    let freqs = positions.broadcast_mul(&inv_freq)?;
    let pos_emb = Tensor::cat(&[&freqs.cos()?, &freqs.sin()?], xn::D::Minus1)?;
    xs.to::<f32>()?.broadcast_add(&pos_emb)?.to::<T>()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DepformerConfig {
    pub num_layers: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub dim_feedforward: usize,
    pub low_rank_embeddings: Option<usize>,
    pub norm: crate::NormType,
    pub weights_per_step_schedule: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConditionerConfig {
    pub name: String,
    #[serde(flatten)]
    pub inner: crate::conditioners::ConditionerConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub transformer: crate::transformer::Config,
    pub moshi_name: String,
    pub mimi_name: String,
    pub speaker_wavs_mimi_name: String,
    pub delays: Vec<usize>,
    pub depformer: DepformerConfig,
    pub text_card: usize,
    pub audio_card: usize,
    pub text_card_out: usize,
    pub conditioners: Vec<ConditionerConfig>,
}

pub struct DepformerSlice<Q: BackendQ> {
    transformer: crate::transformer::BatchedTransformer<Q>,
    emb: LowRankEmbeddings<Q>,
    linear_in: Q::LinearQ,
    linear_out: Q::LinearQ,
    norm: crate::transformer::Norm<Q::T, Q::B>,
}

pub struct Model<Q: BackendQ> {
    text_emb: xn::nn::Embedding<Q::T, Q::B>,
    audio_embs: Vec<xn::nn::Embedding<Q::T, Q::B>>,
    transformer: crate::transformer_with_ca::Transformer<Q>,
    depformer: Vec<DepformerSlice<Q>>,
    out_norm: crate::transformer::Norm<Q::T, Q::B>,
    text_linear: Q::LinearQ,
    audio_card: usize,
    audio_delays: Vec<usize>,
    speaker_wavs_output_proj: Q::LinearQ,
    speaker_wavs_learnt_padding: Tensor<Q::T, Q::B>,
    #[allow(dead_code)]
    conditioners: crate::conditioners::Conditioners<Q::T, Q::B>,
    #[allow(dead_code)]
    default_conditions: Option<Tensor<Q::T, Q::B>>,
}

#[derive(Debug, Clone)]
struct PerBatch {
    index: usize,
    // (codebook, time)
    audio_tokens: Vec<Vec<i64>>,
}

impl PerBatch {
    fn new(n_slices: usize) -> Self {
        Self { index: 0, audio_tokens: vec![vec![]; n_slices] }
    }

    fn reset(&mut self) {
        self.index = 0;
        self.audio_tokens.iter_mut().for_each(|v| v.clear());
    }
}

pub struct State<Q: BackendQ> {
    pub model: std::sync::Arc<Model<Q>>,
    pub transformer: crate::transformer::BatchedTransformerState<Q::T, Q::B>,
    pub temperature: Tensor<f32, Q::B>,
    per_batch: Vec<PerBatch>,
}

#[derive(Debug, Clone)]
struct LowRankEmbeddings<Q: BackendQ> {
    embeddings: xn::nn::Embedding<Q::T, Q::B>,
    low_rank: Option<Q::LinearQ>,
}

impl<Q: BackendQ> LowRankEmbeddings<Q> {
    fn load(
        vb: &Path<Q::B>,
        in_vocab_size: usize,
        dim: usize,
        low_rank_dim: Option<usize>,
    ) -> Result<Self> {
        let (low_rank, embeddings) = match low_rank_dim {
            None => {
                let embeddings = xn::nn::Embedding::load(vb, in_vocab_size, dim)?;
                (None, embeddings)
            }
            Some(low_rank_dim) => {
                let low_rank = Q::linear_load(vb.pp("low_rank"), low_rank_dim, dim)?;
                let embeddings = xn::nn::Embedding::load(vb, in_vocab_size, low_rank_dim)?;
                (Some(low_rank), embeddings)
            }
        };
        Ok(Self { embeddings, low_rank })
    }

    fn forward(&self, xs: &Tensor<i64, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;

        let embs = self.embeddings.forward(xs)?;
        match self.low_rank.as_ref() {
            None => Ok(embs),
            Some(lr) => lr.forward(&embs),
        }
    }
}

impl<Q: BackendQ> Model<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let transformer =
            crate::transformer_with_ca::Transformer::load(&vb.pp("transformer"), &cfg.transformer)?;
        let n_q = cfg.depformer.weights_per_step_schedule.len();
        if cfg.depformer.norm != crate::NormType::LayerNorm {
            xn::bail!("only LayerNorm is currently supported for the depformer slices");
        }
        let depformer_config = crate::transformer::Config {
            d_model: cfg.depformer.dim,
            num_heads: cfg.depformer.num_heads,
            dim_feedforward: cfg.depformer.dim_feedforward,
            num_layers: cfg.depformer.num_layers,
            norm: crate::NormType::RmsNorm,
            bias_attn: false,
            bias_ff: false,
            causal: true,
            context: n_q,
            conv_layout: false,
            gating: Some(crate::seanet::Activation::Silu),
            kv_repeat: 1,
            layer_scale: None,
            max_period: 10_000,
            norm_first: true,
            positional_embedding: crate::transformer::PositionalEmbedding::None,
            use_conv_block: false,
        };
        let mut depformer = Vec::with_capacity(n_q);
        let mut audio_embs = Vec::with_capacity(n_q);
        let df_vb = vb.pp("depformer");
        let emb_vb = vb.pp("emb");
        // The safetensor exporter merges the weights per step appropriately.
        for slice_idx in 0..n_q {
            let df_vb = df_vb.pp(slice_idx);
            let transformer = crate::transformer::BatchedTransformer::load(
                &df_vb.pp("transformer"),
                &depformer_config,
            )?;
            let in_vocab_size = if slice_idx == 0 { cfg.text_card } else { cfg.audio_card };
            let emb = LowRankEmbeddings::load(
                &df_vb.pp("emb"),
                in_vocab_size + 1,
                cfg.transformer.d_model,
                cfg.depformer.low_rank_embeddings,
            )?;
            let linear_in =
                Q::linear_load(df_vb.pp("linear_in"), cfg.transformer.d_model, cfg.depformer.dim)?;
            let linear_out =
                Q::linear_load(df_vb.pp("linear_out"), cfg.depformer.dim, cfg.audio_card)?;
            let norm = crate::transformer::Norm::load(
                df_vb.pp("norm"),
                cfg.depformer.dim,
                cfg.depformer.norm,
            )?;
            let df = DepformerSlice { transformer, emb, linear_in, linear_out, norm };
            depformer.push(df);
            let audio_emb = xn::nn::Embedding::load(
                emb_vb.pp(slice_idx),
                cfg.audio_card + 1,
                cfg.transformer.d_model,
            )?;
            audio_embs.push(audio_emb);
        }
        let text_emb =
            xn::nn::Embedding::load(vb.pp("text_emb"), cfg.text_card + 1, cfg.transformer.d_model)?;
        let text_linear =
            Q::linear_load(vb.pp("text_linear"), cfg.transformer.d_model, cfg.text_card_out)?;
        let out_norm = crate::transformer::Norm::load(
            vb.pp("out_norm"),
            cfg.transformer.d_model,
            cfg.transformer.norm,
        )?;
        let last_delay = match cfg.delays.last() {
            Some(&d) => d,
            None => xn::bail!("at least one delay must be specified"),
        };
        let n_q = depformer.len();
        let audio_delays: Vec<_> =
            // The delays config includes the text stream delay so shift it by one.
            (0..n_q).map(|i| cfg.delays.get(i + 1).cloned().unwrap_or(last_delay)).collect();
        let conditioners =
            cfg.conditioners.iter().map(|c| (c.name.clone(), c.inner.clone())).collect();
        let conditioners = crate::conditioners::load(cfg.transformer.d_model, &conditioners, vb)?;
        let default_conditions = conditioners.condition_sum(&std::collections::HashMap::new())?;
        let speaker_wavs_output_proj = Q::linear_load(
            vb.pp("condition_provider.conditioners.speaker_wavs.output_proj"),
            512, // TODO(laurent): get the mimi dim from some config.
            cfg.transformer.d_model,
        )?;
        let speaker_wavs_learnt_padding = vb.tensor(
            "condition_provider.conditioners.speaker_wavs.learnt_padding",
            (1, 1, cfg.transformer.d_model),
        )?;
        Ok(Self {
            transformer,
            depformer,
            audio_embs,
            text_emb,
            text_linear,
            out_norm,
            audio_card: cfg.audio_card,
            audio_delays,
            conditioners,
            default_conditions,
            speaker_wavs_output_proj,
            speaker_wavs_learnt_padding,
        })
    }

    pub fn speaker_wavs_ca_src(
        &self,
        speaker_wavs: &Tensor<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let speaker_wavs = speaker_wavs.t()?.contiguous()?;
        let projected = self.speaker_wavs_output_proj.forward(&speaker_wavs)?;
        let (_b, embs, dim) = projected.dims3()?;
        let learnt_padding =
            self.speaker_wavs_learnt_padding.expand((1, 2 * embs, dim))?.contiguous()?;
        let projected = Tensor::cat(&[&projected, &learnt_padding], 1)?;
        let projected = add_sin_embeddings(&projected)?;
        Ok(projected)
    }

    pub fn condition_sum(
        &self,
        values: &std::collections::HashMap<String, crate::conditioners::Value>,
    ) -> Result<Option<Tensor<Q::T, Q::B>>> {
        self.conditioners.condition_sum(values)
    }

    pub fn init_state(
        self: &std::sync::Arc<Self>,
        batch_size: usize,
        temperature: f32,
    ) -> Result<State<Q>> {
        let temperature: Tensor<f32, Q::B> =
            Tensor::full(temperature, (batch_size, 1), self.device())?;
        let n_slices = self.depformer.len();
        Ok(State {
            model: self.clone(),
            transformer: self.transformer.init_state(batch_size)?,
            temperature,
            per_batch: vec![PerBatch::new(n_slices); batch_size],
        })
    }

    pub fn device(&self) -> &Q::B {
        self.text_emb.device()
    }

    pub fn n_slices(&self) -> usize {
        self.depformer.len()
    }
}

impl<Q: BackendQ> State<Q> {
    pub fn batch_size(&self) -> usize {
        self.per_batch.len()
    }

    pub fn n_slices(&self) -> usize {
        self.model.n_slices()
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        if batch_idx >= self.batch_size() {
            xn::bail!("batch_idx out of bounds");
        }
        self.transformer.reset_batch_idx(batch_idx)?;
        self.per_batch[batch_idx].reset();
        Ok(())
    }

    pub fn frames_processed(&self, batch_idx: usize) -> usize {
        self.per_batch[batch_idx].index
    }

    pub fn device(&self) -> &Q::B {
        self.model.device()
    }

    /// Single forward step. `text_ids` and per-codebook `audio_tokens` are
    /// `(batch_size, codebooks)`. Returns `(text_logits, transformer_out)` of shape
    /// `(batch, 1, text_card_out)` and `(batch, 1, d_model)` respectively.
    #[allow(clippy::type_complexity)]
    fn forward(
        &mut self,
        audio_tokens: &[Vec<i64>],
        ca_src: &CaSrc<Q>,
        mask: &StreamMask,
        condition_sum: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        use xn::ModuleT;
        let model = &self.model;
        let device = model.device();
        let d_model = model.text_emb.hidden_size();
        let mut emb = Tensor::zeros((self.batch_size(), 1, d_model), device)?;
        // There are only audio embeddings and no text embeddings as gen_text is false for
        // this model.
        for (audio_emb, ids) in model.audio_embs.iter().zip(audio_tokens.iter()) {
            let ids_t = Tensor::from_vec(ids.clone(), ids.len(), device)?;
            let e = audio_emb.forward(&ids_t)?.unsqueeze(1)?;
            emb = emb.add(&e)?;
        }
        let emb = match condition_sum {
            None => emb,
            Some(cond) => emb.broadcast_add(cond)?,
        };
        let ys = model.transformer.forward(&emb, ca_src, &mut self.transformer, mask)?;
        let ys = model.out_norm.forward(&ys)?;
        let logits = model.text_linear.forward(&ys)?;
        Ok((logits, ys))
    }

    /// Sample one audio token per codebook via the depformer, conditioned on the
    /// main transformer hidden state `ys` (shape `(batch, 1, d_model)`) and the
    /// previously sampled `text_tokens` (one per batch element).
    /// `temperature` has shape `(batch, 1)` — when zero this collapses to greedy
    /// argmax sampling.
    /// Returns a `Vec` of length `n_slices`, each entry of length `batch_size`.
    fn depformer_sample(
        &self,
        ys: &Tensor<Q::T, Q::B>,
        temperature: &Tensor<f32, Q::B>,
        semantic_tokens: &[i64],
    ) -> Result<Vec<Vec<i64>>> {
        use xn::ModuleT;

        let batch_size = self.batch_size();
        if semantic_tokens.len() != batch_size {
            xn::bail!(
                "semantic_tokens length {} does not match batch_size {batch_size}",
                semantic_tokens.len(),
            );
        }
        let model = &self.model;
        let device = model.device();
        // The depformer slices share the same architecture, so a single state
        // can be reused: every slice extends the kv-cache by one position,
        // matching the moshi-rs `copy_state` propagation.
        let mut state = model.depformer[0].transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);

        let mut all_tokens: Vec<Vec<i64>> = Vec::with_capacity(model.depformer.len());

        for (slice_idx, slice) in model.depformer.iter().enumerate() {
            let xs = slice.linear_in.forward(ys)?;
            let xs = match all_tokens.last() {
                None => xs,
                Some(tokens) => {
                    let token_id = Tensor::from_vec(tokens.clone(), batch_size, device)?;
                    let token_emb = slice.emb.forward(&token_id)?.unsqueeze(1)?;
                    xs.add(&token_emb)?
                }
            };
            let xs = slice.transformer.forward(&xs, &mut state, &mask)?;
            let xs = slice.norm.forward(&xs)?;
            let logits = slice.linear_out.forward(&xs)?;
            let (b, _t, vocab) = logits.dims3()?;
            let logits_2d = logits.reshape((b, vocab))?;
            let sampled = crate::sampling::gumbel_max(&logits_2d, temperature)?;
            let mut sampled_v: Vec<i64> = sampled.to_vec()?;
            if slice_idx == 0 {
                sampled_v.copy_from_slice(semantic_tokens);
            }
            all_tokens.push(sampled_v);
        }
        Ok(all_tokens)
    }

    fn audio_tokens_for_current_step(&self) -> Result<Vec<Vec<i64>>> {
        use xn::Context;

        let mut audio_tokens = vec![];
        for (cb_idx, delay) in self.model.audio_delays.iter().enumerate() {
            let mut tokens = Vec::with_capacity(self.batch_size());
            for per_batch in self.per_batch.iter() {
                let audio_token = if per_batch.index > *delay {
                    let prev_token = per_batch.audio_tokens[cb_idx]
                        .last()
                        .context("audio_tokens should not be empty")?;
                    *prev_token
                } else {
                    self.model.audio_card as i64
                };
                tokens.push(audio_token);
            }
            audio_tokens.push(tokens)
        }
        Ok(audio_tokens)
    }

    pub fn step(
        &mut self,
        ca_src: &CaSrc<Q>,
        mask: &StreamMask,
        condition_sum: Option<&Tensor<Q::T, Q::B>>,
        semantic_tokens: &[i64],
    ) -> Result<()> {
        let audio_tokens = self.audio_tokens_for_current_step()?;
        let (_text_logits, ys) = self.forward(&audio_tokens, ca_src, mask, condition_sum)?;
        let audio_tokens = self.depformer_sample(&ys, &self.temperature, semantic_tokens)?;
        for (batch_idx, per_batch) in self.per_batch.iter_mut().enumerate() {
            per_batch.index += 1;
            for (tokens, new_token) in per_batch.audio_tokens.iter_mut().zip(audio_tokens.iter()) {
                tokens.push(new_token[batch_idx]);
            }
        }
        Ok(())
    }

    pub fn last_audio_tokens(&self) -> Option<Vec<Vec<i64>>> {
        let max_delay = match self.model.audio_delays.iter().max() {
            Some(d) => *d,
            None => return None,
        };
        let mut last_tokens = Vec::with_capacity(self.batch_size());
        let n_slices = self.model.depformer.len();
        for per_batch in self.per_batch.iter() {
            let mut tokens = Vec::with_capacity(n_slices);
            for (cb_idx, delay) in self.model.audio_delays.iter().enumerate() {
                if per_batch.index + *delay > max_delay {
                    let step_idx = per_batch.index + delay - max_delay - 1;
                    tokens.push(per_batch.audio_tokens[cb_idx][step_idx]);
                } else {
                    return None;
                }
            }
            last_tokens.push(tokens);
        }
        Some(last_tokens)
    }
}
