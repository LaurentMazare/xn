use crate::transformer_with_ca::CaSrc;
use xn::nn::var_builder::Path;
use xn::streaming::StreamMask;
use xn::{BackendQ, Result, Tensor};

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
}

pub struct DepformerSlice<Q: BackendQ> {
    transformer: crate::transformer::BatchedTransformer<Q>,
    emb: LowRankEmbeddings<Q>,
    linear_in: Q::LinearQ,
    linear_out: Q::LinearQ,
}

pub struct Model<Q: BackendQ> {
    text_emb: xn::nn::Embedding<Q::T, Q::B>,
    audio_embs: Vec<xn::nn::Embedding<Q::T, Q::B>>,
    transformer: crate::transformer_with_ca::Transformer<Q>,
    depformer: Vec<DepformerSlice<Q>>,
    out_norm: crate::transformer::Norm<Q::T, Q::B>,
    text_linear: Q::LinearQ,
    text_card: usize,
    audio_card: usize,
    audio_delays: Vec<usize>,
}

pub struct State<Q: BackendQ> {
    pub model: std::sync::Arc<Model<Q>>,
    pub transformer: crate::transformer::BatchedTransformerState<Q::T, Q::B>,
    pub temperature: Tensor<f32, Q::B>,
    pub index: usize,
    // Time-step, codebook, batch element.
    pub audio_tokens: Vec<Vec<Vec<u32>>>,
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
            let df = DepformerSlice { transformer, emb, linear_in, linear_out };
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
            (0..n_q).map(|i| cfg.delays.get(i).cloned().unwrap_or(last_delay)).collect();
        Ok(Self {
            transformer,
            depformer,
            audio_embs,
            text_emb,
            text_linear,
            out_norm,
            text_card: cfg.text_card,
            audio_card: cfg.audio_card,
            audio_delays,
        })
    }

    pub fn init_state(
        self: &std::sync::Arc<Self>,
        batch_size: usize,
        temperature: f32,
    ) -> Result<State<Q>> {
        let temperature: Tensor<f32, Q::B> =
            Tensor::full(temperature, (batch_size, 1), self.device())?;
        Ok(State {
            model: self.clone(),
            transformer: self.transformer.init_state(batch_size)?,
            temperature,
            index: 0,
            audio_tokens: Vec::new(),
        })
    }

    pub fn device(&self) -> &Q::B {
        self.text_emb.device()
    }

    /// Pre-compute the K/V projections for the cross-attention source so they
    /// can be reused across timesteps.
    pub fn maybe_precompute_ca_kv(&self, ca_src: CaSrc<Q>) -> Result<CaSrc<Q>> {
        self.transformer.maybe_precompute_ca_kv(ca_src)
    }

    /// Embed audio codes by summing the per-codebook embeddings.
    /// `codes` shape: `(batch, codebooks, frames)`. Returns `(batch, frames, d_model)`.
    pub fn embed_audio_codes(&self, codes: &Tensor<i64, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let (b, n_cb, t) = codes.dims3()?;
        let n = n_cb.min(self.audio_embs.len());
        if n == 0 {
            xn::bail!("no audio embeddings available")
        }
        let mut acc: Option<Tensor<Q::T, Q::B>> = None;
        for cb in 0..n {
            let codes_cb = codes.narrow(1, cb..cb + 1)?.reshape((b, t))?.contiguous()?;
            let e = self.audio_embs[cb].forward(&codes_cb)?;
            acc = Some(match acc {
                None => e,
                Some(prev) => prev.add(&e)?,
            });
        }
        Ok(acc.unwrap())
    }
}

impl<Q: BackendQ> State<Q> {
    pub fn frames_processed(&self) -> usize {
        self.index
    }

    pub fn device(&self) -> &Q::B {
        self.model.device()
    }

    /// Single forward step. `text_ids` and per-codebook `audio_ids` are
    /// `(batch_size,)`. Returns `(text_logits, transformer_out)` of shape
    /// `(batch, 1, text_card_out)` and `(batch, 1, d_model)` respectively.
    #[allow(clippy::type_complexity)]
    fn forward(
        &mut self,
        text_ids: Option<&[u32]>,
        audio_ids: &[Vec<u32>],
        ca_src: &CaSrc<Q>,
        mask: &StreamMask,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        use xn::ModuleT;
        let model = &self.model;
        let device = model.device();
        let mut emb = match text_ids {
            Some(ids) => {
                let ids_t =
                    Tensor::from_vec(ids.iter().map(|&x| x as i64).collect(), ids.len(), device)?;
                model.text_emb.forward(&ids_t)?.unsqueeze(1)?
            }
            None => {
                let d_model = model.text_emb.hidden_size();
                let batch_size = self.transformer.batch_size();
                Tensor::zeros((batch_size, 1, d_model), device)?
            }
        };
        for (audio_emb, ids) in model.audio_embs.iter().zip(audio_ids.iter()) {
            let ids_t =
                Tensor::from_vec(ids.iter().map(|&x| x as i64).collect(), ids.len(), device)?;
            let e = audio_emb.forward(&ids_t)?.unsqueeze(1)?;
            emb = emb.add(&e)?;
        }
        let ys = model.transformer.forward(&emb, ca_src, &mut self.transformer, mask)?;
        let ys = model.out_norm.forward(&ys)?;
        let logits = model.text_linear.forward(&ys)?;
        Ok((logits, ys))
    }

    /// Sample one audio token per codebook via the depformer, conditioned on the
    /// main transformer hidden state `ys` (shape `(batch, 1, d_model)`) and the
    /// previously sampled `text_token` (one per batch element).
    /// `temperature` has shape `(batch, 1)` — when zero this collapses to greedy
    /// argmax sampling.
    /// Returns a `Vec` of length `n_slices`, each entry of length `batch_size`.
    fn depformer_sample(
        &self,
        ys: &Tensor<Q::T, Q::B>,
        text_token: &[u32],
        temperature: &Tensor<f32, Q::B>,
        semantic_token: i64,
    ) -> Result<Vec<Vec<u32>>> {
        use xn::ModuleT;
        let model = &self.model;
        let device = model.device();
        let batch_size = self.transformer.batch_size();
        if text_token.len() != batch_size {
            xn::bail!("text_token len {} does not match batch_size {batch_size}", text_token.len())
        }
        // The depformer slices share the same architecture, so a single state
        // can be reused: every slice extends the kv-cache by one position,
        // matching the moshi-rs `copy_state` propagation.
        let mut state = model.depformer[0].transformer.init_state(batch_size)?;
        let mask = StreamMask::all_active(batch_size);

        let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(model.depformer.len());
        let mut last_token: Vec<u32> = text_token.to_vec();

        for (slice_idx, slice) in model.depformer.iter().enumerate() {
            let xs = slice.linear_in.forward(ys)?;
            let token_id = Tensor::from_vec(
                last_token.iter().map(|&x| x as i64).collect(),
                batch_size,
                device,
            )?;
            let token_emb = slice.emb.forward(&token_id)?.unsqueeze(1)?;
            let xs = xs.add(&token_emb)?;
            let xs = slice.transformer.forward(&xs, &mut state, &mask)?;
            let logits = slice.linear_out.forward(&xs)?;
            let (b, _t, vocab) = logits.dims3()?;
            let logits_2d = logits.reshape((b, vocab))?;
            let sampled = crate::sampling::gumbel_max(&logits_2d, temperature)?;
            let mut sampled_v: Vec<i64> = sampled.to_vec()?;
            if slice_idx == 0 {
                sampled_v.fill(semantic_token);
            }
            let tokens: Vec<u32> = sampled_v.into_iter().map(|x| x as u32).collect();
            last_token = tokens.clone();
            all_tokens.push(tokens);
        }
        Ok(all_tokens)
    }

    fn audio_tokens_for_current_step(&self) -> Result<Vec<Vec<u32>>> {
        use xn::Context;

        let mut audio_tokens = vec![];
        for (i, delay) in self.model.audio_delays.iter().enumerate() {
            let audio_token = if self.index > *delay {
                let prev_tokens =
                    self.audio_tokens.last().context("audio_tokens should not be empty")?;
                prev_tokens[i][0]
            } else {
                self.model.audio_card as u32
            };
            audio_tokens.push(vec![audio_token]);
        }
        Ok(audio_tokens)
    }

    fn text_token_for_current_step(&self) -> Vec<u32> {
        if self.index == 0 { vec![self.model.text_card as u32] } else { vec![3] }
    }

    pub fn step(
        &mut self,
        ca_src: &CaSrc<Q>,
        mask: &StreamMask,
        semantic_token: i64,
    ) -> Result<()> {
        // TODO(laurent): support for batch size greater than 1.
        let pad_token = 3;
        let text_tokens = self.text_token_for_current_step();
        let audio_tokens = self.audio_tokens_for_current_step()?;
        let (_text_logits, ys) = self.forward(Some(&text_tokens), &audio_tokens, ca_src, mask)?;
        let audio_tokens =
            self.depformer_sample(&ys, &[pad_token], &self.temperature, semantic_token)?;
        self.audio_tokens.push(audio_tokens);
        self.index += 1;
        Ok(())
    }

    pub fn last_audio_tokens(&self) -> Option<Vec<u32>> {
        let mut last_tokens = vec![];
        for (cb_idx, delay) in self.model.audio_delays.iter().enumerate() {
            if self.index > *delay {
                let step_idx = self.index - delay - 1;
                last_tokens.push(self.audio_tokens[step_idx][cb_idx][0]);
            } else {
                return None;
            }
        }
        Some(last_tokens)
    }
}
