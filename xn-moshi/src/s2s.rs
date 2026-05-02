#![allow(unused)]
use crate::transformer::{BatchedTransformerState, Config as TransformerConfig, Norm};
use crate::transformer_with_ca::CaSrc;
use xn::nn::{Embedding, Linear, var_builder::Path};
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
}

pub struct State<Q: BackendQ> {
    pub model: std::sync::Arc<Model<Q>>,
    pub transformer: BatchedTransformerState<Q::T, Q::B>,
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
        Ok(Self { transformer, depformer, audio_embs, text_emb, text_linear, out_norm })
    }

    pub fn init_state(self: &std::sync::Arc<Self>, batch_size: usize) -> Result<State<Q>> {
        Ok(State { model: self.clone(), transformer: self.transformer.init_state(batch_size)? })
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
    pub fn device(&self) -> &Q::B {
        self.model.device()
    }

    /// Single forward step. `text_ids` and per-codebook `audio_ids` are
    /// `(batch_size,)`. Returns `(text_logits, transformer_out)` of shape
    /// `(batch, 1, text_card_out)` and `(batch, 1, d_model)` respectively.
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &mut self,
        text_ids: Option<&[u32]>,
        audio_ids: &[Option<&[u32]>],
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
        for (audio_emb, audio_ids) in model.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(ids) = audio_ids {
                let ids_t =
                    Tensor::from_vec(ids.iter().map(|&x| x as i64).collect(), ids.len(), device)?;
                let e = audio_emb.forward(&ids_t)?.unsqueeze(1)?;
                emb = emb.add(&e)?;
            }
        }
        let ys = model.transformer.forward(&emb, ca_src, &mut self.transformer, mask)?;
        let ys = model.out_norm.forward(&ys)?;
        let logits = model.text_linear.forward(&ys)?;
        Ok((logits, ys))
    }
}
