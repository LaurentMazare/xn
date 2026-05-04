use crate::conditioners::{ConditionerConfig, Conditioners};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gating {
    Silu,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub card: usize,
    pub n_q: usize,
    pub dep_q: usize,
    pub delays: Vec<usize>,
    pub dim: usize,
    pub text_card: usize,
    pub existing_text_padding_id: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_scale: f64,
    pub causal: bool,
    pub context: usize,
    pub max_period: usize,
    pub gating: Gating,
    pub extra_heads_num_heads: usize,
    pub extra_heads_dim: usize,
    pub conditioners: std::collections::HashMap<String, ConditionerConfig>,
}

impl Config {
    /// Load conditioners from weights based on the config.
    pub fn load_conditioners<T: xn::WithDTypeF, B: xn::Backend>(
        &self,
        vb: &xn::nn::var_builder::Path<B>,
    ) -> xn::Result<Conditioners<T, B>> {
        crate::conditioners::load(self.dim, &self.conditioners, vb)
    }

    pub fn to_lm_config(&self) -> crate::lm::Config {
        let dim_feedforward = (self.dim as f64 * self.hidden_scale) as usize;
        let extra_heads = if self.extra_heads_num_heads > 0 {
            Some(crate::lm::ExtraHeadsConfig {
                num_heads: self.extra_heads_num_heads,
                dim: self.extra_heads_dim,
            })
        } else {
            None
        };
        let transformer = crate::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period,
            use_conv_block: false,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: crate::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
        };
        crate::lm::Config {
            transformer,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            extra_heads,
        }
    }
}
