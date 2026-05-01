pub mod asr;
pub mod conditioners;
pub mod conv;
pub mod lm;
pub mod mimi;
pub mod moshi;
pub mod quantization;
pub mod s2s;
pub mod seanet;
pub mod transformer;
pub mod transformer_with_ca;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    #[serde(alias = "real_rms_norm_f32")]
    RmsNorm,
    LayerNorm,
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
}
