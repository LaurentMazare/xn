pub mod asr;
pub mod batched_transformer;
pub mod conditioners;
pub mod conv;
pub mod lm;
pub mod mimi;
pub mod moshi;
pub mod quantization;
pub mod seanet;
pub mod transformer;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
}
