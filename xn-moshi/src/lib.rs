pub mod batched_transformer;
pub mod conv;
pub mod mimi;
pub mod quantization;
pub mod seanet;
pub mod streaming;
pub mod transformer;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}
