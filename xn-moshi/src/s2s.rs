#![allow(unused)]
use crate::transformer::{self, BatchedTransformerState, Config as TransformerConfig, Norm};
use xn::nn::{Embedding, Linear, var_builder::Path};
use xn::streaming::StreamMask;
use xn::{BackendQ, Result, Tensor};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub transformer: crate::transformer::Config,
    pub weights_name: String,
}

pub struct Model<Q: BackendQ> {
    transformer: transformer::BatchedTransformer<Q>,
}

pub struct State<Q: BackendQ> {
    pub model: std::sync::Arc<Model<Q>>,
    pub transformer: BatchedTransformerState<Q::T, Q::B>,
}

impl<Q: BackendQ> Model<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let transformer =
            transformer::BatchedTransformer::load(&vb.pp("transformer"), &cfg.transformer)?;
        Ok(Self { transformer })
    }

    pub fn init_state(self: &std::sync::Arc<Self>, batch_size: usize) -> Result<State<Q>> {
        Ok(State { model: self.clone(), transformer: self.transformer.init_state(batch_size)? })
    }
}
