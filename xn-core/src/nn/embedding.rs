use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDTypeF};

#[derive(Clone, Debug)]
pub struct Embedding<T: WithDTypeF, B: Backend> {
    embeddings: Tensor<T, B>,
    hidden_size: usize,
}

impl<T: WithDTypeF, B: Backend> Embedding<T, B> {
    pub fn new(embeddings: Tensor<T, B>, hidden_size: usize) -> Self {
        Self { embeddings, hidden_size }
    }

    pub fn embeddings(&self) -> &Tensor<T, B> {
        &self.embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn forward(&self, indexes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }

    pub fn load<V: std::borrow::Borrow<Path<B>>>(
        vb: V,
        in_size: usize,
        out_size: usize,
    ) -> Result<Self> {
        let vb = vb.borrow();
        let embeddings = vb.tensor("weight", (in_size, out_size))?;
        Ok(Self::new(embeddings, out_size))
    }

    pub fn device(&self) -> &B {
        self.embeddings.device()
    }
}
