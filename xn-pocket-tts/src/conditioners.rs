use crate::Tokenizer;
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct LUTConditioner<T: WithDTypeF, B: Backend> {
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    embed: Tensor<T, B>,
    pub dim: usize,
    pub output_dim: usize,
}

impl<T: WithDTypeF, B: Backend> LUTConditioner<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_bins: usize,
        tokenizer: Box<dyn Tokenizer + Send + Sync>,
        dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let embed = vb.tensor("embed.weight", (n_bins + 1, dim))?;
        Ok(Self { tokenizer, embed, dim, output_dim })
    }

    /// Tokenize text and return token ids.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Get embeddings for token ids. Returns [1, num_tokens, dim].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor<T, B>> {
        if token_ids.is_empty() {
            let dev = self.embed.device();
            return Tensor::zeros((1, 0, self.dim), dev);
        }
        let ids_t = Tensor::from_vec(
            token_ids.iter().map(|&x| x as i64).collect(),
            token_ids.len(),
            self.embed.device(),
        )?;
        let emb = self.embed.index_select(&ids_t, 0)?;
        emb.reshape((1, token_ids.len(), self.dim))
    }
}
