use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct LUTConditioner<T: WithDTypeF, B: Backend> {
    pub tokenizer: sentencepiece::SentencePieceProcessor,
    embed: Tensor<T, B>,
    pub dim: usize,
    pub output_dim: usize,
}

impl<T: WithDTypeF, B: Backend> LUTConditioner<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_bins: usize,
        tokenizer_path: &str,
        dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let tokenizer = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
            .map_err(|e| xn::Error::Msg(format!("Failed to load tokenizer: {e}")))?;
        let embed = vb.tensor("embed.weight", (n_bins + 1, dim))?;
        Ok(Self { tokenizer, embed, dim, output_dim })
    }

    /// Tokenize text and return token ids.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let pieces = self.tokenizer.encode(text).unwrap_or_default();
        pieces.iter().map(|p| p.id).collect()
    }

    /// Get embeddings for token ids. Returns [1, num_tokens, dim].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor<T, B>> {
        if token_ids.is_empty() {
            let dev = self.embed.device();
            return Tensor::zeros((1, 0, self.dim), dev);
        }
        let emb = self.embed.index_select(token_ids, 0)?;
        emb.reshape((1, token_ids.len(), self.dim))
    }
}
