use crate::Tokenizer;
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

pub struct LUTConditioner<T: WithDTypeF, B: Backend> {
    pub tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
    embed: Tensor<T, B>,
    learnt_padding: Option<Tensor<T, B>>,
    learnt_padding_id: Option<u32>,
    pub dim: usize,
    pub output_dim: usize,
}

impl<T: WithDTypeF, B: Backend> LUTConditioner<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_bins: usize,
        tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
        dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let embed = vb.tensor("embed.weight", (n_bins + 1, dim))?;
        let learnt_padding = if vb.contains("learnt_padding") {
            Some(vb.tensor("learnt_padding", (1, 1, output_dim))?)
        } else {
            None
        };
        let embed = if vb.contains("output_proj.weight") {
            let proj = Linear::load(vb.pp("output_proj"), dim, output_dim)?;
            proj.forward(&embed)?
        } else {
            embed
        };
        let (embed, learnt_padding_id) = match learnt_padding.as_ref() {
            Some(learnt_padding) => {
                let learnt_padding = learnt_padding.squeeze(0)?;
                let embed = Tensor::cat(&[&embed, &learnt_padding], 0)?;
                (embed, Some(n_bins as u32 + 1))
            }
            None => (embed, None),
        };
        Ok(Self { tokenizer, embed, dim, output_dim, learnt_padding, learnt_padding_id })
    }

    pub fn learnt_padding_id(&self) -> Option<u32> {
        self.learnt_padding_id
    }

    /// Tokenize text and return token ids.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self.tokenizer.as_ref() {
            Some(tokenizer) => Ok(tokenizer.encode(text)),
            None => xn::bail!("No tokenizer available for LUTConditioner"),
        }
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
        let emb = emb.reshape((1, token_ids.len(), self.output_dim))?;
        Ok(emb)
    }

    pub fn learnt_padding(&self) -> Option<&Tensor<T, B>> {
        self.learnt_padding.as_ref()
    }
}
