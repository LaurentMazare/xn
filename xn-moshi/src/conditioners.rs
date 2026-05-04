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
    pub config: LutConfig,
}

impl<T: WithDTypeF, B: Backend> LUTConditioner<T, B> {
    pub fn load(
        vb: &Path<B>,
        tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
        output_dim: usize,
        config: LutConfig,
    ) -> Result<Self> {
        let n_bins = config.n_bins;
        let dim = config.dim;
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
        Ok(Self { tokenizer, embed, dim, output_dim, learnt_padding, learnt_padding_id, config })
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

pub struct ContinuousConditioner<T: WithDTypeF, B: Backend> {
    pub dim: usize,
    pub output_dim: usize,
    pub scale_factor: f64,
    pub max_period: f64,
    output_proj: Option<Linear<T, B>>,
    learnt_padding: Option<Tensor<T, B>>,
    dev: B,
}

impl<T: WithDTypeF, B: Backend> ContinuousConditioner<T, B> {
    pub fn load(vb: &Path<B>, dim: usize, output_dim: usize, scale_factor: f64) -> Result<Self> {
        let output_proj = if vb.contains("output_proj.weight") {
            Some(Linear::load(vb.pp("output_proj"), dim, output_dim)?)
        } else {
            None
        };
        let learnt_padding = if vb.contains("learnt_padding") {
            Some(vb.tensor("learnt_padding", (1, 1, output_dim))?)
        } else {
            None
        };
        let dev = vb.device().clone();
        Ok(Self {
            dim,
            output_dim,
            scale_factor,
            max_period: 10000.0,
            output_proj,
            learnt_padding,
            dev,
        })
    }

    /// Create sin embeddings for a continuous value, projected and with learnt padding applied.
    /// Returns [1, 1, output_dim].
    pub fn embed(&self, value: f64) -> Result<Tensor<T, B>> {
        let scaled = value * self.scale_factor;
        let half_dim = self.dim / 2;
        let mut data = vec![T::zero(); self.dim];
        for i in 0..half_dim {
            let freq = (-(i as f64) / (half_dim - 1) as f64 * self.max_period.ln()).exp();
            let angle = scaled * freq;
            data[i] = T::from_f32(angle.cos() as f32);
            data[i + half_dim] = T::from_f32(angle.sin() as f32);
        }
        let sin_emb = Tensor::from_vec(data, (1, 1, self.dim), &self.dev)?;
        let cond = match &self.output_proj {
            Some(proj) => proj.forward(&sin_emb)?,
            None => sin_emb,
        };
        Ok(cond)
    }

    pub fn learnt_padding(&self) -> Option<&Tensor<T, B>> {
        self.learnt_padding.as_ref()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub default_value: Option<String>,
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContinuousConfig {
    pub dim: usize,
    pub scale_factor: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerConfig {
    Lut { lut: LutConfig },
    Continuous { continuous: ContinuousConfig },
}

pub struct Conditioners<T: xn::WithDTypeF, B: xn::Backend> {
    pub lut: std::collections::HashMap<String, LUTConditioner<T, B>>,
    pub continuous: std::collections::HashMap<String, ContinuousConditioner<T, B>>,
}

pub enum Value {
    Str(String),
    Num(f64),
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Num(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl<T: xn::WithDTypeF, B: xn::Backend> Conditioners<T, B> {
    pub fn condition_sum(
        &self,
        values: &std::collections::HashMap<String, Value>,
    ) -> xn::Result<Option<xn::Tensor<T, B>>> {
        let mut result: Option<xn::Tensor<T, B>> = None;
        for (name, lut) in self.lut.iter() {
            let emb = match values.get(name) {
                Some(Value::Str(s)) => {
                    let index = lut.config.possible_values.iter().position(|v| v == s).ok_or_else(|| {
                        xn::Error::Msg(format!(
                            "Invalid value for LUT conditioner {name}: {s}. Expected one of: {:?}",
                            lut.config.possible_values
                        ))
                    })?;
                    let index = Tensor::from_vec(vec![index as i64], (1,), lut.embed.device())?;
                    lut.embed.index_select(&index, 0)?.reshape((1, 1, lut.output_dim))?
                }
                Some(Value::Num(n)) => {
                    xn::bail!("Expected string value for LUT conditioner {name}, got number {n}")
                }
                None => match lut.learnt_padding() {
                    Some(lp) => lp.copy()?,
                    None => continue,
                },
            };
            result = Some(match result {
                Some(acc) => acc.add(&emb)?,
                None => emb,
            });
        }
        for (name, cont) in self.continuous.iter() {
            let value = match values.get(name) {
                Some(Value::Num(n)) => *n,
                Some(Value::Str(s)) => s.parse::<f64>().map_err(|_| {
                    xn::Error::Msg(format!("invalid value for continuous conditioner {name}: {s}"))
                })?,
                None => continue, // No value provided for this conditioner, skip it.
            };
            let emb = cont.embed(value)?;
            result = Some(match result {
                Some(acc) => acc.add(&emb)?,
                None => emb,
            });
        }
        Ok(result)
    }
}

/// Load conditioners from weights based on the config.
pub fn load<T: xn::WithDTypeF, B: xn::Backend>(
    output_dim: usize,
    conditioner_configs: &std::collections::HashMap<String, ConditionerConfig>,
    vb: &xn::nn::var_builder::Path<B>,
) -> xn::Result<Conditioners<T, B>> {
    let mut lut = std::collections::HashMap::new();
    let mut continuous = std::collections::HashMap::new();
    for (name, cond_config) in conditioner_configs {
        match cond_config {
            ConditionerConfig::Lut { lut: lut_cfg } => {
                let conditioner = crate::conditioners::LUTConditioner::load(
                    &vb.pp("condition_provider").pp("conditioners").pp(name),
                    None,
                    output_dim,
                    lut_cfg.clone(),
                )?;
                lut.insert(name.clone(), conditioner);
            }
            ConditionerConfig::Continuous { continuous: cont_cfg } => {
                let conditioner = crate::conditioners::ContinuousConditioner::load(
                    &vb.pp("condition_provider").pp("conditioners").pp(name),
                    cont_cfg.dim,
                    output_dim,
                    cont_cfg.scale_factor,
                )?;
                continuous.insert(name.clone(), conditioner);
            }
        }
    }
    Ok(Conditioners { lut, continuous })
}
