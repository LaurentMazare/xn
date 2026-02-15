use crate::conditioners::LUTConditioner;
use crate::mimi_transformer::{StreamingTransformer, StreamingTransformerState};
use crate::mlp::SimpleMLPAdaLN;
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

pub trait Rng {
    fn sample(&mut self) -> f32;
}

/// Lagrangian Self Distillation decode.
/// Rebuilds the data sample from starting point x_0.
fn lsd_decode<T: WithDTypeF, B: Backend>(
    flow_net: &SimpleMLPAdaLN<T, B>,
    transformer_out: &Tensor<T, B>,
    x_0: &Tensor<T, B>,
    num_steps: usize,
) -> Result<Tensor<T, B>> {
    let mut current = x_0.clone();
    let dev = x_0.device();

    for i in 0..num_steps {
        let s_val = i as f32 / num_steps as f32;
        let t_val = (i + 1) as f32 / num_steps as f32;

        // Create s and t tensors matching x_0 shape but with last dim = 1
        let shape: Vec<usize> =
            x_0.dims().iter().copied().take(x_0.rank() - 1).chain([1]).collect();
        let s = Tensor::full(T::from_f32(s_val), shape.clone(), dev)?;
        let t = Tensor::full(T::from_f32(t_val), shape, dev)?;

        let flow_dir = flow_net.forward(transformer_out, &[&s, &t], &current)?;
        let step_scale = T::from_f32(1.0 / num_steps as f32);
        current = current.add(&flow_dir.scale(step_scale)?)?;
    }
    Ok(current)
}

pub struct FlowLMConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub max_period: f32,
    pub n_bins: usize,
    pub tokenizer_path: String,
    pub lut_dim: usize,
    pub flow_dim: usize,
    pub flow_depth: usize,
    pub ldim: usize,
}

/// Transformer-based flow language model.
pub struct FlowLM<T: WithDTypeF, B: Backend> {
    pub conditioner: LUTConditioner<T, B>,
    flow_net: SimpleMLPAdaLN<T, B>,
    pub transformer: StreamingTransformer<T, B>,
    pub emb_std: Tensor<T, B>,
    pub emb_mean: Tensor<T, B>,
    bos_emb: Tensor<T, B>,
    pub input_linear: Linear<T, B>,
    out_norm_weight: Tensor<T, B>,
    out_norm_bias: Tensor<T, B>,
    out_eos: Linear<T, B>,
    pub dim: usize,
    pub ldim: usize,
}

pub struct FlowLMState<T: WithDTypeF, B: Backend> {
    pub transformer_state: StreamingTransformerState<T, B>,
}

impl<T: WithDTypeF, B: Backend> FlowLM<T, B> {
    pub fn load(vb: &Path<B>, cfg: &FlowLMConfig) -> Result<Self> {
        let conditioner = LUTConditioner::load(
            &vb.pp("conditioner"),
            cfg.n_bins,
            &cfg.tokenizer_path,
            cfg.lut_dim,
            cfg.d_model,
        )?;

        let flow_net = SimpleMLPAdaLN::load(
            &vb.pp("flow_net"),
            cfg.ldim,       // in_channels
            cfg.flow_dim,   // model_channels
            cfg.ldim,       // out_channels
            cfg.d_model,    // cond_channels
            cfg.flow_depth, // num_res_blocks
            2,              // num_time_conds
        )?;

        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            None,
            cfg.dim_feedforward,
            None,
            cfg.max_period,
            crate::mimi_transformer::Kind::FlowLm,
        )?;

        let emb_std = vb.tensor("emb_std", (cfg.ldim,))?;
        let emb_mean = vb.tensor("emb_mean", (cfg.ldim,))?;
        let bos_emb = vb.tensor("bos_emb", (cfg.ldim,))?;
        let input_linear = Linear::load(&vb.pp("input_linear"), cfg.ldim, cfg.d_model)?;
        let out_norm_weight = vb.pp("out_norm").tensor("weight", (cfg.d_model,))?;
        let out_norm_bias = vb.pp("out_norm").tensor("bias", (cfg.d_model,))?;
        let out_eos = Linear::load_b(&vb.pp("out_eos"), cfg.d_model, 1)?;

        Ok(Self {
            conditioner,
            flow_net,
            transformer,
            emb_std,
            emb_mean,
            bos_emb,
            input_linear,
            out_norm_weight,
            out_norm_bias,
            out_eos,
            dim: cfg.d_model,
            ldim: cfg.ldim,
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<FlowLMState<T, B>> {
        let transformer_state = self.transformer.init_state(batch_size, sequence_length)?;
        Ok(FlowLMState { transformer_state })
    }

    /// Run the backbone: concat text_embeddings + input, run transformer, strip prefix.
    fn backbone(
        &self,
        input: &Tensor<T, B>,
        text_embeddings: &Tensor<T, B>,
        seq_len: usize,
        state: &mut FlowLMState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let input = Tensor::cat(&[text_embeddings, input], 1)?;
        let out = self.transformer.forward(&input, &mut state.transformer_state)?;
        let out = out.layer_norm(&self.out_norm_weight, &self.out_norm_bias, 1e-5)?;
        // Remove prefix, keep only last seq_len positions
        let total = out.dim(1usize)?;
        let start = total - seq_len;
        out.narrow(1, start..total)?.contiguous()
    }

    /// Sample next latent using flow matching.
    /// Returns (next_latent [B, 1, ldim], is_eos [B, 1]).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_next_latent(
        &self,
        sequence: &Tensor<T, B>,
        text_embeddings: &Tensor<T, B>,
        state: &mut FlowLMState<T, B>,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor<T, B>, bool)> {
        let (b, s, _) = sequence.dims3()?;
        let dev = sequence.device();

        let sequence = self.replace_nan_with_bos(sequence)?;
        let input = self.input_linear.forward(&sequence)?;
        let transformer_out = self.backbone(&input, text_embeddings, s, state)?;
        let t_len = transformer_out.dim(1usize)?;
        let transformer_out = transformer_out.narrow(1, t_len - 1..t_len)?.contiguous()?;
        let transformer_out = transformer_out.reshape((b, self.dim))?;

        let eos_logit = self.out_eos.forward(&transformer_out)?;
        let eos_val = eos_logit.to_vec()?;
        let is_eos = eos_val[0].to_f32() > eos_threshold;
        let noise_data: Vec<T> = (0..b * self.ldim).map(|_| T::from_f32(rng.sample())).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), dev)?;
        let latent = lsd_decode(&self.flow_net, &transformer_out, &noise, lsd_decode_steps)?;
        let latent = latent.reshape((b, 1, self.ldim))?;
        Ok((latent, is_eos))
    }

    /// Replace NaN values in sequence with bos_emb.
    fn replace_nan_with_bos(&self, sequence: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let data = sequence.to_vec()?;
        // TODO(laurent): avoid the `to_vec` below. For this, we could introduce
        // something like torch.where.
        let bos_data = self.bos_emb.to_vec()?;
        let mut out_data = data.clone();
        let ldim = self.ldim;

        for i in 0..out_data.len() {
            if out_data[i].to_f32().is_nan() {
                out_data[i] = bos_data[i % ldim];
            }
        }
        Tensor::from_vec(out_data, sequence.shape().clone(), sequence.device())
    }
}
