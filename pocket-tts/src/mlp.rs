use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

fn modulate<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    shift: &Tensor<T, B>,
    scale: &Tensor<T, B>,
) -> Result<Tensor<T, B>> {
    let one_plus_scale = scale.add_scalar(T::from_f32(1.0))?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

// ---- TimestepEmbedder ----

pub struct TimestepEmbedder<T: WithDTypeF, B: Backend> {
    linear1: Linear<T, B>,
    linear2: Linear<T, B>,
    rms_norm: LayerNorm<T, B>,
    freqs: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> TimestepEmbedder<T, B> {
    pub fn load(vb: &Path<B>, hidden_size: usize, frequency_embedding_size: usize) -> Result<Self> {
        let mlp = vb.pp("mlp");
        let linear1 = Linear::load_b(&mlp.pp("0"), frequency_embedding_size, hidden_size)?;
        let linear2 = Linear::load_b(&mlp.pp("2"), hidden_size, hidden_size)?;

        // This is slightly different from the python implementation which uses an unbiased
        // variance estimate whereas the op in mimi-rs uses a biased one.
        // This should not have much impact but maybe we want to have the option to make the
        // estimator unbiased in layer-norm.
        let ln_w = mlp.tensor("3.alpha", (hidden_size,))?;
        let ln_b = ln_w.zeros_like()?;
        let rms_norm = LayerNorm::new(ln_w, ln_b, 1e-5).remove_mean(false);

        let freqs = vb.tensor("freqs", (frequency_embedding_size / 2,))?;

        Ok(Self { linear1, linear2, rms_norm, freqs })
    }

    #[tracing::instrument(name = "ts-embedder", skip_all)]
    pub fn forward(&self, t: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // t: [..., 1] -> frequency embedding
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let embedding = Tensor::cat(&[&cos, &sin], embedding_last_dim(&cos)?)?;

        // MLP: linear -> silu -> linear -> rmsnorm
        let mut x = self.linear1.forward(&embedding)?;
        x = x.silu()?;
        x = self.linear2.forward(&x)?;
        x = self.rms_norm.forward(&x)?;
        Ok(x)
    }
}

fn embedding_last_dim<T: WithDTypeF, B: Backend>(t: &Tensor<T, B>) -> Result<usize> {
    Ok(t.rank() - 1)
}

// ---- ResBlock ----

pub struct ResBlock<T: WithDTypeF, B: Backend> {
    in_ln: LayerNorm<T, B>,
    mlp_linear1: Linear<T, B>,
    mlp_linear2: Linear<T, B>,
    ada_ln_silu_linear: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> ResBlock<T, B> {
    pub fn load(vb: &Path<B>, channels: usize) -> Result<Self> {
        let in_ln = LayerNorm::load(&vb.pp("in_ln"), channels, 1e-6)?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = Linear::load_b(&mlp.pp("0"), channels, channels)?;
        let mlp_linear2 = Linear::load_b(&mlp.pp("2"), channels, channels)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = Linear::load_b(&ada.pp("1"), channels, 3 * channels)?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    #[tracing::instrument(name = "resblock", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>, y: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = self.ada_ln_silu_linear.forward(&y.silu()?)?;
        let channels = x.dim(xn::D::Minus1)?;
        let shift_mlp = ada.narrow(ada.rank() - 1, 0..channels)?.contiguous()?;
        let scale_mlp = ada.narrow(ada.rank() - 1, channels..2 * channels)?.contiguous()?;
        let gate_mlp = ada.narrow(ada.rank() - 1, 2 * channels..3 * channels)?.contiguous()?;

        // h = modulate(ln(x), shift, scale)
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift_mlp, &scale_mlp)?;

        // MLP
        let h = self.mlp_linear1.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp_linear2.forward(&h)?;
        x.add(&gate_mlp.broadcast_mul(&h)?)
    }
}

// ---- FinalLayer ----

pub struct FinalLayer<T: WithDTypeF, B: Backend> {
    norm_final: LayerNorm<T, B>,
    linear: Linear<T, B>,
    ada_ln_silu_linear: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> FinalLayer<T, B> {
    pub fn load(vb: &Path<B>, model_channels: usize, out_channels: usize) -> Result<Self> {
        let zeros = Tensor::zeros(model_channels, vb.device())?;
        let ones = zeros.add_scalar(T::from_f32(1.0))?;
        let norm_final = LayerNorm::new(ones, zeros, 1e-6);
        let linear = Linear::load_b(&vb.pp("linear"), model_channels, out_channels)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = Linear::load_b(&ada.pp("1"), model_channels, 2 * model_channels)?;
        Ok(Self { norm_final, linear, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor<T, B>, c: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = self.ada_ln_silu_linear.forward(&c.silu()?)?;
        let model_channels = x.dim(xn::D::Minus1)?;
        let shift = ada.narrow(ada.rank() - 1, 0..model_channels)?.contiguous()?;
        let scale = ada.narrow(ada.rank() - 1, model_channels..2 * model_channels)?.contiguous()?;

        let x = self.norm_final.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

// ---- SimpleMLPAdaLN ----

pub struct SimpleMLPAdaLN<T: WithDTypeF, B: Backend> {
    time_embeds: Vec<TimestepEmbedder<T, B>>,
    cond_embed: Linear<T, B>,
    input_proj: Linear<T, B>,
    res_blocks: Vec<ResBlock<T, B>>,
    final_layer: FinalLayer<T, B>,
    pub num_time_conds: usize,
}

impl<T: WithDTypeF, B: Backend> SimpleMLPAdaLN<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load(
                &vb.pp("time_embed").pp(i),
                model_channels,
                256,
            )?);
        }

        let cond_embed = Linear::load_b(&vb.pp("cond_embed"), cond_channels, model_channels)?;
        let input_proj = Linear::load_b(&vb.pp("input_proj"), in_channels, model_channels)?;

        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::load(&vb.pp("res_blocks").pp(i), model_channels)?);
        }

        let final_layer = FinalLayer::load(&vb.pp("final_layer"), model_channels, out_channels)?;

        Ok(Self { time_embeds, cond_embed, input_proj, res_blocks, final_layer, num_time_conds })
    }

    /// Forward pass.
    /// c: conditioning from AR transformer
    /// s: start time tensor
    /// t: target time tensor
    /// x: input tensor [N, C]
    #[tracing::instrument(name = "mlp-adaln", skip_all)]
    pub fn forward(
        &self,
        c: &Tensor<T, B>,
        ts: &[&Tensor<T, B>],
        x: &Tensor<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut x = self.input_proj.forward(x)?;
        let mut t_combined = self.time_embeds[0].forward(ts[0])?;
        for (embed, &t_input) in self.time_embeds[1..].iter().zip(ts[1..].iter()) {
            t_combined = t_combined.add(&embed.forward(t_input)?)?;
        }
        let scale = T::from_f32(1.0 / self.num_time_conds as f32);
        t_combined = t_combined.scale(scale)?;

        let c = self.cond_embed.forward(c)?;
        let y = t_combined.add(&c)?;
        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }
        self.final_layer.forward(&x, &y)
    }
}
