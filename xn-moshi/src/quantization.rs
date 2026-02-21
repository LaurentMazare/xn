use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

// ============================================================================
// EuclideanCodebook
// ============================================================================

pub struct EuclideanCodebook<T: WithDTypeF, B: Backend> {
    embedding: Tensor<T, B>,
    c2: Tensor<T, B>,
    dim: usize,
}

impl<T: WithDTypeF, B: Backend> EuclideanCodebook<T, B> {
    pub fn load(vb: &Path<B>, dim: usize, codebook_size: usize) -> Result<Self> {
        let epsilon = 1e-5;
        let cluster_usage = vb.tensor::<T>("cluster_usage", (codebook_size,))?;
        let embedding_sum = vb.tensor::<T>("embedding_sum", (codebook_size, dim))?;

        // embedding = embedding_sum / max(cluster_usage, epsilon)
        let epsilon_t = Tensor::full(
            T::from_f32(epsilon),
            (codebook_size,),
            cluster_usage.device(),
        )?;
        let cluster_usage = cluster_usage.maximum(&epsilon_t)?;
        let cluster_usage = cluster_usage.unsqueeze(1)?;
        let embedding = embedding_sum.broadcast_div(&cluster_usage)?;

        // Precompute c2 = (embedding * embedding).sum(dim=-1) / 2.0
        let c2 = embedding
            .sqr()?
            .sum_keepdim(vec![1])?
            .scale(T::from_f32(0.5))?;
        let c2 = c2.reshape((codebook_size,))?;

        Ok(Self { embedding, c2, dim })
    }

    // TODO: This uses the matmul-based "encode_slow" path. The reference implementation
    // has a custom op (CodebookEncode) that computes pairwise distances with rayon parallelism,
    // avoiding the N*codebook_size intermediate tensor. Would require xn CustomOp2 support.
    #[tracing::instrument(name = "ec-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let mut target_shape: Vec<usize> = xs.dims().to_vec();
        target_shape.pop();

        // Flatten to 2D: [*, dim] -> [N, dim]
        let xs = xs.flatten(0, xs.rank().saturating_sub(2))?;

        // dist = c2 - dot_prod (up to constant ||x||^2/2 which doesn't affect argmin)
        let dot_prod = xs.matmul_t(&self.embedding)?;
        let dists = self.c2.broadcast_sub(&dot_prod)?;

        let codes = dists.argmin(1)?;
        if target_shape.is_empty() {
            Ok(codes)
        } else {
            codes.reshape(target_shape)
        }
    }

    #[tracing::instrument(name = "ec-decode", skip_all)]
    pub fn decode(&self, indices: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let mut final_dims = indices.dims().to_vec();
        final_dims.push(self.dim);

        let flat_indices = indices.flatten(0, indices.rank().saturating_sub(1))?;
        let indices_vec = flat_indices.to_vec()?;
        let indices_u32: Vec<u32> = indices_vec.iter().map(|&x| x as u32).collect();
        let values = self.embedding.index_select(&indices_u32, 0)?;
        values.reshape(final_dims)
    }
}

// ============================================================================
// VectorQuantization
// ============================================================================

pub struct VectorQuantization<T: WithDTypeF, B: Backend> {
    project_in: Option<Tensor<T, B>>,
    project_out: Option<Tensor<T, B>>,
    codebook: EuclideanCodebook<T, B>,
}

impl<T: WithDTypeF, B: Backend> VectorQuantization<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
    ) -> Result<Self> {
        let codebook_dim = codebook_dim.unwrap_or(dim);
        let (project_in, project_out) = if codebook_dim == dim {
            (None, None)
        } else {
            let p_in = vb.pp("project_in").tensor("weight", (codebook_dim, dim))?;
            let p_out = vb.pp("project_out").tensor("weight", (dim, codebook_dim))?;
            (Some(p_in), Some(p_out))
        };
        let codebook = EuclideanCodebook::load(&vb.pp("_codebook"), codebook_dim, codebook_size)?;
        Ok(Self {
            project_in,
            project_out,
            codebook,
        })
    }

    #[tracing::instrument(name = "vq-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        // xs: [B, C, T] -> transpose -> [B, T, C]
        let xs = xs.t()?.contiguous()?;
        let xs = match &self.project_in {
            Some(proj) => xs.matmul_t(proj)?,
            None => xs,
        };
        self.codebook.encode(&xs)
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let quantized = self.codebook.decode(codes)?;
        let quantized = match &self.project_out {
            Some(proj) => quantized.matmul_t(proj)?,
            None => quantized,
        };
        quantized.t()?.contiguous()
    }
}

// ============================================================================
// ResidualVectorQuantization
// ============================================================================

pub struct ResidualVectorQuantization<T: WithDTypeF, B: Backend> {
    layers: Vec<VectorQuantization<T, B>>,
}

impl<T: WithDTypeF, B: Backend> ResidualVectorQuantization<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_q: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let layer = VectorQuantization::load(&vb.pp(i), dim, codebook_size, codebook_dim)?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let mut codes = Vec::with_capacity(self.layers.len());
        let mut residual = xs.clone();
        for layer in &self.layers {
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            residual = residual.sub(&quantized)?;
            codes.push(indices);
        }
        let codes_refs: Vec<&Tensor<i64, B>> = codes.iter().collect();
        Tensor::stack(&codes_refs, 0)
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        if self.layers.is_empty() {
            xn::bail!("empty layers in ResidualVectorQuantization");
        }
        let inner_shape: Vec<usize> = codes.dims()[1..].to_vec();
        let mut quantized = self.layers[0].decode(
            &codes
                .narrow(0, ..1)?
                .contiguous()?
                .reshape(inner_shape.clone())?,
        )?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            let layer_codes = codes
                .narrow(0, i..i + 1)?
                .contiguous()?
                .reshape(inner_shape.clone())?;
            quantized = quantized.add(&layer.decode(&layer_codes)?)?;
        }
        Ok(quantized)
    }
}

// ============================================================================
// ResidualVectorQuantizer (with input/output projections)
// ============================================================================

pub struct ResidualVectorQuantizer<T: WithDTypeF, B: Backend> {
    vq: ResidualVectorQuantization<T, B>,
    input_proj: Option<Tensor<T, B>>,
    output_proj: Option<Tensor<T, B>>,
}

impl<T: WithDTypeF, B: Backend> ResidualVectorQuantizer<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        force_projection: bool,
    ) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let output_dim = output_dim.unwrap_or(dim);

        let input_proj = if input_dim != dim || force_projection {
            Some(vb.pp("input_proj").tensor("weight", (dim, input_dim, 1))?)
        } else {
            None
        };
        let output_proj = if output_dim != dim || force_projection {
            Some(
                vb.pp("output_proj")
                    .tensor("weight", (output_dim, dim, 1))?,
            )
        } else {
            None
        };

        let vq = ResidualVectorQuantization::load(&vb.pp("vq"), n_q, dim, bins, None)?;
        Ok(Self {
            vq,
            input_proj,
            output_proj,
        })
    }

    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let xs = match &self.input_proj {
            Some(proj) => xs.conv1d(proj, None, 1, 0, 1, 1)?,
            None => xs.clone(),
        };
        let codes = self.vq.encode(&xs)?;
        codes.transpose(0, 1)?.contiguous()
    }

    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let codes = codes.transpose(0, 1)?.contiguous()?;
        let quantized = self.vq.decode(&codes)?;
        match &self.output_proj {
            Some(proj) => quantized.conv1d(proj, None, 1, 0, 1, 1),
            None => Ok(quantized),
        }
    }
}

// ============================================================================
// SplitResidualVectorQuantizer
// ============================================================================

pub struct SplitResidualVectorQuantizer<T: WithDTypeF, B: Backend> {
    rvq_first: ResidualVectorQuantizer<T, B>,
    rvq_rest: ResidualVectorQuantizer<T, B>,
    n_q: usize,
}

impl<T: WithDTypeF, B: Backend> SplitResidualVectorQuantizer<T, B> {
    pub fn load(
        vb: &Path<B>,
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
    ) -> Result<Self> {
        let rvq_first = ResidualVectorQuantizer::load(
            &vb.pp("rvq_first"),
            dim,
            input_dim,
            output_dim,
            1,
            bins,
            true,
        )?;
        let rvq_rest = ResidualVectorQuantizer::load(
            &vb.pp("rvq_rest"),
            dim,
            input_dim,
            output_dim,
            n_q - 1,
            bins,
            true,
        )?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q,
        })
    }

    #[tracing::instrument(name = "rvq-encode", skip_all)]
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let codes = self.rvq_first.encode(xs)?;
        if self.n_q > 1 {
            let rest_codes = self.rvq_rest.encode(xs)?;
            Tensor::cat(&[&codes, &rest_codes], 1)
        } else {
            Ok(codes)
        }
    }

    #[tracing::instrument(name = "rvq-decode", skip_all)]
    pub fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let first_codes = codes.narrow(1, ..1)?.contiguous()?;
        let quantized = self.rvq_first.decode(&first_codes)?;
        if self.n_q > 1 {
            let rest_codes = codes.narrow(1, 1..self.n_q)?.contiguous()?;
            quantized.add(&self.rvq_rest.decode(&rest_codes)?)
        } else {
            Ok(quantized)
        }
    }
}
