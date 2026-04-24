#[cfg(feature = "accelerate")]
mod accelerate;

pub mod backend;
pub mod cpu_backend;
pub mod display;
pub mod dtype;
pub mod error;
pub mod inplace_ops;
pub mod models;
pub mod nn;
pub mod ops;
pub mod quantized;
pub mod safetensors;
pub mod shape;
pub mod streaming;
pub mod tensor;
pub mod tensor_view;
pub mod utils;

pub use backend::Backend;
pub use dtype::{DType, DTypeQ, WithDType, WithDTypeF};
pub use error::{Context, Error, Result};
pub use shape::{D, Dim, Shape};
pub use tensor::{Tensor, TypedTensor};
pub use tensor_view::{TensorOrView, TensorView};
pub use utils::{get_num_cpus, get_num_threads, set_num_threads};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuDevice;
pub type CpuTensor<T> = Tensor<T, CpuDevice>;

pub const CPU: CpuDevice = CpuDevice;

pub(crate) use inplace_ops::{BinaryOp, UnaryOp};

#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub use cuda_backend::Device as CudaDevice;

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}

pub trait Module {
    fn forward<T: WithDType, B: Backend>(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>>;
}

impl<M: Module> Module for Option<&M> {
    fn forward<T: WithDType, B: Backend>(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

pub trait ModuleT {
    type T: WithDTypeF;
    type B: Backend;

    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>>;
}

impl<M: ModuleT> ModuleT for Option<&M> {
    type T = M::T;
    type B = M::B;
    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

pub trait BackendQ: Clone + 'static {
    type T: WithDTypeF;
    type B: Backend;
    type LinearQ: ModuleT<T = Self::T, B = Self::B> + Send + Sync;

    fn from_linear(l: nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ>;

    fn linear_load<V: std::borrow::Borrow<nn::Path<Self::B>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self::LinearQ> {
        let l = nn::Linear::load(vb, in_features, out_features)?;
        Self::from_linear(l)
    }
}

#[derive(Clone)]
pub struct Unquantized<T: WithDTypeF, B: Backend> {
    _marker1: std::marker::PhantomData<(T, B)>,
}

impl<T: WithDTypeF, B: Backend> BackendQ for Unquantized<T, B> {
    type T = T;
    type B = B;
    type LinearQ = nn::Linear<T, B>;
    fn from_linear(l: nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ> {
        Ok(l)
    }
}

pub trait WithQ {
    fn run<Q: BackendQ>(self, dev: Q::B) -> Result<()>;
}

pub fn run_with_device<W: WithQ>(w: W, _cpu_only: bool, _device_id: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if _cpu_only {
            w.run::<Unquantized<f32, _>>(CpuDevice)?;
        } else {
            let dev = cuda_backend::Device::new(_device_id)?;
            w.run::<Unquantized<half::bf16, _>>(dev)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        w.run::<Unquantized<f32, _>>(CpuDevice)?;
    }
    Ok(())
}

pub struct Runner {
    cpu_only: bool,
    dtype: DTypeQ,
}

impl Runner {
    pub fn new() -> Self {
        Self { cpu_only: false, dtype: DTypeQ::BF16 }
    }

    pub fn cpu_only(mut self, cpu_only: bool) -> Self {
        self.cpu_only = cpu_only;
        self
    }

    pub fn dtype(mut self, dtype: DTypeQ) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn run<W: WithQ>(self, w: W, _device_id: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.cpu_only {
                w.run::<Unquantized<f32, _>>(CpuDevice)?;
            } else {
                let dev = cuda_backend::Device::new(_device_id)?;
                match self.dtype {
                    DTypeQ::Fp8 => w.run::<cuda_backend::quantization::Fp8ScalePerTensor>(dev)?,
                    DTypeQ::Fp8PerToken => {
                        w.run::<cuda_backend::quantization::Fp8ScalePerToken>(dev)?
                    }
                    DTypeQ::F16 => w.run::<Unquantized<half::f16, _>>(dev)?,
                    DTypeQ::BF16 => w.run::<Unquantized<half::bf16, _>>(dev)?,
                    DTypeQ::F32 => w.run::<Unquantized<f32, _>>(dev)?,
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            w.run::<Unquantized<f32, _>>(CpuDevice)?;
        }
        Ok(())
    }
}

impl std::default::Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}
