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
pub mod safetensors;
pub mod shape;
pub mod streaming;
pub mod tensor;
pub mod tensor_view;
pub mod utils;

pub use backend::Backend;
pub use dtype::{DType, WithDType, WithDTypeF};
pub use error::{Error, Result};
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
pub mod cuda_kernels;
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

pub trait WithDevice {
    fn run<T: WithDTypeF, B: Backend>(dev: B) -> Result<()>;
}

pub fn run_with_device<W: WithDevice>(_use_cpu: bool, _device_id: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if _use_cpu {
            W::run::<f32, _>(CpuDevice)?;
        } else {
            let dev = cuda_backend::Device::new(0)?;
            W::run::<half::bf16, _>(dev)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        W::run::<f32, _>(CpuDevice)?;
    }
    Ok(())
}
