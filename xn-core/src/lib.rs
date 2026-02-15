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
pub mod shape;
pub mod tensor;
pub mod tensor_view;
pub mod utils;

pub use backend::Backend;
pub use dtype::{DType, WithDType, WithDTypeF};
pub use error::{Error, Result};
pub use shape::{D, Dim, Shape};
pub use tensor::Tensor;
pub use tensor_view::{TensorOrView, TensorView};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuDevice;
pub type CpuTensor<T> = Tensor<T, CpuDevice>;

pub const CPU: CpuDevice = CpuDevice;

pub(crate) use inplace_ops::{BinaryOp, UnaryOp};

#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub mod cuda_kernels;

pub fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

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
