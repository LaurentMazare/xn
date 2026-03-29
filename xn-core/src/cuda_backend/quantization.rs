// Fp8 quantization support.
use super::{Device, PTXModule, Storage};
use crate::{Result, Shape, Tensor, WithDType};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use half::{bf16, f16};
use std::sync::{Arc, RwLock};

/// Trait for types that can be quantized to/from FP8.
pub trait Fp8Quantizable: WithDType {
    /// Suffix used in kernel names, e.g. "bf16", "f16", or "f32".
    fn fp8_suffix() -> &'static str;
}

impl Fp8Quantizable for bf16 {
    fn fp8_suffix() -> &'static str {
        "bf16"
    }
}

impl Fp8Quantizable for f16 {
    fn fp8_suffix() -> &'static str {
        "f16"
    }
}

impl Fp8Quantizable for f32 {
    fn fp8_suffix() -> &'static str {
        "f32"
    }
}

pub struct Fp8Tensor {
    pub data: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub device: Device,
    pub shape: Shape,
}

impl Fp8Tensor {
    /// Quantize a `Tensor<T, Device>` into an `Fp8Tensor` using dynamic per-tensor scaling.
    pub fn quantize<T: Fp8Quantizable>(src: &Tensor<T, Device>) -> Result<Self> {
        let shape = src.shape();
        if shape.rank() < 2 {
            crate::bail!("quantize_fp8 requires at least 2 dimensions, got {}", shape.rank());
        }
        let dims = shape.dims();
        let hidden_size = dims[shape.rank() - 1];
        let num_tokens: usize = dims[..shape.rank() - 1].iter().product();
        let storage = src.storage()?;
        quantize_fp8(&storage.device, &storage.data, num_tokens, hidden_size, shape.clone())
    }

    /// Dequantize this `Fp8Tensor` back to a `Tensor<T, Device>`.
    pub fn dequantize<T: Fp8Quantizable>(&self) -> Result<Tensor<T, Device>> {
        let numel = self.shape.elem_count();
        let mut out: CudaSlice<T> = unsafe { self.device.stream().alloc::<T>(numel) }?;

        let kname = format!("fp8_dequant_{}", T::fp8_suffix());
        let func = self.device.get_func(&kname, PTXModule::Fp8)?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let n = numel as u32;

        let mut args = self.device.stream().launch_builder(&func);
        args.arg(&mut out);
        args.arg(&self.data);
        args.arg(&self.scales);
        args.arg(&n);
        unsafe { args.launch(cfg) }?;

        let storage = Storage { data: out, device: self.device.clone() };
        Ok(Tensor {
            data: Arc::new(RwLock::new(storage)),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// FP8 matrix multiplication: `C = self × rhs^T` with output in bf16.
    ///
    /// This computes a standard linear-layer matmul where:
    /// - `self` has shape `[M, K]` (e.g. activations)
    /// - `rhs` has shape `[N, K]` (e.g. weight matrix `[out_features, in_features]`)
    /// - Result has shape `[M, N]` in bf16
    ///
    /// The per-tensor scales from both operands are applied by cuBLASLt.
    /// Requires a GPU with compute capability >= 8.9 (Ada Lovelace / Hopper).
    pub fn matmul_t(&self, rhs: &Fp8Tensor) -> Result<Tensor<bf16, Device>> {
        let self_dims = self.shape.dims();
        let rhs_dims = rhs.shape.dims();
        if self_dims.len() < 2 || rhs_dims.len() < 2 {
            crate::bail!("matmul_t requires at least 2D tensors");
        }
        let m = self_dims[self_dims.len() - 2];
        let k = self_dims[self_dims.len() - 1];
        let n = rhs_dims[rhs_dims.len() - 2];
        let k2 = rhs_dims[rhs_dims.len() - 1];
        if k != k2 {
            crate::bail!(
                "matmul_t dimension mismatch: self [..., {m}, {k}] vs rhs [..., {n}, {k2}]"
            );
        }

        let stream = self.device.stream();
        let mut out: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m * n) }?;

        self.device.blas_lt.matmul_f8(
            &rhs.data,
            &self.data,
            &rhs.scales,
            &self.scales,
            &mut out,
            m,
            n,
            k,
        )?;

        let out_shape: Shape = (m, n).into();
        let storage = Storage { data: out, device: self.device.clone() };
        Ok(Tensor {
            data: Arc::new(RwLock::new(storage)),
            shape: out_shape,
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        })
    }
}

/// Quantize a contiguous buffer to FP8 E4M3 using dynamic per-tensor scaling.
///
/// `src` is a contiguous slice of `num_tokens * hidden_size` elements, laid out
/// as `[num_tokens, hidden_size]` in row-major order.
///
/// Returns an `Fp8Tensor` with:
/// - `data`: `num_tokens * hidden_size` u8 values (FP8 E4M3 encoded)
/// - `scales`: a single f32 scale value (absmax / 448.0)
pub fn quantize_fp8<T: Fp8Quantizable>(
    device: &Device,
    src: &CudaSlice<T>,
    num_tokens: usize,
    hidden_size: usize,
    shape: Shape,
) -> Result<Fp8Tensor> {
    let numel = num_tokens * hidden_size;
    assert!(src.len() >= numel, "src too small: {} < {}", src.len(), numel);

    let suffix = T::fp8_suffix();

    // Allocate scale on device, zero-initialized (the reduction kernel uses atomicMax
    // starting from 0).
    let scale: CudaSlice<f32> = device.stream().clone_htod(&[0.0f32])?;

    // Allocate output buffer.
    let mut out: CudaSlice<u8> = unsafe { device.stream().alloc::<u8>(numel) }?;

    // --- Pass 1: compute per-tensor absmax -> scale = absmax / FP8_E4M3_MAX ---
    {
        let kname = format!("segmented_max_reduction_{suffix}");
        let func = device.get_func(&kname, PTXModule::Fp8)?;
        let block_dim = 256u32;
        let grid_dim = num_tokens as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let hs = hidden_size as i32;
        let in_row_stride = hidden_size as i64;
        let nt = num_tokens as i64;

        let mut args = device.stream().launch_builder(&func);
        args.arg(&scale);
        args.arg(src);
        args.arg(&hs);
        args.arg(&in_row_stride);
        args.arg(&nt);
        unsafe { args.launch(cfg) }?;
    }

    // --- Pass 2: quantize -> fp8 using the computed scale ---
    {
        let kname = format!("scaled_fp8_quant_dynamic_{suffix}");
        let func = device.get_func(&kname, PTXModule::Fp8)?;
        let block_dim = 256u32;
        let grid_dim = num_tokens as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let hs = hidden_size as i32;
        let in_row_stride = hidden_size as i64;
        let out_row_stride = hidden_size as i64;

        let mut args = device.stream().launch_builder(&func);
        args.arg(&mut out);
        args.arg(src);
        args.arg(&scale);
        args.arg(&hs);
        args.arg(&in_row_stride);
        args.arg(&out_row_stride);
        unsafe { args.launch(cfg) }?;
    }

    Ok(Fp8Tensor { data: out, scales: scale, device: device.clone(), shape })
}
