// Fp8 quantization support.
use super::{Device, PTXModule, Storage};
use crate::{Result, Shape, Tensor, WithDType};
use cudarc::cublaslt::{result as lt, sys as lt_sys};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use half::{bf16, f16};
use std::ffi::c_void;
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

        // cuBLASLt FP8 matmul requires TN layout.
        // We compute C^T in col-major: D(N,M) = A^T(N,K) × B(K,M)
        // where A = rhs (row-major [N,K] = col-major [K,N], ld=K)
        //       B = self (row-major [M,K] = col-major [K,M], ld=K)
        //       D = out (row-major [M,N] = col-major [N,M], ld=N)
        let handle = self.device.blas_lt.0;

        // Matmul descriptor: compute in f32, scale type f32.
        let desc = lt::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt_sys::cudaDataType_t::CUDA_R_32F,
        )?;

        let op_t = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
        let op_n = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
        unsafe {
            set_desc_attr(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &op_t,
            )?;
            set_desc_attr(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &op_n,
            )?;
        }

        // Set per-tensor scale pointers.
        let (a_sc_ptr, _ga) = rhs.scales.device_ptr(stream);
        let (b_sc_ptr, _gb) = self.scales.device_ptr(stream);
        let a_sc_p = a_sc_ptr as *const c_void;
        let b_sc_p = b_sc_ptr as *const c_void;
        unsafe {
            set_desc_attr(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &a_sc_p,
            )?;
            set_desc_attr(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &b_sc_p,
            )?;
        }

        // Matrix layouts (FP8 E4M3 for A and B, BF16 for C and D).
        let fp8_type = lt_sys::cudaDataType_t::CUDA_R_8F_E4M3;
        let bf16_type = lt_sys::cudaDataType_t::CUDA_R_16BF;

        // A = rhs: stored as [K, N] col-major, ld = K
        let a_lay = lt::create_matrix_layout(fp8_type, k as u64, n as u64, k as i64)?;
        // B = self: stored as [K, M] col-major, ld = K
        let b_lay = lt::create_matrix_layout(fp8_type, k as u64, m as u64, k as i64)?;
        // C and D: [N, M] col-major, ld = N
        let c_lay = lt::create_matrix_layout(bf16_type, n as u64, m as u64, n as i64)?;
        let d_lay = lt::create_matrix_layout(bf16_type, n as u64, m as u64, n as i64)?;

        // Algorithm selection.
        let pref = lt::create_matmul_pref()?;
        let ws_size = self.device.blas_lt_workspace.len();
        unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const c_void,
                std::mem::size_of::<usize>(),
            )?;
        }

        let heuristic = unsafe {
            lt::get_matmul_algo_heuristic(handle, desc, a_lay, b_lay, c_lay, d_lay, pref)?
        };

        // Launch (scoped to drop borrow guards before moving `out`).
        {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let (a_ptr, _ra) = rhs.data.device_ptr(stream);
            let (b_ptr, _rb) = self.data.device_ptr(stream);
            let (d_ptr, _rd) = out.device_ptr_mut(stream);
            let (w_ptr, _rw) = self.device.blas_lt_workspace.device_ptr(stream);

            unsafe {
                lt::matmul(
                    handle,
                    desc,
                    &alpha as *const f32 as *const c_void,
                    &beta as *const f32 as *const c_void,
                    a_ptr as *const c_void,
                    a_lay,
                    b_ptr as *const c_void,
                    b_lay,
                    d_ptr as *const c_void, // C input (unused, beta=0)
                    c_lay,
                    d_ptr as *mut c_void, // D output
                    d_lay,
                    &heuristic.algo as *const _,
                    w_ptr as *mut c_void,
                    ws_size,
                    stream.cu_stream() as *mut _,
                )?;
            }
        }

        // Cleanup cuBLASLt resources.
        unsafe {
            lt::destroy_matmul_pref(pref)?;
            lt::destroy_matrix_layout(d_lay)?;
            lt::destroy_matrix_layout(c_lay)?;
            lt::destroy_matrix_layout(b_lay)?;
            lt::destroy_matrix_layout(a_lay)?;
            lt::destroy_matmul_desc(desc)?;
        }

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

/// Helper to set a matmul descriptor attribute.
unsafe fn set_desc_attr<T>(
    desc: lt_sys::cublasLtMatmulDesc_t,
    attr: lt_sys::cublasLtMatmulDescAttributes_t,
    val: &T,
) -> Result<()> {
    unsafe {
        lt::set_matmul_desc_attribute(
            desc,
            attr,
            val as *const T as *const c_void,
            std::mem::size_of::<T>(),
        )?;
    }
    Ok(())
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
