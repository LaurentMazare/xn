use super::{Device, PTXModule};
use crate::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use half::bf16;

pub struct Fp8Tensor {
    pub data: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub device: Device,
}

/// Quantize a contiguous bf16 buffer to FP8 E4M3 using dynamic per-tensor scaling.
///
/// `src` is a contiguous bf16 slice of `num_tokens * hidden_size` elements, laid out
/// as `[num_tokens, hidden_size]` in row-major order.
///
/// Returns an `Fp8Tensor` with:
/// - `data`: `num_tokens * hidden_size` u8 values (FP8 E4M3 encoded)
/// - `scales`: a single f32 scale value (absmax / 448.0)
pub fn quantize_fp8(
    device: &Device,
    src: &CudaSlice<bf16>,
    num_tokens: usize,
    hidden_size: usize,
) -> Result<Fp8Tensor> {
    let numel = num_tokens * hidden_size;
    assert!(src.len() >= numel, "src too small: {} < {}", src.len(), numel);

    // Allocate scale on device, zero-initialized (the reduction kernel uses atomicMax
    // starting from 0).
    let scale: CudaSlice<f32> = device.stream().clone_htod(&[0.0f32])?;

    // Allocate output buffer.
    let mut out: CudaSlice<u8> = unsafe { device.stream().alloc::<u8>(numel) }?;

    // --- Pass 1: compute per-tensor absmax -> scale = absmax / FP8_E4M3_MAX ---
    {
        let func = device.get_func("segmented_max_reduction_bf16", PTXModule::Fp8)?;
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

    // --- Pass 2: quantize bf16 -> fp8 using the computed scale ---
    {
        let func = device.get_func("scaled_fp8_quant_dynamic_bf16", PTXModule::Fp8)?;
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

    Ok(Fp8Tensor { data: out, scales: scale, device: device.clone() })
}
