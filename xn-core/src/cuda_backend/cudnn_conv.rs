//! Optional cuDNN path for f32 conv1d / conv_transpose1d, enabled with
//! XN_CUDNN=1. 1D convolutions run as 4D NCHW convolutions with H=1, which
//! writes the output directly in (b, c, l) layout — no im2col buffer and no
//! transpose. conv1d fuses the per-channel bias via
//! cudnnConvolutionBiasActivationForward with an identity activation;
//! conv_transpose1d maps to cudnnConvolutionBackwardData (bias not fused).
use super::{Device, Storage};
use crate::Result;
use cudarc::cudnn::{Cudnn, sys};
use cudarc::driver::CudaSlice;
use std::sync::{Arc, Mutex};

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XN_CUDNN").is_ok_and(|v| v == "1"))
}

fn tf32() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XN_TF32").is_ok_and(|v| v == "1"))
}

/// XN_CUDNN_AUTOTUNE=0 disables the measured algorithm search and falls back to
/// the cuDNN v7 heuristic. Autotuning is what XLA does: its dumped conv configs
/// carry per-shape engine ids and tuning knobs picked by measurement, not by
/// `cudnnGetConvolutionForwardAlgorithm_v7`.
fn autotune() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XN_CUDNN_AUTOTUNE").is_ok_and(|v| v == "0"))
}

/// Identifies a conv shape for the algorithm cache. Two calls with the same key
/// run the same cuDNN problem, so the measured winner carries over.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct ConvKey {
    batch: usize,
    in_c: usize,
    out_c: usize,
    length: usize,
    out_length: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

/// Lazily-created cuDNN handle plus the per-shape algorithm caches. `Cudnn` is
/// not Send/Sync because cuDNN handles require externally serialized access;
/// the mutex is held for the whole descriptor-setup + launch sequence, which
/// provides that serialization.
pub(crate) struct CudnnCtx {
    handle: Mutex<Option<Arc<Cudnn>>>,
    fwd: Mutex<std::collections::HashMap<ConvKey, sys::cudnnConvolutionFwdAlgo_t>>,
    bwd: Mutex<std::collections::HashMap<ConvKey, sys::cudnnConvolutionBwdDataAlgo_t>>,
}
unsafe impl Send for CudnnCtx {}
unsafe impl Sync for CudnnCtx {}

impl CudnnCtx {
    pub(crate) fn new() -> Self {
        Self {
            handle: Mutex::new(None),
            fwd: Mutex::new(Default::default()),
            bwd: Mutex::new(Default::default()),
        }
    }
}

fn with_cudnn<R>(device: &Device, f: impl FnOnce(&Arc<Cudnn>) -> Result<R>) -> Result<R> {
    let mut guard = device.cudnn.handle.lock().unwrap();
    if guard.is_none() {
        *guard = Some(Cudnn::new(device.stream().clone())?);
    }
    f(guard.as_ref().unwrap())
}

/// Every forward algorithm cuDNN defines, as autotuning candidates. Ones that
/// do not support the problem fail at the workspace query and are skipped.
const FWD_ALGOS: [sys::cudnnConvolutionFwdAlgo_t; 8] = {
    use sys::cudnnConvolutionFwdAlgo_t as A;
    [
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    ]
};

const BWD_ALGOS: [sys::cudnnConvolutionBwdDataAlgo_t; 6] = {
    use sys::cudnnConvolutionBwdDataAlgo_t as A;
    [
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        A::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
    ]
};

/// Time `run` for each candidate and return the fastest, mirroring XLA's
/// autotuner: 2 warmup plus 5 timed reps per candidate, device-synchronized.
/// Candidates that error (unsupported for this problem, or out of memory) are
/// skipped; if every one fails the caller's fallback algorithm is returned.
fn autotune_algo<A: Copy>(
    device: &Device,
    candidates: &[A],
    fallback: A,
    mut run: impl FnMut(A) -> Result<()>,
) -> A {
    let mut best = (f64::INFINITY, fallback);
    for &algo in candidates {
        if run(algo).is_err() {
            continue;
        }
        for _ in 0..2 {
            if run(algo).is_err() {
                continue;
            }
        }
        if device.stream().synchronize().is_err() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut ok = true;
        for _ in 0..5 {
            if run(algo).is_err() {
                ok = false;
                break;
            }
        }
        if !ok || device.stream().synchronize().is_err() {
            continue;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best.0 {
            best = (elapsed, algo);
        }
    }
    best.1
}

/// XN_CUDNN_NOFUSE=1: skip the fused bias epilogue so the forward conv can use
/// the heuristically-picked algorithm instead of IMPLICIT_PRECOMP_GEMM, which
/// is the only algorithm the identity activation supports. Bias is then added
/// by the caller. Experiment gate: fusing bias costs the tensor-core path.
fn nofuse_bias() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XN_CUDNN_NOFUSE").is_ok_and(|v| v == "1"))
}

fn math_type() -> sys::cudnnMathType_t {
    if tf32() {
        sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
    } else {
        sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_f32(
    dst: &mut Storage<f32>,
    src: &Storage<f32>,
    kernel: &Storage<f32>,
    bias: Option<&CudaSlice<f32>>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<bool> {
    let device = dst.device.clone();
    with_cudnn(&device, |cudnn| {
        const NCHW: sys::cudnnTensorFormat_t = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let x_desc = cudnn
            .create_4d_tensor::<f32>(NCHW, [batch as i32, in_channels as i32, 1, length as i32])?;
        let w_desc = cudnn.create_4d_filter::<f32>(
            NCHW,
            [out_channels as i32, in_channels as i32, 1, kernel_size as i32],
        )?;
        let y_desc = cudnn.create_4d_tensor::<f32>(
            NCHW,
            [batch as i32, out_channels as i32, 1, out_length as i32],
        )?;
        let mut conv = cudnn.create_conv2d::<f32>(
            [0, padding as i32],
            [1, stride as i32],
            [1, dilation as i32],
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        conv.set_math_type(math_type())?;

        match bias.filter(|_| !nofuse_bias()) {
            Some(bias) => {
                let bias_desc =
                    cudnn.create_4d_tensor::<f32>(NCHW, [1, out_channels as i32, 1, 1])?;
                let act = cudnn.create_activation::<f32>(
                    sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                    sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    0.0,
                )?;
                let op = cudarc::cudnn::ConvBiasActivationForward::<f32, f32, f32, f32> {
                    conv: &conv,
                    act: &act,
                    x: &x_desc,
                    w: &w_desc,
                    z: &y_desc,
                    bias: &bias_desc,
                    y: &y_desc,
                };
                // The identity activation only supports this algorithm.
                let algo = sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
                let ws_size = op.get_workspace_size(algo)?;
                let mut workspace: CudaSlice<u8> =
                    unsafe { device.stream().alloc(ws_size.max(1)) }?;
                // alpha2 = 0 so the z tensor is never dereferenced; src stands
                // in as a valid device pointer of the right dtype.
                unsafe {
                    op.launch(
                        algo,
                        Some(&mut workspace),
                        (1.0f32, 0.0f32),
                        &*src.data,
                        &*kernel.data,
                        &*src.data,
                        bias,
                        &mut *dst.data,
                    )
                }?;
                return Ok(true);
            }
            None => {
                let op = cudarc::cudnn::ConvForward::<f32, f32, f32> {
                    conv: &conv,
                    x: &x_desc,
                    w: &w_desc,
                    y: &y_desc,
                };
                let key = ConvKey {
                    batch,
                    in_c: in_channels,
                    out_c: out_channels,
                    length,
                    out_length,
                    k: kernel_size,
                    stride,
                    padding,
                    dilation,
                };
                let cached = device.cudnn.fwd.lock().unwrap().get(&key).copied();
                let algo = match cached {
                    Some(algo) => algo,
                    None => {
                        let heuristic = op.pick_algorithm()?;
                        let algo = if autotune() {
                            autotune_algo(&device, &FWD_ALGOS, heuristic, |algo| {
                                let ws = op.get_workspace_size(algo)?;
                                let mut workspace: CudaSlice<u8> =
                                    unsafe { device.stream().alloc(ws.max(1)) }?;
                                unsafe {
                                    op.launch(
                                        algo,
                                        Some(&mut workspace),
                                        (1.0f32, 0.0f32),
                                        &*src.data,
                                        &*kernel.data,
                                        &mut *dst.data,
                                    )
                                }?;
                                Ok(())
                            })
                        } else {
                            heuristic
                        };
                        device.cudnn.fwd.lock().unwrap().insert(key, algo);
                        algo
                    }
                };
                let ws_size = op.get_workspace_size(algo)?;
                let mut workspace: CudaSlice<u8> =
                    unsafe { device.stream().alloc(ws_size.max(1)) }?;
                unsafe {
                    op.launch(
                        algo,
                        Some(&mut workspace),
                        (1.0f32, 0.0f32),
                        &*src.data,
                        &*kernel.data,
                        &mut *dst.data,
                    )
                }?;
            }
        }
        Ok(false)
    })
}

/// Transposed convolution as the data-gradient of the equivalent forward
/// convolution. The xn kernel layout (in_channels, out_channels, k) is
/// exactly cuDNN's filter layout for that virtual forward conv.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose1d_f32(
    dst: &mut Storage<f32>,
    src: &Storage<f32>,
    kernel: &Storage<f32>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    let device = dst.device.clone();
    with_cudnn(&device, |cudnn| {
        const NCHW: sys::cudnnTensorFormat_t = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let dy_desc = cudnn
            .create_4d_tensor::<f32>(NCHW, [batch as i32, in_channels as i32, 1, length as i32])?;
        let w_desc = cudnn.create_4d_filter::<f32>(
            NCHW,
            [in_channels as i32, out_channels as i32, 1, kernel_size as i32],
        )?;
        let dx_desc = cudnn.create_4d_tensor::<f32>(
            NCHW,
            [batch as i32, out_channels as i32, 1, out_length as i32],
        )?;
        let mut conv = cudnn.create_conv2d::<f32>(
            [0, 0],
            [1, stride as i32],
            [1, 1],
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        conv.set_math_type(math_type())?;

        let op = cudarc::cudnn::ConvBackwardData::<f32, f32, f32> {
            conv: &conv,
            dx: &dx_desc,
            w: &w_desc,
            dy: &dy_desc,
        };
        let key = ConvKey {
            batch,
            in_c: in_channels,
            out_c: out_channels,
            length,
            out_length,
            k: kernel_size,
            stride,
            padding: 0,
            dilation: 1,
        };
        let cached = device.cudnn.bwd.lock().unwrap().get(&key).copied();
        let algo = match cached {
            Some(algo) => algo,
            None => {
                let heuristic = op.pick_algorithm()?;
                let algo = if autotune() {
                    autotune_algo(&device, &BWD_ALGOS, heuristic, |algo| {
                        let ws = op.get_workspace_size(algo)?;
                        let mut workspace: CudaSlice<u8> =
                            unsafe { device.stream().alloc(ws.max(1)) }?;
                        unsafe {
                            op.launch(
                                algo,
                                Some(&mut workspace),
                                (1.0f32, 0.0f32),
                                &mut *dst.data,
                                &*kernel.data,
                                &*src.data,
                            )
                        }?;
                        Ok(())
                    })
                } else {
                    heuristic
                };
                device.cudnn.bwd.lock().unwrap().insert(key, algo);
                algo
            }
        };
        let ws_size = op.get_workspace_size(algo)?;
        let mut workspace: CudaSlice<u8> = unsafe { device.stream().alloc(ws_size.max(1)) }?;
        unsafe {
            op.launch(
                algo,
                Some(&mut workspace),
                (1.0f32, 0.0f32),
                &mut *dst.data,
                &*kernel.data,
                &*src.data,
            )
        }?;
        Ok(())
    })
}
