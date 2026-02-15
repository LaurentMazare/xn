#![allow(clippy::too_many_arguments)]
use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PTXModule {
    Arithmetic,
    Broadcast,
    Conv,
    Fill,
    Indexing,
    Layout,
    Reduce,
    Rope,
}

#[derive(Default)]
struct ModuleCache {
    arithmetic: Option<Arc<cudarc::driver::CudaModule>>,
    broadcast: Option<Arc<cudarc::driver::CudaModule>>,
    conv: Option<Arc<cudarc::driver::CudaModule>>,
    fill: Option<Arc<cudarc::driver::CudaModule>>,
    indexing: Option<Arc<cudarc::driver::CudaModule>>,
    layout: Option<Arc<cudarc::driver::CudaModule>>,
    reduce: Option<Arc<cudarc::driver::CudaModule>>,
    rope: Option<Arc<cudarc::driver::CudaModule>>,
}

#[derive(Clone)]
pub struct Device {
    cuda: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Arc<cudarc::cublas::CudaBlas>,
    /// Cache for loaded PTX modules
    modules: Arc<Mutex<ModuleCache>>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let cuda = cudarc::driver::CudaContext::new(ordinal)?;
        let stream = cuda.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone())?;
        Ok(Self {
            cuda,
            stream,
            blas: Arc::new(blas),
            modules: Arc::new(Mutex::new(Default::default())),
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    fn get_or_load_module(&self, ptx: PTXModule) -> Result<Arc<cudarc::driver::CudaModule>> {
        let mut modules = self.modules.lock().unwrap();
        match ptx {
            PTXModule::Arithmetic => {
                if let Some(ref m) = modules.arithmetic {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::ARITHMETIC.into())?;
                modules.arithmetic = Some(m.clone());
                Ok(m)
            }
            PTXModule::Broadcast => {
                if let Some(ref m) = modules.broadcast {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::BROADCAST.into())?;
                modules.broadcast = Some(m.clone());
                Ok(m)
            }
            PTXModule::Conv => {
                if let Some(ref m) = modules.conv {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::CONV.into())?;
                modules.conv = Some(m.clone());
                Ok(m)
            }
            PTXModule::Fill => {
                if let Some(ref m) = modules.fill {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::FILL.into())?;
                modules.fill = Some(m.clone());
                Ok(m)
            }
            PTXModule::Indexing => {
                if let Some(ref m) = modules.indexing {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::INDEXING.into())?;
                modules.indexing = Some(m.clone());
                Ok(m)
            }
            PTXModule::Layout => {
                if let Some(ref m) = modules.layout {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::LAYOUT.into())?;
                modules.layout = Some(m.clone());
                Ok(m)
            }
            PTXModule::Reduce => {
                if let Some(ref m) = modules.reduce {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::REDUCE.into())?;
                modules.reduce = Some(m.clone());
                Ok(m)
            }
            PTXModule::Rope => {
                if let Some(ref m) = modules.rope {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(crate::cuda_kernels::ROPE.into())?;
                modules.rope = Some(m.clone());
                Ok(m)
            }
        }
    }

    fn get_func(&self, name: &str, mdl: PTXModule) -> Result<CudaFunction> {
        let module = self.get_or_load_module(mdl).map_err(|e| e.context(format!("{mdl:?}")))?;
        let func = module
            .load_function(name)
            .map_err(|e| crate::Error::from(e).context(format!("{mdl:?} {name}")))?;
        Ok(func)
    }

    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        unsafe { self.cuda.disable_event_tracking() }
    }

    pub fn is_event_tracking(&self) -> bool {
        self.cuda.is_event_tracking()
    }
}

/// CUDA storage that holds both the device data and a reference to the device.
pub struct Storage<T: WithDType> {
    pub data: CudaSlice<T>,
    pub device: Device,
}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

fn kernel_name<T: WithDType>(base_name: &str) -> String {
    let dtype_str = match T::DTYPE {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::I64 => "i64",
        DType::U8 => "u8",
    };
    format!("{base_name}_{dtype_str}")
}

/// Implementation of GEMM using cuBLAS for f32.
fn gemm_f32(
    dst: &mut Storage<f32>,
    lhs: (&Storage<f32>, usize),
    rhs: (&Storage<f32>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // Determine transposition and leading dimension for rhs (A in cuBLAS terms)
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    // Determine transposition and leading dimension for lhs (B in cuBLAS terms)
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: rhs_b_stride as i64,
        stride_b: lhs_b_stride as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for f16.
fn gemm_f16(
    dst: &mut Storage<f16>,
    lhs: (&Storage<f16>, usize),
    rhs: (&Storage<f16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: rhs_b_stride as i64,
        stride_b: lhs_b_stride as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for bf16.
fn gemm_bf16(
    dst: &mut Storage<bf16>,
    lhs: (&Storage<bf16>, usize),
    rhs: (&Storage<bf16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: rhs_b_stride as i64,
        stride_b: lhs_b_stride as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
}

// Reduced precision settings
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("CUDA Device {}", self.cuda.ordinal())
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let data = unsafe { dev.stream.alloc::<T>(len) }?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let data = dev.stream.clone_htod(&v)?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        let kname = kernel_name::<T>("fill");
        let func = dst.device.get_func(&kname, PTXModule::Fill)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&elem);
        launch_args.arg(&len);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let src_slice = src.data.slice(..len);
        let mut dst_slice = dst.data.slice_mut(..len);
        dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        Ok(())
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        let data = src.device.stream.clone_dtoh(&src.data.slice(..len))?;
        Ok(std::borrow::Cow::Owned(data))
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("inplace_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("inplace_sin"), None),
            UnaryOp::Sqr => (kernel_name::<T>("inplace_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("inplace_sqrt"), None),
            UnaryOp::Rsqrt => (kernel_name::<T>("inplace_rsqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("inplace_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("inplace_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("inplace_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("inplace_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("inplace_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("inplace_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("inplace_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("assign_add"),
            BinaryOp::Sub => kernel_name::<T>("assign_sub"),
            BinaryOp::Mul => kernel_name::<T>("assign_mul"),
            BinaryOp::Div => kernel_name::<T>("assign_div"),
            BinaryOp::Maximum => kernel_name::<T>("assign_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("assign_minimum"),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&s.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("unary_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("unary_sin"), None),
            UnaryOp::Sqr => (kernel_name::<T>("unary_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("unary_sqrt"), None),
            UnaryOp::Rsqrt => (kernel_name::<T>("unary_rsqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("unary_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("unary_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("unary_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("unary_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("unary_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("unary_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("unary_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("binary_add"),
            BinaryOp::Sub => kernel_name::<T>("binary_sub"),
            BinaryOp::Mul => kernel_name::<T>("binary_mul"),
            BinaryOp::Div => kernel_name::<T>("binary_div"),
            BinaryOp::Maximum => kernel_name::<T>("binary_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("binary_minimum"),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&lhs.data);
        launch_args.arg(&rhs.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        let zero = T::zero();
        let one = T::one();
        if add == zero && scale == one {
            return Self::copy(dst, src, len);
        }
        let kname = kernel_name::<T>("scale_add");
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&scale);
        launch_args.arg(&add);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            // Simple copy when no real transpose needed
            let src_slice = src.data.slice(..numel);
            let mut dst_slice = dst.data.slice_mut(..numel);
            dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_i: usize = dims[..dim1].iter().product();
            let d_j: usize = dims[dim1 + 1..dim2].iter().product();
            let d_k: usize = dims[(dim2 + 1)..].iter().product();
            let d1 = dims[dim1] as u32;
            let d2 = dims[dim2] as u32;
            let d_i = d_i as u32;
            let d_j = d_j as u32;
            let d_k = d_k as u32;

            let kname = kernel_name::<T>("transpose");
            let func = dst.device.get_func(&kname, PTXModule::Layout)?;
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&d1);
            launch_args.arg(&d2);
            launch_args.arg(&d_i);
            launch_args.arg(&d_j);
            launch_args.arg(&d_k);
            launch_args.arg(&src.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn copy2d<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("copy2d");
        let func = dst.device.get_func(&kname, PTXModule::Fill)?;

        let d1 = d1 as u32;
        let d2 = d2 as u32;
        let src_s = src_s as u32;
        let dst_s = dst_s as u32;

        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let src_slice = src.data.slice(src_o..);
        let mut dst_slice = dst.data.slice_mut(dst_o..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_slice);
        launch_args.arg(&mut dst_slice);
        launch_args.arg(&d1);
        launch_args.arg(&d2);
        launch_args.arg(&src_s);
        launch_args.arg(&dst_s);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rope<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope");
        let func = dst.device.get_func(&kname, PTXModule::Rope)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_u32 = d as u32;
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);

        // Slice cos/sin to start at the correct position (like CPU does)
        let cos_offset = pos * d / 2;
        let sin_offset = pos * d / 2;
        let cos_slice = cos.data.slice(cos_offset..);
        let sin_slice = sin.data.slice(sin_offset..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos_slice);
        launch_args.arg(&sin_slice);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        launch_args.arg(&d_u32);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope_i");
        let func = dst.device.get_func(&kname, PTXModule::Rope)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);

        // Slice cos/sin to start at the correct position (like CPU does)
        let cos_offset = pos * d / 2;
        let sin_offset = pos * d / 2;
        let cos_slice = cos.data.slice(cos_offset..);
        let sin_slice = sin.data.slice(sin_offset..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos_slice);
        launch_args.arg(&sin_slice);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: (&Self::Storage<T>, usize),
        rhs: (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        dst_strides: (usize, usize),
        lhs_strides: (usize, usize),
        rhs_strides: (usize, usize),
    ) -> Result<()> {
        // Dispatch to type-specific GEMM implementations
        // We use pointer casting since we know the exact type from DTYPE
        match T::DTYPE {
            DType::F32 => {
                // SAFETY: T::DTYPE == F32 guarantees T is f32
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f32>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f32>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f32>) };
                gemm_f32(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            DType::F16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f16>) };
                gemm_f16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            DType::BF16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<bf16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<bf16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<bf16>) };
                gemm_bf16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            _ => crate::bail!("GEMM not supported for dtype {:?}", T::DTYPE),
        }
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if numel == 0 {
            return Ok(());
        }

        let num_dims = dims.len();
        let info: Vec<usize> = dims.iter().chain(src_strides.iter()).copied().collect();
        let info_dev = dst.device.stream.clone_htod(&info)?;

        let kname = kernel_name::<T>("copy_strided");
        let func = dst.device.get_func(&kname, PTXModule::Layout)?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let num_dims_u32 = num_dims as u32;
        let src_offset_u32 = src_offset as u32;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&numel);
        launch_args.arg(&num_dims_u32);
        launch_args.arg(&info_dev);
        launch_args.arg(&src_offset_u32);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];
        let numel: usize = src_dims.iter().product();

        let kname = kernel_name::<T>("scatter_set");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let numel_i32 = numel as i32;
        let right_size_i32 = right_size as i32;
        let src_dim_size_i32 = src_dim_size as i32;
        let dst_dim_size_i32 = dst_dim_size as i32;

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&src.data);
        launch_args.arg(&ids.data);
        launch_args.arg(&numel_i32);
        launch_args.arg(&right_size_i32);
        launch_args.arg(&src_dim_size_i32);
        launch_args.arg(&dst_dim_size_i32);
        unsafe { launch_args.launch(cfg) }?;

        Ok(())
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = dims[..dim].iter().product();
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);

        // The kernel handles selection along the first dimension.
        // For dim > 0, we loop over left positions.
        let ids_dev = dst.device.stream.clone_htod(ids)?;

        let kname = kernel_name::<T>("is_u32");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        const NUM_THREADS: u32 = 1024;
        let ids_len = ids.len() as u32;
        let right_size_u32 = right_size as u32;
        let threads_x = u32::min(NUM_THREADS, ids_len);
        let threads_y = u32::min(NUM_THREADS / threads_x, right_size_u32).max(1);
        let num_blocks_x = ids_len.div_ceil(threads_x);
        let num_blocks_y = right_size_u32.div_ceil(threads_y);

        let cfg = LaunchConfig {
            block_dim: (threads_x, threads_y, 1),
            grid_dim: (num_blocks_x, num_blocks_y, 1),
            shared_mem_bytes: 0,
        };

        let ids_len_i32 = ids.len() as i32;
        let right_size_i32 = right_size as i32;
        let src_dim_size = dims[dim];

        for left in 0..left_size {
            let src_offset = left * src_dim_size * right_size;
            let dst_offset = left * ids.len() * right_size;
            let src_slice = src.data.slice(src_offset..);
            let mut dst_slice = dst.data.slice_mut(dst_offset..);

            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&ids_len_i32);
            launch_args.arg(&right_size_i32);
            launch_args.arg(&ids_dev);
            launch_args.arg(&src_slice);
            launch_args.arg(&mut dst_slice);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        let total = bh * t1 * t2;
        let kname = kernel_name::<T>("causality_mask");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let bh = bh as u32;
        let t1 = t1 as u32;
        let t2 = t2 as u32;
        let offset = offset as u32;

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&t1);
        launch_args.arg(&t2);
        launch_args.arg(&offset);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as u32;

        let kname = kernel_name::<T>("softmax");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        // Kernel uses: row = blockDim.x*blockIdx.x + threadIdx.x, tid = threadIdx.y
        // One row per block, 32 threads per row for warp-based reduction
        let block_dim = (1, 32, 1);
        let grid_dim = (nrows, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&ncols);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as i32;

        let kname = kernel_name::<T>("rmsnorm");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;
        let block_size = if ncols < 1024 { 32 } else { 1024 };
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&alpha.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size);
        launch_args.arg(&eps);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn layer_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as i32;
        let remove_mean: i32 = if remove_mean { 1 } else { 0 };

        let kname = kernel_name::<T>("layernorm");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;
        let block_size = if ncols < 1024 { 32 } else { 1024 };
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&weight.data);
        launch_args.arg(&bias.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size);
        launch_args.arg(&eps);
        launch_args.arg(&remove_mean);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.stream.clone_htod(&info)?;

        let kname = kernel_name::<T>("fast_max");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.stream.clone_htod(&info)?;

        let kname = kernel_name::<T>("fast_min");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.stream.clone_htod(&info)?;

        let kname = kernel_name::<T>("fast_argmin");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.stream.clone_htod(&info)?;

        let kname = kernel_name::<T>("fast_sum");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn broadcast_binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()> {
        let numel: usize = dst_shape.iter().product();
        if numel == 0 {
            return Ok(());
        }

        let op_name = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Maximum => "max",
            BinaryOp::Minimum => "min",
        };

        let lhs_no_zero = lhs_strides.iter().all(|&s| s > 0);
        let rhs_no_zero = rhs_strides.iter().all(|&s| s > 0);

        let cfg = LaunchConfig::for_num_elems(numel as u32);

        // Dispatch to optimized kernels based on stride patterns
        if lhs_no_zero && rhs_no_zero {
            // Both operands contiguous
            let kname = format!("broadcast_{}_contiguous_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if lhs_no_zero && dst_shape.len() == 2 && rhs_strides == [0, 1] {
            // rhs broadcasts along first dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_rhs_row_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if lhs_no_zero && dst_shape.len() == 2 && rhs_strides == [1, 0] {
            // rhs broadcasts along second dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_rhs_col_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if rhs_no_zero && dst_shape.len() == 2 && lhs_strides == [0, 1] {
            // lhs broadcasts along first dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_lhs_row_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if rhs_no_zero && dst_shape.len() == 2 && lhs_strides == [1, 0] {
            // lhs broadcasts along second dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_lhs_col_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else {
            // General strided case
            let num_dims = dst_shape.len();
            let info: Vec<usize> = dst_shape
                .iter()
                .chain(lhs_strides.iter())
                .chain(rhs_strides.iter())
                .copied()
                .collect();
            let info_dev = dst.device.stream.clone_htod(&info)?;

            let kname = format!("broadcast_{}_strided_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let num_dims_u32 = num_dims as u32;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&num_dims_u32);
            launch_args.arg(&info_dev);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn conv1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        if groups == 1 {
            // IM2COL approach: transform conv1d into matrix multiplication
            // 1. Im2Col: transform input [B, C, L] -> [B, L_out, C * K]
            // 2. Matmul: [B, L_out, C*K] x [C*K, out_channels] -> [B, L_out, out_channels]
            // 3. Transpose result to [B, out_channels, L_out]
            conv1d_im2col(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
            )
        } else {
            // Direct kernel for grouped convolutions
            conv1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            )
        }
    }

    fn conv_transpose1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        // COL2IM approach can be used when:
        // - groups == 1
        // - padding == 0
        // - output_padding == 0
        let can_use_col2im = groups == 1 && padding == 0 && output_padding == 0;

        if can_use_col2im {
            // COL2IM approach: matmul + col2im transformation
            conv_transpose1d_col2im(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
            )
        } else {
            // Direct kernel for grouped convolutions or with padding
            conv_transpose1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
            )
        }
    }
}

// ============================================================================
// Conv1d implementation using im2col + cuBLAS gemm
// ============================================================================

fn conv1d_im2col<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<()> {
    let k = in_channels * kernel_size;
    let col_numel = batch * out_length * k;

    // Step 1: Im2Col transformation
    // Allocate temporary buffer for col data [B, L_out, C_in * K]
    let mut col: CudaSlice<T> = unsafe { dst.device.stream.alloc(col_numel) }?;

    let kname = format!("im2col1d_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(col_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&col_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&dilation);
    launch_args.arg(&src.data);
    launch_args.arg(&mut col);
    unsafe { launch_args.launch(cfg) }?;

    // Step 2: Matrix multiplication using cuBLAS
    // col: [B, L_out, K] where K = in_channels * kernel_size
    // kernel: [out_channels, K] (stored as [out_channels, in_channels, kernel_size])
    // result: [B, L_out, out_channels]
    //
    // We need to allocate a temporary buffer for the result since it's in a different layout
    let result_numel = batch * out_length * out_channels;
    let mut result: CudaSlice<T> = unsafe { dst.device.stream.alloc(result_numel) }?;

    // For each batch, perform: result[b] = col[b] @ kernel^T
    // col[b]: [L_out, K], kernel: [out_channels, K], result[b]: [L_out, out_channels]
    // Using cuBLAS: C = alpha * A * B + beta * C
    // We want: result = col @ kernel^T
    // cuBLAS uses column-major, so we compute: result^T = kernel @ col^T
    // Which gives us result in row-major as [L_out, out_channels]
    conv1d_gemm(&dst.device, &col, &kernel.data, &mut result, batch, out_length, out_channels, k)?;

    // Step 3: Transpose from [B, L_out, out_channels] to [B, out_channels, L_out]
    let kname = format!("transpose_blc_bcl_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(result_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&result_numel);
    launch_args.arg(&batch);
    launch_args.arg(&out_length);
    launch_args.arg(&out_channels);
    launch_args.arg(&result);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

/// Batched GEMM for conv1d: result[b] = col[b] @ kernel^T
/// col: [B, M, K], kernel: [N, K], result: [B, M, N]
fn conv1d_gemm<T: WithDTypeF>(
    device: &Device,
    col: &CudaSlice<T>,
    kernel: &CudaSlice<T>,
    result: &mut CudaSlice<T>,
    batch: usize,
    m: usize, // out_length
    n: usize, // out_channels
    k: usize, // in_channels * kernel_size
) -> Result<()> {
    match T::DTYPE {
        DType::F32 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f32>) };
            conv1d_gemm_f32(device, col, kernel, result, batch, m, n, k)
        }
        DType::F16 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f16>) };
            conv1d_gemm_f16(device, col, kernel, result, batch, m, n, k)
        }
        DType::BF16 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<bf16>) };
            conv1d_gemm_bf16(device, col, kernel, result, batch, m, n, k)
        }
        _ => crate::bail!("conv1d GEMM not supported for dtype {:?}", T::DTYPE),
    }
}

fn conv1d_gemm_f32(
    device: &Device,
    col: &CudaSlice<f32>,
    kernel: &CudaSlice<f32>,
    result: &mut CudaSlice<f32>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // We compute: result = col @ kernel^T (row-major)
    // col[b]: [M, K] row-major, kernel: [N, K] row-major
    // result[b]: [M, N] row-major
    //
    // cuBLAS is column-major. Row-major [R, C] with stride C is seen as column-major [C, R]:
    // - kernel stored [N, K] row-major -> cuBLAS sees [K, N], lda = K
    // - col stored [M, K] row-major -> cuBLAS sees [K, M], ldb = K
    // - result stored [M, N] row-major -> cuBLAS sees [N, M], ldc = N
    //
    // We want: result = col @ kernel^T (row-major)
    // Taking transpose: result^T = kernel @ col^T
    // In cuBLAS terms for C = op(A) * op(B) where C is [N, M]:
    // - A = kernel, seen as [K, N], with transa=T gives [N, K]
    // - B = col, seen as [K, M], with transb=N gives [K, M]
    // - Result: [N, K] @ [K, M] = [N, M] ✓

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,   // rows of C (N = out_channels)
        n: m as i32,   // cols of C (M = out_length)
        k: k as i32,   // inner dimension
        lda: k as i32, // leading dim of A (row stride of kernel)
        ldb: k as i32, // leading dim of B (row stride of col)
        ldc: n as i32, // leading dim of C (row stride of result)
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,              // kernel is shared across batches
        stride_b: (m * k) as i64, // col stride per batch
        stride_c: (m * n) as i64, // result stride per batch
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, col, result)?;
    }

    Ok(())
}

fn conv1d_gemm_f16(
    device: &Device,
    col: &CudaSlice<f16>,
    kernel: &CudaSlice<f16>,
    result: &mut CudaSlice<f16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: k as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, col, result)?;
    }

    Ok(())
}

fn conv1d_gemm_bf16(
    device: &Device,
    col: &CudaSlice<bf16>,
    kernel: &CudaSlice<bf16>,
    result: &mut CudaSlice<bf16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: k as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, col, result)?;
    }

    Ok(())
}

// ============================================================================
// Conv1d direct implementation (fallback for grouped convolutions)
// ============================================================================

fn conv1d_direct<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let dst_numel = batch * out_channels * out_length;

    let kname = format!("conv1d_direct_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(dst_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&dst_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&dilation);
    launch_args.arg(&groups);
    launch_args.arg(&src.data);
    launch_args.arg(&kernel.data);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

// ============================================================================
// Conv transpose 1d implementation using col2im
// ============================================================================

fn conv_transpose1d_col2im<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,     // input length
    out_length: usize, // output length
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    // COL2IM approach:
    // 1. Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
    // 2. Matmul: [B, L_in, C_in] @ [C_in, C_out * K] -> [B, L_in, C_out * K]
    // 3. Col2Im: [B, L_in, C_out, K] -> [B, C_out, L_out]

    let src_numel = batch * in_channels * length;
    let n = out_channels * kernel_size;
    let col_numel = batch * length * n;
    let dst_numel = batch * out_channels * out_length;

    // Step 1: Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
    let mut src_transposed: CudaSlice<T> = unsafe { dst.device.stream.alloc(src_numel) }?;

    let kname = format!("transpose_bcl_blc_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(src_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&src_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&src.data);
    launch_args.arg(&mut src_transposed);
    unsafe { launch_args.launch(cfg) }?;

    // Step 2: Matrix multiplication
    // src_transposed: [B, L_in, C_in]
    // kernel: [C_in, C_out, K] stored row-major, treat as [C_in, C_out * K]
    // result (col): [B, L_in, C_out * K]
    let mut col: CudaSlice<T> = unsafe { dst.device.stream.alloc(col_numel) }?;

    conv_transpose1d_gemm(
        &dst.device,
        &src_transposed,
        &kernel.data,
        &mut col,
        batch,
        length,
        n,
        in_channels,
    )?;

    // Step 3: Col2Im transformation
    // col: [B, L_in, C_out * K] = [B, L_in, C_out, K]
    // output: [B, C_out, L_out]
    let kname = format!("col2im1d_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(dst_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&dst_numel);
    launch_args.arg(&batch);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&col);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

/// Batched GEMM for conv_transpose1d: result[b] = src[b] @ kernel
/// src: [B, M, K], kernel: [K, N], result: [B, M, N]
fn conv_transpose1d_gemm<T: WithDTypeF>(
    device: &Device,
    src: &CudaSlice<T>,
    kernel: &CudaSlice<T>,
    result: &mut CudaSlice<T>,
    batch: usize,
    m: usize, // length (L_in)
    n: usize, // out_channels * kernel_size
    k: usize, // in_channels
) -> Result<()> {
    match T::DTYPE {
        DType::F32 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f32>) };
            conv_transpose1d_gemm_f32(device, src, kernel, result, batch, m, n, k)
        }
        DType::F16 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f16>) };
            conv_transpose1d_gemm_f16(device, src, kernel, result, batch, m, n, k)
        }
        DType::BF16 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<bf16>) };
            conv_transpose1d_gemm_bf16(device, src, kernel, result, batch, m, n, k)
        }
        _ => crate::bail!("conv_transpose1d GEMM not supported for dtype {:?}", T::DTYPE),
    }
}

fn conv_transpose1d_gemm_f32(
    device: &Device,
    src: &CudaSlice<f32>,
    kernel: &CudaSlice<f32>,
    result: &mut CudaSlice<f32>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // We compute: result = src @ kernel (row-major)
    // src[b]: [M, K] row-major, kernel: [K, N] row-major
    // result[b]: [M, N] row-major
    //
    // cuBLAS is column-major. Row-major data appears transposed:
    // - src stored [M, K] -> cuBLAS sees [K, M]
    // - kernel stored [K, N] -> cuBLAS sees [N, K]
    // - result stored [M, N] -> cuBLAS sees [N, M]
    //
    // For row-major C = A @ B: C^T = B^T @ A^T in column-major
    // So: result^T[N, M] = kernel^T[N, K] @ src^T[K, M]
    //
    // cuBLAS sees kernel as [N, K] and src as [K, M], no transpose needed
    // Row-major [R, C] has row stride C, cuBLAS sees as col-major [C, R] with ld = C
    // - kernel [K, N] row-major -> cuBLAS [N, K], lda = N
    // - src [M, K] row-major -> cuBLAS [K, M], ldb = K
    // transa = N, transb = N

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,   // rows of C = N
        n: m as i32,   // cols of C = M
        k: k as i32,   // inner dimension = K
        lda: n as i32, // leading dim of kernel (row stride N)
        ldb: k as i32, // leading dim of src (row stride K)
        ldc: n as i32, // leading dim of result (row stride N)
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,              // kernel is shared across batches
        stride_b: (m * k) as i64, // src stride per batch
        stride_c: (m * n) as i64, // result stride per batch
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, src, result)?;
    }

    Ok(())
}

fn conv_transpose1d_gemm_f16(
    device: &Device,
    src: &CudaSlice<f16>,
    kernel: &CudaSlice<f16>,
    result: &mut CudaSlice<f16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, src, result)?;
    }

    Ok(())
}

fn conv_transpose1d_gemm_bf16(
    device: &Device,
    src: &CudaSlice<bf16>,
    kernel: &CudaSlice<bf16>,
    result: &mut CudaSlice<bf16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // Row-major [R, C] has row stride C, cuBLAS sees as col-major [C, R] with ld = C
    // - kernel [K, N] row-major -> cuBLAS [N, K], lda = N
    // - src [M, K] row-major -> cuBLAS [K, M], ldb = K
    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };

    let cfg = StridedBatchedConfig {
        batch_size: batch as i32,
        gemm,
        stride_a: 0,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        device.blas.gemm_strided_batched(cfg, kernel, src, result)?;
    }

    Ok(())
}

// ============================================================================
// Conv transpose 1d direct implementation (fallback)
// ============================================================================

fn conv_transpose1d_direct<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Result<()> {
    let dst_numel = batch * out_channels * out_length;

    let kname = format!("conv_transpose1d_direct_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(dst_numel as u32);

    // Note: dilation is fixed to 1 for now (matching CPU backend behavior)
    let dilation: usize = 1;

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&dst_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&output_padding);
    launch_args.arg(&dilation);
    launch_args.arg(&groups);
    launch_args.arg(&src.data);
    launch_args.arg(&kernel.data);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}
