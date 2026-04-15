use cudarc::cublaslt::result::CublasError;
use cudarc::cublaslt::sys as lt_sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError};
use std::ffi::c_void;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Workspace {
    buffer: CudaSlice<u8>,
    size: usize,
}

impl Workspace {
    /// Creates a CublasLt workspace buffer on the provided device
    fn new(stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        stream.context().bind_to_thread()?;

        let major = stream.context().attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };

        let buffer = unsafe { stream.alloc::<u8>(workspace_size)? };
        Ok(Self { buffer, size: workspace_size })
    }
}

#[derive(Debug)]
pub(crate) struct CudaBlasLT {
    handle: cudarc::cublaslt::sys::cublasLtHandle_t,
    workspace: Workspace,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CudaBlasLT {}
unsafe impl Sync for CudaBlasLT {}

impl CudaBlasLT {
    /// Creates a new cublasLt handle.
    pub fn new(stream: Arc<CudaStream>) -> crate::Result<Self> {
        let handle = cudarc::cublaslt::result::create_handle()?;
        let workspace = Workspace::new(stream.clone())?;
        Ok(Self { handle, workspace, stream })
    }
}

impl Drop for CudaBlasLT {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { cudarc::cublaslt::result::destroy_handle(self.handle).ok() };
        }
    }
}

/// MatrixLayout helper type
struct MatrixLayout {
    handle: cudarc::cublaslt::sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn new(
        matrix_type: cudarc::cublaslt::sys::cudaDataType,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> Result<Self, CublasError> {
        let handle = cudarc::cublaslt::result::create_matrix_layout(matrix_type, rows, cols, ld)?;
        Ok(Self { handle })
    }

    fn set_batch(&self, size: core::ffi::c_int, stride: i64) -> Result<(), CublasError> {
        unsafe {
            // Set batch size
            cudarc::cublaslt::result::set_matrix_layout_attribute(
                self.handle,
                cudarc::cublaslt::sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&size) as *const _ as *const _,
                std::mem::size_of::<core::ffi::c_int>(),
            )?;
            // Set batch stride
            cudarc::cublaslt::result::set_matrix_layout_attribute(
                self.handle,
                cudarc::cublaslt::sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride) as *const _ as *const _,
                std::mem::size_of::<i64>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // panic on failure
        unsafe {
            cudarc::cublaslt::result::destroy_matrix_layout(self.handle).ok();
        }
    }
}

#[allow(dead_code)]
enum Matrix {
    A,
    B,
    C,
}

/// MatmulDesc helper type
struct MatmulDesc {
    handle: cudarc::cublaslt::sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn new(
        compute_type: cudarc::cublaslt::sys::cublasComputeType_t,
        scale_type: cudarc::cublaslt::sys::cudaDataType,
    ) -> Result<Self, CublasError> {
        let handle = cudarc::cublaslt::result::create_matmul_desc(compute_type, scale_type)?;
        Ok(Self { handle })
    }

    fn set_transpose(&self, transpose: bool, matrix: Matrix) -> Result<(), CublasError> {
        // Set transpose
        // 1 == T, 0 == N
        let transpose = transpose as i32;
        let attr = match matrix {
            Matrix::A => {
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA
            }
            Matrix::B => {
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB
            }
            Matrix::C => {
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC
            }
        };

        unsafe {
            cudarc::cublaslt::result::set_matmul_desc_attribute(
                self.handle,
                attr,
                (&transpose) as *const _ as *const _,
                std::mem::size_of::<u32>(),
            )?;
        }
        Ok(())
    }

    fn set_attr<T>(
        &self,
        attr: lt_sys::cublasLtMatmulDescAttributes_t,
        val: &T,
    ) -> Result<(), CublasError> {
        unsafe {
            cudarc::cublaslt::result::set_matmul_desc_attribute(
                self.handle,
                attr,
                val as *const T as *const _,
                std::mem::size_of::<T>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe {
            cudarc::cublaslt::result::destroy_matmul_desc(self.handle).ok();
        }
    }
}

/// MatmulPref helper type
struct MatmulPref {
    handle: cudarc::cublaslt::sys::cublasLtMatmulPreference_t,
}

impl MatmulPref {
    fn new() -> Result<Self, CublasError> {
        let handle = cudarc::cublaslt::result::create_matmul_pref()?;
        Ok(Self { handle })
    }

    fn set_workspace_size(&self, size: usize) -> Result<(), CublasError> {
        unsafe {
            // Set workspace size
            cudarc::cublaslt::result::set_matmul_pref_attribute(
                self.handle,
                cudarc::cublaslt::sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&size) as *const _ as *const _,
                std::mem::size_of::<usize>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulPref {
    fn drop(&mut self) {
        unsafe {
            cudarc::cublaslt::result::destroy_matmul_pref(self.handle).ok();
        }
    }
}

/// Configuration for batched FP8 matmul.
pub(crate) struct BatchConfig {
    /// Number of matrices in the batch.
    pub count: i32,
    /// Stride between consecutive A matrices (in elements). 0 to broadcast.
    pub stride_a: i64,
    /// Stride between consecutive B matrices (in elements). 0 to broadcast.
    pub stride_b: i64,
    /// Stride between consecutive output matrices (in elements).
    pub stride_out: i64,
}

impl CudaBlasLT {
    /// FP8 matrix multiplication: computes `C = A^T × B` where A and B are FP8 E4M3.
    ///
    /// This implements the TN layout required by cuBLASLt for FP8:
    /// - `a_data`: rhs FP8 data, row-major `[N, K]` (= col-major `[K, N]`)
    /// - `b_data`: lhs FP8 data, row-major `[M, K]` (= col-major `[K, M]`)
    /// - `a_scale`, `b_scale`: f32 scales on device — either scalars (per-tensor)
    ///   or vectors (outer-vector mode).
    /// - `use_outer_vec`: when true, use `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`
    ///   for both A and B scale modes. In this mode `a_scale` must have N elements
    ///   and `b_scale` must have M elements. Requires CUDA 12.9+ on Hopper/Ada.
    /// - `out`: output bf16 buffer, row-major `[M, N]` (= col-major `[N, M]`)
    /// - `m`, `n`, `k`: matrix dimensions
    /// - `batch`: optional batch configuration
    pub fn matmul_f8(
        &self,
        a_data: &CudaSlice<u8>,
        b_data: &CudaSlice<u8>,
        a_scale: &CudaSlice<f32>,
        b_scale: &CudaSlice<f32>,
        use_outer_vec: bool,
        out: &mut CudaSlice<half::bf16>,
        m: usize,
        n: usize,
        k: usize,
        batch: Option<&BatchConfig>,
    ) -> crate::Result<()> {
        let stream = &self.stream;
        let fp8_type = lt_sys::cudaDataType_t::CUDA_R_8F_E4M3;
        let bf16_type = lt_sys::cudaDataType_t::CUDA_R_16BF;

        // Scale mode value (CUDA 12.9+):
        //   CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F = 3
        const OUTER_VEC_32F: i32 = 3;

        // Matmul descriptor: compute in f32, scale type f32.
        let matmul_desc = MatmulDesc::new(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt_sys::cudaDataType_t::CUDA_R_32F,
        )?;
        matmul_desc.set_transpose(true, Matrix::A)?;

        // Set scale pointers.
        let (a_sc_ptr, _ga) = a_scale.device_ptr(stream);
        let (b_sc_ptr, _gb) = b_scale.device_ptr(stream);
        let a_sc_p = a_sc_ptr as *const c_void;
        let b_sc_p = b_sc_ptr as *const c_void;
        matmul_desc.set_attr(
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_sc_p,
        )?;
        matmul_desc.set_attr(
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_sc_p,
        )?;

        // OUTER_VEC_32F requires BOTH A and B scale modes to be set together.
        if use_outer_vec {
            matmul_desc.set_attr(
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &OUTER_VEC_32F,
            )?;
            matmul_desc.set_attr(
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &OUTER_VEC_32F,
            )?;
        }

        // Matrix layouts.
        // A = rhs: stored as [K, N] col-major, ld = K
        let a_layout = MatrixLayout::new(fp8_type, k as u64, n as u64, k as i64)?;
        // B = lhs: stored as [K, M] col-major, ld = K
        let b_layout = MatrixLayout::new(fp8_type, k as u64, m as u64, k as i64)?;
        // C and D: [N, M] col-major, ld = N
        let c_layout = MatrixLayout::new(bf16_type, n as u64, m as u64, n as i64)?;

        if let Some(batch) = batch {
            a_layout.set_batch(batch.count, batch.stride_a)?;
            b_layout.set_batch(batch.count, batch.stride_b)?;
            c_layout.set_batch(batch.count, batch.stride_out)?;
        }

        // Algorithm selection.
        let matmul_pref = MatmulPref::new()?;
        matmul_pref.set_workspace_size(self.workspace.size)?;

        let heuristic = unsafe {
            cudarc::cublaslt::result::get_matmul_algo_heuristic(
                self.handle,
                matmul_desc.handle,
                a_layout.handle,
                b_layout.handle,
                c_layout.handle,
                c_layout.handle,
                matmul_pref.handle,
            )?
        };

        // Launch matmul.
        {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let (a_ptr, _ra) = a_data.device_ptr(stream);
            let (b_ptr, _rb) = b_data.device_ptr(stream);
            let (d_ptr, _rd) = out.device_ptr_mut(stream);
            let (w_ptr, _rw) = self.workspace.buffer.device_ptr(stream);

            unsafe {
                cudarc::cublaslt::result::matmul(
                    self.handle,
                    matmul_desc.handle,
                    &alpha as *const f32 as *const c_void,
                    &beta as *const f32 as *const c_void,
                    a_ptr as *const c_void,
                    a_layout.handle,
                    b_ptr as *const c_void,
                    b_layout.handle,
                    d_ptr as *const c_void, // C input (unused, beta=0)
                    c_layout.handle,
                    d_ptr as *mut c_void, // D output
                    c_layout.handle,
                    &heuristic.algo as *const _,
                    w_ptr as *mut c_void,
                    self.workspace.size,
                    stream.cu_stream() as *mut _,
                )?;
            }
        }

        Ok(())
    }
}
