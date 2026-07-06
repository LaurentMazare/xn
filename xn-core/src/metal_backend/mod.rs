//! Metal compute backend.
//!
//! This backend targets Apple GPUs, where the GPU shares system memory with
//! the CPU (unified memory). It allocates all tensor storage as
//! `StorageModeShared` `MTLBuffer`s, so uploads, readbacks and fills are plain
//! `memcpy`s through the buffer's `contents()` pointer with no staging
//! buffers.
//!
//! Compute kernels are MSL compute shaders (see `metal-kernels/`) embedded in
//! the crate and compiled at device creation into three libraries: an `f32`
//! variant plus `USE_F16`/`USE_BF16` variants that change the buffer element
//! type while keeping f32 compute. Data-movement ops (copy, fill, dtype
//! conversion, and the layout/indexing ops for non-float element types) run on
//! the host over the shared memory.
//!
//! Synchronization model: dispatches are recorded into a single serial compute
//! command encoder and only submitted when the batch is flushed (on host
//! readback / `synchronize` / before host access to shared memory). The serial
//! encoder guarantees op ordering on the GPU; the flush waits for completion,
//! so host accesses to shared memory are always consistent.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use metal::objc::rc::autoreleasepool;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Shared dtype prelude, prepended to the concatenated kernel sources.
const DTYPE_SRC: &str = include_str!("../../metal-kernels/dtype.metal");

/// MLX steel GEMM kernels (simdgroup-matrix based), taken from
/// candle-metal-kernels which extracted them from MLX. Compiled as its own
/// library: the file is self-contained and instantiates `gemm_{nn,nt,tn,tt}_*`
/// specializations selected via function constants.
const MLX_GEMM_SRC: &str = include_str!("../../metal-kernels/mlx_gemm.metal");

/// MLX GEMV kernels (`gemv` for a row-major weight matrix / matmul_t, `gemv_t`
/// for the transposed layout), adapted from MLX with f32 accumulation.
const MLX_GEMV_SRC: &str = include_str!("../../metal-kernels/mlx_gemv.metal");

/// All kernel bodies; concatenated (after the prelude) into one MSL library
/// per dtype variant.
const KERNEL_SRCS: &[&str] = &[
    include_str!("../../metal-kernels/unary.metal"),
    include_str!("../../metal-kernels/binary.metal"),
    include_str!("../../metal-kernels/scale_add.metal"),
    include_str!("../../metal-kernels/broadcast.metal"),
    include_str!("../../metal-kernels/softmax.metal"),
    include_str!("../../metal-kernels/rmsnorm.metal"),
    include_str!("../../metal-kernels/layernorm.metal"),
    include_str!("../../metal-kernels/reduce.metal"),
    include_str!("../../metal-kernels/reduce_arg.metal"),
    include_str!("../../metal-kernels/rope.metal"),
    include_str!("../../metal-kernels/rope_i.metal"),
    include_str!("../../metal-kernels/transpose.metal"),
    include_str!("../../metal-kernels/copy2d.metal"),
    include_str!("../../metal-kernels/copy_strided.metal"),
    include_str!("../../metal-kernels/index_select.metal"),
    include_str!("../../metal-kernels/causality_mask.metal"),
    include_str!("../../metal-kernels/scatter_set.metal"),
    include_str!("../../metal-kernels/gemm_tiled.metal"),
    include_str!("../../metal-kernels/gemv.metal"),
    include_str!("../../metal-kernels/conv1d.metal"),
    include_str!("../../metal-kernels/conv_transpose1d.metal"),
    include_str!("../../metal-kernels/im2col1d.metal"),
    include_str!("../../metal-kernels/col2im1d.metal"),
    include_str!("../../metal-kernels/cast.metal"),
];

/// The dtype-generic kernels (integer arithmetic or pure data movement); only
/// these are compiled into the i64/u8 library variants, so that non-float
/// tensors stay on the GPU instead of falling back to host loops that drain
/// the pipeline.
const KERNEL_SRCS_INT: &[&str] = &[
    include_str!("../../metal-kernels/binary.metal"),
    include_str!("../../metal-kernels/broadcast.metal"),
    include_str!("../../metal-kernels/transpose.metal"),
    include_str!("../../metal-kernels/copy2d.metal"),
    include_str!("../../metal-kernels/copy_strided.metal"),
    include_str!("../../metal-kernels/index_select.metal"),
    include_str!("../../metal-kernels/scatter_set.metal"),
];

/// Definition of a compute kernel given a dtype-suffixed name such as
/// `"unary_f16"`: returns its library index (0 = f32, 1 = f16, 2 = bf16), the
/// MSL entry point, its number of storage-buffer bindings (bound at buffer
/// indices `0..bindings`, push constants at index `bindings`), and its
/// threads-per-threadgroup. Unsupported (kernel, dtype) combinations return
/// `None` so that a wrong dispatch fails loudly instead of silently running
/// the wrong variant.
type KernelDef<'a> = (usize, &'a str, u32, (u64, u64, u64));
fn kernel_def(name: &str) -> Option<KernelDef<'_>> {
    let wg1d = (WORKGROUP_SIZE as u64, 1, 1);
    // Dtype casts name both types explicitly and live in the f32 library.
    if name.starts_with("cast_") {
        return Some((0, name, 2, wg1d));
    }
    let (base, dt) = name.rsplit_once('_')?;
    // Which dtype variants each kernel exists for.
    #[derive(PartialEq)]
    enum Dt {
        F32Only,
        Float,
        Any,
    }
    // (bindings, threads-per-threadgroup, dtype coverage)
    let (bindings, wg, dts) = match base {
        "unary" => (2, wg1d, Dt::Float),
        "binary" => (3, wg1d, Dt::Any),
        "scale_add" => (2, wg1d, Dt::Float),
        "broadcast" => (4, wg1d, Dt::Any),
        "softmax" => (2, wg1d, Dt::Float),
        "rmsnorm" => (3, wg1d, Dt::Float),
        "layernorm" => (4, wg1d, Dt::Float),
        "rope" => (4, wg1d, Dt::Float),
        "rope_i" => (4, wg1d, Dt::Float),
        "reduce" => (2, wg1d, Dt::Float),
        "reduce_arg" => (2, wg1d, Dt::Float),
        "transpose" => (2, wg1d, Dt::Any),
        "copy2d" => (2, wg1d, Dt::Any),
        "copy_strided" => (3, wg1d, Dt::Any),
        "index_select" => (3, wg1d, Dt::Any),
        "causality_mask" => (1, wg1d, Dt::Float),
        "scatter_set" => (3, wg1d, Dt::Any),
        "gemm_tiled" => (3, (16, 16, 1), Dt::Float),
        "gemv" => (3, wg1d, Dt::Float),
        // conv kernels are f32-only; other dtypes must fail pipeline lookup.
        "conv1d" => (3, wg1d, Dt::F32Only),
        "conv_transpose1d" => (3, wg1d, Dt::F32Only),
        "im2col1d" => (2, wg1d, Dt::F32Only),
        "col2im1d" => (2, wg1d, Dt::F32Only),
        _ => return None,
    };
    let lib_idx = match dt {
        "f32" => 0,
        "f16" if dts != Dt::F32Only => 1,
        "bf16" if dts != Dt::F32Only => 2,
        "i64" if dts == Dt::Any => 3,
        "u8" if dts == Dt::Any => 4,
        _ => return None,
    };
    Some((lib_idx, base, bindings, wg))
}

const WORKGROUP_SIZE: u32 = 256;

fn mtlerr<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> crate::Error + '_ {
    move |e| crate::Error::msg(format!("metal: {context}: {e:?}"))
}

/// GPUStartTime/GPUEndTime of a completed command buffer, in seconds. Not
/// exposed by metal-rs, so the Objective-C selectors are sent directly
/// (registered at runtime rather than via `msg_send!`, whose expansion trips
/// the `unexpected_cfgs` lint).
fn command_buffer_gpu_times(cmd: &metal::CommandBufferRef) -> (f64, f64) {
    use metal::objc::Message;
    use metal::objc::runtime::Sel;
    let send = |name: &str| -> f64 {
        unsafe { cmd.send_message(Sel::register(name), ()) }.unwrap_or_default()
    };
    (send("GPUStartTime"), send("GPUEndTime"))
}

/// Little-endian push-constant byte builder. The MSL kernels declare their
/// push constants as a struct of `uint`/`float` fields, which have identical
/// sequential layout.
#[derive(Default)]
struct Pc {
    bytes: Vec<u8>,
}

impl Pc {
    fn new() -> Self {
        Self { bytes: Vec::with_capacity(64) }
    }
    fn u32(mut self, v: u32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn f32(mut self, v: f32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn usize(self, v: usize) -> Self {
        self.u32(v as u32)
    }
}

struct CachedPipeline {
    pipeline: metal::ComputePipelineState,
    bindings: u32,
    wg: (u64, u64, u64),
}

/// Parameter block for the MLX steel GEMM kernels; layout must match the
/// `GEMMParams` struct in `mlx_gemm.metal`.
#[repr(C)]
struct MlxGemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

/// MLX GEMM tile configuration: block size (bm, bn, bk) = (32, 32, 16) and
/// (wm, wn) = (2, 2) simdgroups per threadgroup, matching the instantiations
/// in `mlx_gemm.metal`.
const MLX_BM: usize = 32;
const MLX_BN: usize = 32;
const MLX_BK: usize = 16;

/// Output columns per gemv threadgroup (one simdgroup each); must match
/// `GEMV_NSG` in `gemv.metal` and the kernel's 32 * GEMV_NSG threadgroup size
/// in `kernel_def`.
const GEMV_NSG: u32 = 8;

/// Commands recorded per command buffer before it is committed (without
/// waiting): the GPU starts executing early batches while the host keeps
/// recording later ones, instead of sitting idle until the flush.
const SUBMIT_CHUNK: u32 = 256;

/// Max committed-but-unretired command buffers before recording applies
/// backpressure by waiting for the oldest one. Bounds how much memory the
/// deferred buffer recycling can hold back.
const MAX_IN_FLIGHT: usize = 16;

/// Command-recording state, guarded by a mutex.
///
/// Dispatches are recorded into a serial compute encoder on `cmd_buffer`.
/// Every `SUBMIT_CHUNK` commands the buffer is committed *without* waiting
/// (see [`Device::maybe_submit`]) so GPU execution overlaps host recording; a
/// flush (host readback / `synchronize` / host access to shared memory)
/// submits the tail and waits for everything in flight.
struct OpCtx {
    /// Open command buffer holding recorded, unsubmitted commands.
    cmd_buffer: Option<metal::CommandBuffer>,
    /// Open serial compute encoder on `cmd_buffer` (closed around blits).
    encoder: Option<metal::ComputeCommandEncoder>,
    /// Number of commands recorded in the current batch.
    n_dispatches: u32,
    /// Buffers (dropped tensors + scratch) to recycle into the pool once the
    /// current batch has finished executing on the GPU.
    free_bufs: Vec<PooledBuf>,
    /// Profiling: kernel name per recorded command in the current batch.
    prof_names: Vec<String>,
    /// Committed batches, oldest first (the queue executes them in commit
    /// order). Their pooled buffers are recycled when they retire.
    in_flight: std::collections::VecDeque<InFlight>,
}

/// A committed, possibly still executing command buffer.
struct InFlight {
    cmd: metal::CommandBuffer,
    /// Buffers to recycle once this batch has completed.
    free_bufs: Vec<PooledBuf>,
    /// Profiling: kernel names recorded in this batch (one entry per batch in
    /// per-kernel mode, unused otherwise).
    names: Vec<String>,
}

/// Accumulated profiling counters.
/// `XN_METAL_PROFILE=1`: per-kernel mode — every command is flushed in its
/// own command buffer, so the buffer's GPUStartTime/GPUEndTime delta is an
/// accurate per-command GPU duration (per-dispatch timestamps inside a batch
/// are not available on Apple GPUs). Batching is disabled, so wall-clock
/// numbers are pessimistic; the per-kernel GPU times and shares are the
/// useful output.
/// `XN_METAL_PROFILE=2`: batch mode — batching/pipelining stays exactly as in
/// normal runs and only per-command-buffer totals are collected: GPU busy
/// time, GPU idle gaps between consecutive buffers (host recording, kernel
/// scheduling latency, host compute), and CPU wait time. This is the mode to
/// diagnose wall-clock vs GPU-time discrepancies.
#[derive(Default)]
struct ProfStats {
    /// kernel name -> (dispatch count, total gpu ns)
    per_kernel: HashMap<String, (u64, u128)>,
    gpu_ns: u128,
    dispatches: u64,
    /// Committed command buffers.
    submissions: u64,
    /// Blocking waits (host readbacks / synchronize / backpressure).
    flushes: u64,
    /// CPU time spent blocked waiting on the GPU.
    wait_ns: u128,
    /// Blocking waits broken down by cause -> (count, wait ns).
    wait_reasons: HashMap<&'static str, (u64, u128)>,
    /// GPU idle time between consecutive command buffers.
    gap_ns: u128,
    /// GPUEndTime of the last retired command buffer.
    last_gpu_end: f64,
}

/// A buffer plus its shared-memory pointer, as kept in the recycling pool. The
/// pointer is stored as `usize` so the struct is trivially `Send`/`Sync`
/// behind the pool mutex.
struct PooledBuf {
    buffer: metal::Buffer,
    ptr: usize,
    class: u64,
}

/// Recycling pool for buffer allocations, keyed by size class. Buffer
/// allocation is not free and decoding allocates hundreds of intermediate
/// tensors per token, so freed buffers are returned here (after the batch
/// referencing them completes) and reused instead of being destroyed.
#[derive(Default)]
struct BufferPool {
    free: HashMap<u64, Vec<PooledBuf>>,
    hits: u64,
    misses: u64,
}

/// Round a byte size up to its allocation class: the next power of two below
/// 1 MiB (256 B minimum), a 1/16 subdivision of the enclosing power of two
/// above it (max ~12.5% waste). Buffers are created with the class size so any
/// same-class request can reuse them.
fn size_class(bytes: usize) -> u64 {
    let bytes = bytes.max(4) as u64;
    let np2 = bytes.next_power_of_two();
    if np2 <= (1 << 20) { np2.max(256) } else { bytes.div_ceil(np2 / 16) * (np2 / 16) }
}

pub struct DeviceInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    /// Per-dtype-variant libraries: [f32, f16, bf16, i64, u8] (the integer
    /// variants only hold the dtype-generic kernels).
    libraries: [metal::Library; 5],
    /// MLX steel GEMM library (simdgroup-matrix kernels, own source file).
    mlx_gemm_library: metal::Library,
    /// MLX GEMV library (own source file).
    mlx_gemv_library: metal::Library,
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    /// MLX GEMM/GEMV pipelines, keyed by (kernel name, function constants).
    /// `None` caches a failed specialization (e.g. bf16 on an OS without
    /// `bfloat`) so the caller falls back to the generic kernels without
    /// retrying.
    mlx_pipelines: Mutex<HashMap<String, Option<metal::ComputePipelineState>>>,
    pool: Mutex<BufferPool>,
    ctx: Mutex<OpCtx>,
    /// 0 = off; 1 = per-kernel profiling (flush after every command); 2 =
    /// batch-level profiling (normal batching, per-command-buffer stats).
    profile_mode: u8,
    /// MLX gemv kernels for large m == 1 matmuls (disable with
    /// `XN_METAL_NO_MLX_GEMV=1`).
    use_mlx_gemv: bool,
    pstats: Mutex<ProfStats>,
    device_name: String,
}

#[derive(Clone)]
pub struct Device(Arc<DeviceInner>);

impl std::ops::Deref for Device {
    type Target = DeviceInner;
    fn deref(&self) -> &DeviceInner {
        &self.0
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice").field("name", &self.device_name).finish()
    }
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let devices = metal::Device::all();
        if devices.is_empty() {
            crate::bail!("metal: no devices found");
        }
        // `ordinal` selects among the enumerated devices. An explicit
        // `XN_METAL_DEVICE` env var overrides the ordinal.
        let idx = std::env::var("XN_METAL_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(ordinal);
        // Out-of-range ordinals fall back to the system default device.
        let device = match devices.into_iter().nth(idx).or_else(metal::Device::system_default) {
            Some(d) => d,
            None => crate::bail!("metal: no device at index {idx}"),
        };
        let device_name = device.name().to_string();
        let queue = device.new_command_queue();

        // Compile the dtype variants of the kernel library.
        let compile = |defines: &str, body: &str| -> Result<metal::Library> {
            let src = format!("{defines}{DTYPE_SRC}{body}");
            let options = metal::CompileOptions::new();
            device
                .new_library_with_source(&src, &options)
                .map_err(|e| crate::Error::msg(format!("metal: kernel compilation failed: {e}")))
        };
        let body: String = KERNEL_SRCS.concat();
        let int_body: String = KERNEL_SRCS_INT.concat();
        let libraries = [
            compile("", &body)?,
            compile("#define USE_F16 1\n", &body)?,
            compile("#define USE_BF16 1\n", &body)?,
            compile("#define USE_I64 1\n", &int_body)?,
            compile("#define USE_U8 1\n", &int_body)?,
        ];
        let mlx_gemm_library = device
            .new_library_with_source(MLX_GEMM_SRC, &metal::CompileOptions::new())
            .map_err(|e| crate::Error::msg(format!("metal: mlx gemm compilation failed: {e}")))?;
        let mlx_gemv_library = device
            .new_library_with_source(MLX_GEMV_SRC, &metal::CompileOptions::new())
            .map_err(|e| crate::Error::msg(format!("metal: mlx gemv compilation failed: {e}")))?;

        let profile_mode = match std::env::var("XN_METAL_PROFILE").ok().as_deref() {
            None | Some("") | Some("0") => 0,
            Some("2") => 2,
            Some(_) => 1,
        };
        let use_mlx_gemv =
            !std::env::var("XN_METAL_NO_MLX_GEMV").is_ok_and(|v| !v.is_empty() && v != "0");

        let inner = DeviceInner {
            device,
            queue,
            libraries,
            mlx_gemm_library,
            mlx_gemv_library,
            pipelines: Mutex::new(HashMap::new()),
            mlx_pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            ctx: Mutex::new(OpCtx {
                cmd_buffer: None,
                encoder: None,
                n_dispatches: 0,
                free_bufs: Vec::new(),
                prof_names: Vec::new(),
                in_flight: std::collections::VecDeque::new(),
            }),
            profile_mode,
            use_mlx_gemv,
            pstats: Mutex::new(ProfStats::default()),
            device_name,
        };
        Ok(Self(Arc::new(inner)))
    }

    /// Whether this device supports f16 compute + storage. Always true on
    /// Metal (`half` is a core MSL type); kept for API parity with the Vulkan
    /// backend.
    pub fn supports_f16(&self) -> bool {
        true
    }

    /// Whether this device supports bf16 storage (emulated over `ushort`
    /// buffers with f32 compute). Always true on Metal; kept for API parity
    /// with the Vulkan backend.
    pub fn supports_bf16(&self) -> bool {
        true
    }

    /// Allocate a buffer of at least `size_bytes`, reusing a pooled buffer of
    /// the same size class when one is available. Returns the buffer, its
    /// shared-memory pointer, and the size class (needed to return the buffer
    /// to the pool on free).
    fn alloc_buffer(&self, size_bytes: usize) -> Result<(metal::Buffer, *mut u8, u64)> {
        let class = size_class(size_bytes);
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(b) = pool.free.get_mut(&class).and_then(|v| v.pop()) {
                pool.hits += 1;
                return Ok((b.buffer, b.ptr as *mut u8, class));
            }
            pool.misses += 1;
        }
        // Buffers are created with the full class size so that any same-class
        // request can reuse them.
        let buffer = self.device.new_buffer(class, metal::MTLResourceOptions::StorageModeShared);
        let ptr = buffer.contents() as *mut u8;
        if ptr.is_null() {
            crate::bail!("metal: buffer allocation of {class} bytes failed");
        }
        Ok((buffer, ptr, class))
    }

    fn get_pipeline(
        &self,
        name: &str,
    ) -> Result<(metal::ComputePipelineState, u32, (u64, u64, u64))> {
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(p) = pipelines.get(name) {
            return Ok((p.pipeline.clone(), p.bindings, p.wg));
        }
        let (lib_idx, entry, bindings, wg) = kernel_def(name)
            .ok_or_else(|| crate::Error::msg(format!("metal: unknown kernel {name}")))?;
        let function =
            self.libraries[lib_idx].get_function(entry, None).map_err(mtlerr("get_function"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(mtlerr("new_compute_pipeline_state"))?;
        pipelines
            .insert(name.to_string(), CachedPipeline { pipeline: pipeline.clone(), bindings, wg });
        Ok((pipeline, bindings, wg))
    }

    /// Look up (or build) an MLX GEMM pipeline specialization. Returns `None`
    /// when the specialization is unavailable (e.g. bf16 on an OS without
    /// `bfloat` support); the failure is cached so the caller falls back to
    /// the plain tiled kernel without retrying the compilation.
    fn get_mlx_gemm_pipeline(
        &self,
        trans: &str,
        dt: &str,
        aligned: (bool, bool, bool),
        has_batch: bool,
    ) -> Option<metal::ComputePipelineState> {
        let (align_m, align_n, align_k) = aligned;
        let name = format!("gemm_{trans}_{dt}_{dt}_32_32_16_2_2");
        let key = format!(
            "{name}_{}{}{}{}",
            align_m as u8, align_n as u8, align_k as u8, has_batch as u8
        );
        let mut cache = self.mlx_pipelines.lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        let set_bool = |v: &bool, idx: u64| {
            fcv.set_constant_value_at_index(
                v as *const bool as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                idx,
            );
        };
        set_bool(&has_batch, 10);
        set_bool(&false, 100); // use_out_source
        set_bool(&false, 110); // do_axpby
        set_bool(&align_m, 200);
        set_bool(&align_n, 201);
        set_bool(&align_k, 202);
        set_bool(&false, 300); // do_gather
        let pipeline = self
            .mlx_gemm_library
            .get_function(&name, Some(fcv))
            .ok()
            .and_then(|f| self.device.new_compute_pipeline_state_with_function(&f).ok());
        cache.insert(key, pipeline.clone());
        pipeline
    }

    /// Look up (or build) an MLX GEMV pipeline. Returns `None` (cached) when
    /// the instantiation is unavailable, e.g. bf16 without `bfloat` support.
    fn get_mlx_gemv_pipeline(&self, name: &str) -> Option<metal::ComputePipelineState> {
        let mut cache = self.mlx_pipelines.lock().unwrap();
        if let Some(p) = cache.get(name) {
            return p.clone();
        }
        let pipeline = self
            .mlx_gemv_library
            .get_function(name, None)
            .ok()
            .and_then(|f| self.device.new_compute_pipeline_state_with_function(&f).ok());
        cache.insert(name.to_string(), pipeline.clone());
        pipeline
    }

    /// Record an MLX GEMV dispatch into the current batch. `mat`/`vec` carry
    /// byte offsets; `sizes` is (in_vec_size, out_vec_size, matrix ld).
    fn dispatch_mlx_gemv(
        &self,
        pipeline: &metal::ComputePipelineState,
        name: &str,
        mat: (&metal::BufferRef, u64),
        vec: (&metal::BufferRef, u64),
        out: &metal::BufferRef,
        sizes: (i32, i32, i32),
        batch_strides: (u64, u64),
        grid: (u64, u64, u64),
        group: (u64, u64, u64),
    ) -> Result<()> {
        use std::ffi::c_void;
        let mut ctx = self.ctx.lock().unwrap();
        let enc = self.compute_encoder(&mut ctx)?;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(mat.0), mat.1);
        enc.set_buffer(1, Some(vec.0), vec.1);
        enc.set_buffer(2, Some(out), 0);
        let (in_size, out_size, ld) = sizes;
        enc.set_bytes(3, 4, &in_size as *const i32 as *const c_void);
        enc.set_bytes(4, 4, &out_size as *const i32 as *const c_void);
        enc.set_bytes(5, 4, &ld as *const i32 as *const c_void);
        enc.set_bytes(6, 8, &batch_strides.0 as *const u64 as *const c_void);
        enc.set_bytes(7, 8, &batch_strides.1 as *const u64 as *const c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(grid.0, grid.1, grid.2),
            metal::MTLSize::new(group.0, group.1, group.2),
        );
        ctx.n_dispatches += 1;
        if self.profile_mode == 1 {
            let name = format!("mlx_{name} (n={out_size}, k={in_size})");
            self.profile_flush(&mut ctx, name)?;
        } else {
            self.maybe_submit(&mut ctx)?;
        }
        Ok(())
    }

    /// Record an MLX GEMM dispatch into the current batch. `a`/`b` carry byte
    /// offsets; grid is (tiles_n, tiles_m, batch) with a (32, 2, 2)
    /// threadgroup (one simdgroup per (wn, wm) tile quadrant).
    fn dispatch_mlx_gemm(
        &self,
        pipeline: &metal::ComputePipelineState,
        a: (&metal::BufferRef, u64),
        b: (&metal::BufferRef, u64),
        d: &metal::BufferRef,
        params: &MlxGemmParams,
        batch: i32,
        batch_strides: &[isize; 2],
        groups: (u64, u64, u64),
    ) -> Result<()> {
        use std::ffi::c_void;
        let mut ctx = self.ctx.lock().unwrap();
        let enc = self.compute_encoder(&mut ctx)?;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a.0), a.1);
        enc.set_buffer(1, Some(b.0), b.1);
        enc.set_buffer(3, Some(d), 0);
        enc.set_bytes(
            4,
            std::mem::size_of::<MlxGemmParams>() as u64,
            params as *const MlxGemmParams as *const c_void,
        );
        enc.set_bytes(6, std::mem::size_of::<i32>() as u64, &batch as *const i32 as *const c_void);
        enc.set_bytes(
            7,
            std::mem::size_of::<[isize; 2]>() as u64,
            batch_strides.as_ptr() as *const c_void,
        );
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups.0, groups.1, groups.2),
            metal::MTLSize::new(32, 2, 2),
        );
        ctx.n_dispatches += 1;
        if self.profile_mode == 1 {
            let name = format!("mlx_gemm (m={}, n={}, k={})", params.m, params.n, params.k);
            self.profile_flush(&mut ctx, name)?;
        } else {
            self.maybe_submit(&mut ctx)?;
        }
        Ok(())
    }

    /// Record a single dispatch of `kernel` (1D threadgroup count).
    fn dispatch(
        &self,
        kernel: &str,
        buffers: &[&metal::BufferRef],
        push: &Pc,
        groups_x: u32,
    ) -> Result<()> {
        self.dispatch_nd(kernel, buffers, push, (groups_x, 1, 1))
    }

    /// Record a dispatch of `kernel` with an explicit 3D threadgroup count
    /// into the current batch (deferred; submitted on the next flush).
    fn dispatch_nd(
        &self,
        kernel: &str,
        buffers: &[&metal::BufferRef],
        push: &Pc,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let (gx, gy, gz) = groups;
        if gx == 0 || gy == 0 || gz == 0 {
            return Ok(());
        }
        let (pipeline, bindings, wg) = self.get_pipeline(kernel)?;
        assert_eq!(bindings as usize, buffers.len(), "kernel {kernel} binding count mismatch");
        let mut ctx = self.ctx.lock().unwrap();
        let enc = self.compute_encoder(&mut ctx)?;
        enc.set_compute_pipeline_state(&pipeline);
        for (i, b) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_bytes(
            buffers.len() as u64,
            push.bytes.len() as u64,
            push.bytes.as_ptr() as *const std::ffi::c_void,
        );
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gx as u64, gy as u64, gz as u64),
            metal::MTLSize::new(wg.0, wg.1, wg.2),
        );
        ctx.n_dispatches += 1;
        if self.profile_mode == 1 {
            // The matmul kernels' push constants start with (m, n, k); decode
            // them so the profile splits matmuls by shape.
            let is_matmul = kernel.starts_with("gemv") || kernel.starts_with("gemm_tiled");
            let name = if is_matmul && push.bytes.len() >= 12 {
                let f =
                    |i: usize| u32::from_le_bytes(push.bytes[4 * i..4 * i + 4].try_into().unwrap());
                format!("{kernel} (m={}, n={}, k={})", f(0), f(1), f(2))
            } else {
                kernel.to_string()
            };
            self.profile_flush(&mut ctx, name)?;
        } else {
            self.maybe_submit(&mut ctx)?;
        }
        Ok(())
    }

    /// Record a buffer-to-buffer copy of `bytes` into the current batch. The
    /// copy runs in a blit encoder; automatic hazard tracking orders it with
    /// the surrounding compute encoders.
    fn record_copy(
        &self,
        dst: &metal::BufferRef,
        src: &metal::BufferRef,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut ctx = self.ctx.lock().unwrap();
        // Blits need their own encoder type: close the compute encoder first.
        if let Some(enc) = ctx.encoder.take() {
            enc.end_encoding();
        }
        let cmd = self.command_buffer(&mut ctx)?.to_owned();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(src, 0, dst, 0, bytes as u64);
            blit.end_encoding();
        });
        ctx.n_dispatches += 1;
        if self.profile_mode == 1 {
            self.profile_flush(&mut ctx, "buffer_copy".to_string())?;
        } else {
            self.maybe_submit(&mut ctx)?;
        }
        Ok(())
    }

    /// Profiling mode: attribute the just-recorded command to `name` and
    /// flush it in its own command buffer so its GPU start/end times measure
    /// exactly that command.
    fn profile_flush(&self, ctx: &mut OpCtx, name: String) -> Result<()> {
        ctx.prof_names.push(name);
        self.flush_locked(ctx, "per-kernel profiling")
    }

    /// Record a buffer fill with a repeated byte into the current batch (used
    /// for zero fills of any dtype, keeping the pipeline intact).
    fn record_fill(&self, dst: &metal::BufferRef, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut ctx = self.ctx.lock().unwrap();
        // Blits need their own encoder type: close the compute encoder first.
        if let Some(enc) = ctx.encoder.take() {
            enc.end_encoding();
        }
        let cmd = self.command_buffer(&mut ctx)?.to_owned();
        autoreleasepool(|| {
            let blit = cmd.new_blit_command_encoder();
            blit.fill_buffer(dst, metal::NSRange::new(0, bytes as u64), value);
            blit.end_encoding();
        });
        ctx.n_dispatches += 1;
        if self.profile_mode == 1 {
            self.profile_flush(&mut ctx, "buffer_fill".to_string())?;
        } else {
            self.maybe_submit(&mut ctx)?;
        }
        Ok(())
    }

    /// The current batch's command buffer, creating it if needed.
    fn command_buffer<'a>(&self, ctx: &'a mut OpCtx) -> Result<&'a metal::CommandBuffer> {
        if ctx.cmd_buffer.is_none() {
            let cmd = autoreleasepool(|| self.queue.new_command_buffer().to_owned());
            ctx.cmd_buffer = Some(cmd);
        }
        Ok(ctx.cmd_buffer.as_ref().unwrap())
    }

    /// The current batch's serial compute encoder, creating the command
    /// buffer and/or encoder if needed. The serial dispatch type makes
    /// successive dispatches execute (and become visible) in order.
    fn compute_encoder<'a>(&self, ctx: &'a mut OpCtx) -> Result<&'a metal::ComputeCommandEncoder> {
        if ctx.encoder.is_none() {
            let cmd = self.command_buffer(ctx)?.to_owned();
            let enc = autoreleasepool(|| {
                cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial)
                    .to_owned()
            });
            ctx.encoder = Some(enc);
        }
        Ok(ctx.encoder.as_ref().unwrap())
    }

    /// Submit any pending recorded commands and wait for completion. Safe to
    /// call when nothing is pending. `reason` labels the wait in the
    /// profiling stats (`XN_METAL_PROFILE=2` prints a per-cause breakdown).
    fn flush(&self, reason: &'static str) -> Result<()> {
        let mut ctx = self.ctx.lock().unwrap();
        self.flush_locked(&mut ctx, reason)
    }

    fn flush_locked(&self, ctx: &mut OpCtx, reason: &'static str) -> Result<()> {
        self.submit_locked(ctx);
        if ctx.in_flight.is_empty() {
            return Ok(());
        }
        let t0 = (self.profile_mode > 0).then(std::time::Instant::now);
        // The queue executes command buffers in commit order, so waiting on
        // the newest one means every older one has completed too.
        if let Some(last) = ctx.in_flight.back() {
            last.cmd.wait_until_completed();
        }
        if let Some(t0) = t0 {
            let wait_ns = t0.elapsed().as_nanos();
            let mut stats = self.pstats.lock().unwrap();
            stats.flushes += 1;
            stats.wait_ns += wait_ns;
            let e = stats.wait_reasons.entry(reason).or_insert((0, 0));
            e.0 += 1;
            e.1 += wait_ns;
        }
        while !ctx.in_flight.is_empty() {
            self.retire_one(ctx);
        }
        Ok(())
    }

    /// Commit the currently-recorded batch without waiting, so the GPU starts
    /// executing it while the host keeps recording. No-op when nothing is
    /// recorded.
    fn submit_locked(&self, ctx: &mut OpCtx) {
        if let Some(enc) = ctx.encoder.take() {
            enc.end_encoding();
        }
        if let Some(cmd) = ctx.cmd_buffer.take() {
            cmd.commit();
            ctx.in_flight.push_back(InFlight {
                cmd,
                free_bufs: std::mem::take(&mut ctx.free_bufs),
                names: std::mem::take(&mut ctx.prof_names),
            });
        }
        ctx.n_dispatches = 0;
    }

    /// Pipelined submission: once `SUBMIT_CHUNK` commands are recorded,
    /// commit the batch without waiting. Completed batches retire eagerly so
    /// their buffers recycle; if too many batches are outstanding, block on
    /// the oldest for backpressure.
    fn maybe_submit(&self, ctx: &mut OpCtx) -> Result<()> {
        if ctx.n_dispatches < SUBMIT_CHUNK {
            return Ok(());
        }
        self.submit_locked(ctx);
        while ctx
            .in_flight
            .front()
            .is_some_and(|f| f.cmd.status() == metal::MTLCommandBufferStatus::Completed)
        {
            self.retire_one(ctx);
        }
        if ctx.in_flight.len() > MAX_IN_FLIGHT {
            let t0 = (self.profile_mode > 0).then(std::time::Instant::now);
            ctx.in_flight.front().unwrap().cmd.wait_until_completed();
            if let Some(t0) = t0 {
                let wait_ns = t0.elapsed().as_nanos();
                let mut stats = self.pstats.lock().unwrap();
                stats.flushes += 1;
                stats.wait_ns += wait_ns;
                let e = stats.wait_reasons.entry("backpressure").or_insert((0, 0));
                e.0 += 1;
                e.1 += wait_ns;
            }
            self.retire_one(ctx);
        }
        Ok(())
    }

    /// Retire the oldest submitted batch, which must have completed: collect
    /// profiling stats and recycle the buffers it referenced.
    fn retire_one(&self, ctx: &mut OpCtx) {
        let Some(fl) = ctx.in_flight.pop_front() else {
            return;
        };
        if self.profile_mode > 0 {
            // GPUStartTime/GPUEndTime are not exposed by metal-rs; call the
            // Objective-C selectors directly (CFTimeInterval seconds).
            let (gpu_t0, gpu_t1) = command_buffer_gpu_times(&fl.cmd);
            let dt_ns = ((gpu_t1 - gpu_t0).max(0.0) * 1e9) as u128;
            let mut stats = self.pstats.lock().unwrap();
            stats.submissions += 1;
            if stats.last_gpu_end > 0.0 && gpu_t0 > stats.last_gpu_end {
                stats.gap_ns += ((gpu_t0 - stats.last_gpu_end) * 1e9) as u128;
            }
            stats.last_gpu_end = gpu_t1;
            stats.gpu_ns += dt_ns;
            // In per-kernel mode each batch holds exactly one command/name.
            for name in fl.names {
                let e = stats.per_kernel.entry(name).or_insert((0, 0));
                e.0 += 1;
                e.1 += dt_ns;
                stats.dispatches += 1;
            }
        }
        // This batch has completed on the GPU: buffers freed during it can be
        // recycled for future allocations.
        if !fl.free_bufs.is_empty() {
            let mut pool = self.pool.lock().unwrap();
            for b in fl.free_bufs {
                pool.free.entry(b.class).or_default().push(b);
            }
        }
    }
}

impl DeviceInner {
    /// Print accumulated per-kernel GPU times (profiling mode only).
    fn print_profile(&self) {
        let stats = self.pstats.lock().unwrap();
        if stats.submissions == 0 {
            return;
        }
        eprintln!("\n=== xn metal profile: {} ===", self.device_name);
        if !stats.per_kernel.is_empty() {
            let mut rows: Vec<_> = stats.per_kernel.iter().collect();
            rows.sort_by_key(|r| std::cmp::Reverse(r.1.1));
            eprintln!(
                "{:<38} {:>9} {:>11} {:>9} {:>7}",
                "kernel", "count", "total ms", "avg us", "%gpu"
            );
            for (name, (cnt, ns)) in rows {
                eprintln!(
                    "{:<38} {:>9} {:>11.2} {:>9.1} {:>6.1}%",
                    name,
                    cnt,
                    *ns as f64 / 1e6,
                    *ns as f64 / 1e3 / *cnt as f64,
                    100.0 * *ns as f64 / stats.gpu_ns as f64,
                );
            }
        }
        eprintln!(
            "gpu busy: {:.2} ms across {} command buffers; gpu idle between buffers: {:.2} ms",
            stats.gpu_ns as f64 / 1e6,
            stats.submissions,
            stats.gap_ns as f64 / 1e6,
        );
        eprintln!(
            "cpu blocking waits: {} totalling {:.2} ms",
            stats.flushes,
            stats.wait_ns as f64 / 1e6,
        );
        let mut reasons: Vec<_> = stats.wait_reasons.iter().collect();
        reasons.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        for (reason, (cnt, ns)) in reasons {
            eprintln!("  {:<28} {:>9} waits {:>11.2} ms", reason, cnt, *ns as f64 / 1e6);
        }
        let pool = self.pool.lock().unwrap();
        let total = pool.hits + pool.misses;
        if total > 0 {
            eprintln!(
                "buffer pool: {} hits / {} allocs ({:.1}% reuse)",
                pool.hits,
                total,
                100.0 * pool.hits as f64 / total as f64,
            );
        }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if self.profile_mode > 0 {
            self.print_profile();
        }
        // The last batch may still be open: recorded but never submitted
        // because nothing read its results back. Metal asserts if a live
        // command encoder is released without endEncoding, so close it here.
        // The uncommitted command buffer (and its now-unreachable results)
        // can simply be dropped; in-flight buffers are kept alive by the
        // queue until they finish executing.
        if let Ok(ctx) = self.ctx.get_mut() {
            if let Some(enc) = ctx.encoder.take() {
                enc.end_encoding();
            }
            ctx.cmd_buffer = None;
        }
    }
}

/// Metal tensor storage: a `StorageModeShared` buffer, accessed from the host
/// through its `contents()` pointer.
pub struct Storage<T: WithDType> {
    buffer: metal::Buffer,
    ptr: *mut u8,
    len: usize,
    /// Allocation size class; used to return the buffer to the pool on drop.
    class: u64,
    /// Whether the buffer has ever been referenced by a recorded GPU command.
    /// While false, host reads/writes need no pipeline drain: the pool only
    /// recycles buffers from completed batches, so a fresh storage has no
    /// pending GPU references.
    gpu_used: std::sync::atomic::AtomicBool,
    device: Device,
    _t: PhantomData<T>,
}

// The shared-memory pointer is only accessed while holding a `&`/`&mut` to
// the storage; the device serializes GPU work. Safe to move across threads.
unsafe impl<T: WithDType> Send for Storage<T> {}
unsafe impl<T: WithDType> Sync for Storage<T> {}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying buffer, for use in a recorded GPU command; marks the
    /// storage as GPU-referenced so later host accesses drain the pipeline.
    fn buf(&self) -> &metal::BufferRef {
        self.gpu_used.store(true, std::sync::atomic::Ordering::Relaxed);
        &self.buffer
    }

    fn is_gpu_used(&self) -> bool {
        self.gpu_used.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Host view of the shared memory as `&[T]`.
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

impl<T: WithDType> Drop for Storage<T> {
    fn drop(&mut self) {
        // The current (unsubmitted) batch may still reference this buffer, so
        // defer recycling it until the next flush completes on the GPU.
        self.device.defer_free(PooledBuf {
            buffer: self.buffer.clone(),
            ptr: self.ptr as usize,
            class: self.class,
        });
    }
}

impl Device {
    /// Allocate a small shared buffer holding `data`, for passing `info`
    /// arrays (dims/strides) to kernels. The caller must pass the returned
    /// `PooledBuf` to [`Self::defer_free`] *after* recording the dispatch that
    /// uses it (see the `defer_free` invariant).
    fn scratch_u32(&self, data: &[u32]) -> Result<PooledBuf> {
        let (buffer, ptr, class) = self.alloc_buffer(data.len() * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, data.len() * 4);
        }
        Ok(PooledBuf { buffer, ptr: ptr as usize, class })
    }

    /// Schedule a buffer to be recycled into the pool on the next flush.
    ///
    /// Invariant: this must only be called once *every* use of the buffer has
    /// been recorded into the command buffer. Recycling happens when a batch
    /// is flushed, and a flush may be forced in the middle of an op sequence
    /// (see `dispatch_nd`); a buffer deferred before its use is recorded would
    /// be recycled by such a flush and overwritten while the subsequently-
    /// recorded dispatch still references it.
    fn defer_free(&self, buf: PooledBuf) {
        self.ctx.lock().unwrap().free_bufs.push(buf);
    }
}

fn check_f32<T: WithDType>(op: &str) -> Result<()> {
    if T::DTYPE != DType::F32 {
        crate::bail!("metal: {op} only supports f32, got {:?}", T::DTYPE);
    }
    Ok(())
}

/// Kernel dtype suffix ("f32"/"f16"/"bf16") for a float storage type; errors
/// on other dtypes.
fn dtype_suffix<T: WithDType>(_dev: &Device, op: &str) -> Result<&'static str> {
    match T::DTYPE {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => crate::bail!("metal: {op} supports f32/f16/bf16, got {d:?}"),
    }
}

/// Suffix for the GPU path of ops whose scalar parameters are pushed as f32,
/// or `None` to take the host fallback (non-float dtypes).
fn float_suffix<T: WithDType>(_dev: &Device) -> Option<&'static str> {
    match T::DTYPE {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
    }
}

/// Kernel dtype suffix for any storage dtype; the dtype-generic kernels exist
/// in all five library variants.
fn any_suffix<T: WithDType>() -> &'static str {
    match T::DTYPE {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I64 => "i64",
        DType::U8 => "u8",
    }
}

fn unary_op_code(op: UnaryOp) -> (u32, f32) {
    match op {
        UnaryOp::Cos => (0, 0.0),
        UnaryOp::Sin => (1, 0.0),
        UnaryOp::Exp => (2, 0.0),
        UnaryOp::Log => (3, 0.0),
        UnaryOp::Neg => (4, 0.0),
        UnaryOp::Sqr => (5, 0.0),
        UnaryOp::Sqrt => (6, 0.0),
        UnaryOp::Rsqrt => (7, 0.0),
        UnaryOp::Abs => (8, 0.0),
        UnaryOp::GeluErf => (9, 0.0),
        UnaryOp::Elu { alpha } => (10, alpha),
        UnaryOp::Relu => (11, 0.0),
        UnaryOp::Silu => (12, 0.0),
        UnaryOp::Tanh => (13, 0.0),
        UnaryOp::Sigmoid => (14, 0.0),
    }
}

fn binary_op_code(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Maximum => 4,
        BinaryOp::Minimum => 5,
    }
}

fn div_ceil(n: usize, d: u32) -> u32 {
    (n as u32).div_ceil(d)
}

include!("backend_impl.rs");
