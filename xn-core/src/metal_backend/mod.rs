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
    let (base, dt) = name.rsplit_once('_')?;
    let wg1d = (WORKGROUP_SIZE as u64, 1, 1);
    // (bindings, threads-per-threadgroup, f32-only)
    let (bindings, wg, f32_only) = match base {
        "unary" => (2, wg1d, false),
        "binary" => (3, wg1d, false),
        "scale_add" => (2, wg1d, false),
        "broadcast" => (4, wg1d, false),
        "softmax" => (2, wg1d, false),
        "rmsnorm" => (3, wg1d, false),
        "layernorm" => (4, wg1d, false),
        "rope" => (4, wg1d, false),
        "rope_i" => (4, wg1d, false),
        "reduce" => (2, wg1d, false),
        "reduce_arg" => (2, wg1d, false),
        "transpose" => (2, wg1d, false),
        "copy2d" => (2, wg1d, false),
        "copy_strided" => (3, wg1d, false),
        "index_select" => (3, wg1d, false),
        "causality_mask" => (1, wg1d, false),
        "scatter_set" => (3, wg1d, false),
        "gemm_tiled" => (3, (16, 16, 1), false),
        "gemv" => (3, wg1d, false),
        // conv kernels are f32-only; other dtypes must fail pipeline lookup.
        "conv1d" => (3, wg1d, true),
        "conv_transpose1d" => (3, wg1d, true),
        _ => return None,
    };
    let lib_idx = match dt {
        "f16" if !f32_only => 1,
        "bf16" if !f32_only => 2,
        "f32" => 0,
        _ => return None,
    };
    Some((lib_idx, base, bindings, wg))
}

const WORKGROUP_SIZE: u32 = 256;

/// Max dispatches per batch before we force a flush. Metal has no hard limit
/// here; this just bounds how much recorded-but-unsubmitted work (and how many
/// to-be-recycled buffers) can accumulate.
const MAX_DISPATCHES_PER_BATCH: u32 = 4096;

fn mtlerr<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> crate::Error + '_ {
    move |e| crate::Error::msg(format!("metal: {context}: {e:?}"))
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

/// Command-recording state, guarded by a mutex.
///
/// Dispatches are recorded into a serial compute encoder on `cmd_buffer` and
/// only submitted when the batch is flushed (on host readback / `synchronize`
/// / before host access to shared memory). This keeps the GPU busy across many
/// ops instead of paying a CPU<->GPU round-trip per op.
struct OpCtx {
    /// Open command buffer holding recorded, unsubmitted commands.
    cmd_buffer: Option<metal::CommandBuffer>,
    /// Open serial compute encoder on `cmd_buffer` (closed around blits).
    encoder: Option<metal::ComputeCommandEncoder>,
    /// Number of commands recorded in the current batch.
    n_dispatches: u32,
    /// Buffers (dropped tensors + scratch) to recycle into the pool on the
    /// next flush, once any batch referencing them has finished executing.
    free_bufs: Vec<PooledBuf>,
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
    /// Per-dtype-variant libraries: [f32, f16, bf16].
    libraries: [metal::Library; 3],
    /// MLX steel GEMM library (simdgroup-matrix kernels, own source file).
    mlx_gemm_library: metal::Library,
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    /// MLX GEMM pipelines, keyed by (kernel name, function constants). `None`
    /// caches a failed specialization (e.g. bf16 on an OS without `bfloat`) so
    /// the caller falls back to the plain tiled kernel without retrying.
    mlx_pipelines: Mutex<HashMap<String, Option<metal::ComputePipelineState>>>,
    pool: Mutex<BufferPool>,
    ctx: Mutex<OpCtx>,
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

        // Compile the three dtype variants of the kernel library.
        let body: String = KERNEL_SRCS.concat();
        let compile = |defines: &str| -> Result<metal::Library> {
            let src = format!("{defines}{DTYPE_SRC}{body}");
            let options = metal::CompileOptions::new();
            device
                .new_library_with_source(&src, &options)
                .map_err(|e| crate::Error::msg(format!("metal: kernel compilation failed: {e}")))
        };
        let libraries =
            [compile("")?, compile("#define USE_F16 1\n")?, compile("#define USE_BF16 1\n")?];
        let mlx_gemm_library = device
            .new_library_with_source(MLX_GEMM_SRC, &metal::CompileOptions::new())
            .map_err(|e| crate::Error::msg(format!("metal: mlx gemm compilation failed: {e}")))?;

        let inner = DeviceInner {
            device,
            queue,
            libraries,
            mlx_gemm_library,
            pipelines: Mutex::new(HashMap::new()),
            mlx_pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            ctx: Mutex::new(OpCtx {
                cmd_buffer: None,
                encoder: None,
                n_dispatches: 0,
                free_bufs: Vec::new(),
            }),
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
        if ctx.n_dispatches >= MAX_DISPATCHES_PER_BATCH {
            self.flush_locked(&mut ctx)?;
        }
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
        if ctx.n_dispatches >= MAX_DISPATCHES_PER_BATCH {
            self.flush_locked(&mut ctx)?;
        }
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
        if ctx.n_dispatches >= MAX_DISPATCHES_PER_BATCH {
            self.flush_locked(&mut ctx)?;
        }
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
    /// call when nothing is pending.
    fn flush(&self) -> Result<()> {
        let mut ctx = self.ctx.lock().unwrap();
        self.flush_locked(&mut ctx)
    }

    fn flush_locked(&self, ctx: &mut OpCtx) -> Result<()> {
        if let Some(enc) = ctx.encoder.take() {
            enc.end_encoding();
        }
        if let Some(cmd) = ctx.cmd_buffer.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
        // The GPU is now idle for this batch: buffers freed during it can be
        // recycled for future allocations.
        if !ctx.free_bufs.is_empty() {
            let mut pool = self.pool.lock().unwrap();
            for b in ctx.free_bufs.drain(..) {
                pool.free.entry(b.class).or_default().push(b);
            }
        }
        ctx.n_dispatches = 0;
        Ok(())
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

/// Suffix for the GPU path of dtype-generic ops, or `None` to take the host
/// fallback (non-float dtypes).
fn float_suffix<T: WithDType>(_dev: &Device) -> Option<&'static str> {
    match T::DTYPE {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
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
