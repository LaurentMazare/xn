//! A persistent worker pool for fan-out inside a single operator.
//!
//! Batch-1 decode is a long sequence of individually tiny kernels: a 12-layer backbone issues
//! roughly fifty quantized matmuls per token, each a few tens of microseconds. Handing each one
//! to `rayon::join`-style work stealing means paying a fork/join per kernel -- measured on an M5
//! at about 10 us with 3 workers and 34 us with 8, against kernels that take 25-70 us. That is
//! why adding threads past three makes decode *slower*: the dispatch grows faster than the work
//! shrinks.
//!
//! ggml avoids it by never dispatching per operator. `ggml_graph_compute_kickoff` wakes the pool
//! once for a whole graph and the workers then walk the node list together, meeting at
//! `ggml_barrier` between nodes -- two atomics and a `yield` spin, comfortably under a
//! microsecond. xn is eager and has no graph to hand a pool, so this module takes the other half
//! of the idea: the workers are permanent and park only when the stream goes quiet, so a
//! dispatch is a sequence-number bump plus a spin, not a task submission.
//!
//! [`dispatch`] is deliberately shaped like the `ith`/`nth` split the kernels already use, so
//! call sites change by one line.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Iterations of `spin_loop` a worker burns before parking, overridable with `XN_SPIN_BUDGET`.
///
/// The budget has to bridge the gap between two consecutive parallel operators -- a few
/// microseconds for a decode step -- because catching the next job while still spinning is what
/// keeps a dispatch sub-microsecond. Overshooting is not free: anything else runnable on the
/// machine competes with the spinners, and on this codebase that is still the f32 matmul, which
/// lives in the `gemm` crate and drives rayon itself.
///
/// While the f32 matmul still drove rayon this had to stay near 1000 to avoid the two pools
/// fighting. With everything on one pool the tradeoff is gone and a longer bridge is uniformly
/// better: measured across 3-8 threads and both vocoder windows, 5000 is at or above every
/// other value tried, and roughly 20% better than 1000 at 8 threads.
///
/// An embedder whose threads are more expensive to wake can raise it through
/// [`PoolConfig::spin_budget`].
const DEFAULT_SPIN_BUDGET: u32 = 5_000;

fn spin_budget() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("XN_SPIN_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SPIN_BUDGET)
    })
}

type Job = dyn Fn(usize, usize) + Sync;

struct Shared {
    /// Bumped once per published job; workers compare it against what they last ran.
    seq: AtomicU64,
    /// Borrowed for exactly as long as `seq` designates an unfinished job. The publisher blocks
    /// until every participating worker has incremented `done`, which is the last thing they do
    /// after their final read, so the reference cannot outlive the caller's frame.
    job: std::cell::UnsafeCell<Option<*const Job>>,
    /// Workers (excluding the publisher) that have finished the current job.
    ///
    /// *Every* worker acknowledges *every* job, even one whose share is empty. That is what
    /// makes reading `job` safe: a worker latches `seq`, then reads the slot, and the publisher
    /// cannot have moved on in between because it is still waiting for this worker's
    /// acknowledgement. Letting some workers skip the count reintroduces that window, and a
    /// lagging worker then runs the *next* job twice and double-counts it.
    done: AtomicUsize,
    /// Set if any worker panicked; the publisher re-raises so a panic is not silently swallowed.
    panicked: AtomicBool,
    quit: AtomicBool,
    /// Park slot for workers whose spin budget ran out.
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` is only read by workers between a `seq` bump (Release) and their `done` bump,
// and only written by the publisher while no worker is inside that window.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

struct Pool {
    shared: &'static Shared,
    /// Total participants including the publisher, i.e. `workers.len() + 1`.
    size: usize,
    /// There is one job slot, so one publisher at a time. Claimed for the whole dispatch.
    ///
    /// Contention means a second thread is already driving the pool -- another stream in the
    /// same process, say. Waiting would be pointless: the cores are busy either way. So a
    /// contended dispatch runs serially on its own thread instead of queueing.
    ///
    /// A plain flag rather than a `Mutex`, because a panic escaping a dispatch would poison a
    /// mutex and silently demote every later dispatch to single-threaded. There is no data
    /// under this lock to be left inconsistent, so poisoning has nothing to protect.
    publish: AtomicBool,
}

thread_local! {
    /// Set while this thread is running a pool job. Nested dispatches run serially rather than
    /// deadlocking against a pool that is already fully occupied.
    static IN_JOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// How a pool is built when the caller brings the threads. See [`start_pool_with`].
pub struct PoolConfig {
    /// Worker threads the caller will spawn. The calling thread joins them as participant 0,
    /// so the pool ends up `workers + 1` wide.
    pub workers: usize,
    /// Iterations a worker spins before parking. `None` takes `XN_SPIN_BUDGET`, else
    /// [`DEFAULT_SPIN_BUDGET`]. Raise it where waking a parked thread is expensive.
    pub spin_budget: Option<u32>,
}

fn new_shared() -> &'static Shared {
    Box::leak(Box::new(Shared {
        seq: AtomicU64::new(0),
        job: std::cell::UnsafeCell::new(None),
        done: AtomicUsize::new(0),
        panicked: AtomicBool::new(false),
        quit: AtomicBool::new(false),
        lock: Mutex::new(()),
        wake: Condvar::new(),
    }))
}

/// Whether this platform lets the pool spawn threads of its own.
///
/// Asked rather than assumed, by spawning one and looking at the result: `std::thread` exists
/// on every target, but on some -- a wasm build for the browser, say -- it cannot start one
/// and says so. The probe costs one thread, once.
fn can_spawn_threads() -> bool {
    static CAN: OnceLock<bool> = OnceLock::new();
    *CAN.get_or_init(|| std::thread::Builder::new().name("xn-probe".into()).spawn(|| {}).is_ok())
}

/// The pool, once something has built it.
///
/// `None` until then on a platform that cannot spawn its own threads, which is not a failure:
/// callers fall back to a rayon fork/join, and an embedder that has threads to lend can still
/// build it with [`start_pool_with`]. Nothing is stored in that case, so a later call wins.
fn pool() -> Option<&'static Pool> {
    if let Some(pool) = POOL.get() {
        return Some(pool);
    }
    if !use_own_threads() || !can_spawn_threads() {
        return None;
    }
    Some(POOL.get_or_init(|| {
        let size = crate::get_num_threads().max(1);
        let shared = new_shared();
        let budget = spin_budget();
        for ith in 1..size {
            std::thread::Builder::new()
                .name(format!("xn-worker-{ith}"))
                .spawn(move || worker(shared, ith, size, budget))
                .expect("spawning an xn worker thread");
        }
        Pool { shared, size, publish: AtomicBool::new(false) }
    }))
}

/// Build the pool on threads the caller provides, rather than on `std::thread`s of its own,
/// which is how a target that cannot spawn any gets one.
///
/// `spawn` must put each job on a thread of its own and let it run forever: a worker never
/// returns, so whatever `spawn` draws on is committed to this pool. With `rayon::spawn` that
/// is rayon's threads, and nothing may go through rayon afterwards, including this module's
/// own fallback. The calling thread is participant 0 and must be one that may block.
///
/// Returns the pool's width, `workers + 1`; a second call is a no-op returning it.
pub fn start_pool_with(
    cfg: PoolConfig,
    spawn: impl Fn(Box<dyn FnOnce() + Send + 'static>),
) -> usize {
    let pool = POOL.get_or_init(|| {
        let size = cfg.workers + 1;
        let shared = new_shared();
        let budget = cfg.spin_budget.unwrap_or_else(spin_budget);
        for ith in 1..size {
            spawn(Box::new(move || worker(shared, ith, size, budget)));
        }
        Pool { shared, size, publish: AtomicBool::new(false) }
    });
    pool.size
}

/// `nth` is fixed for the lifetime of the pool, so a worker never has to read it.
fn worker(shared: &'static Shared, ith: usize, nth: usize, budget: u32) {
    let mut last = 0u64;
    loop {
        // Spin first: between two operators of the same layer the next job usually lands within
        // a few microseconds, and catching it here is what keeps dispatch cheap.
        let mut seq = shared.seq.load(Ordering::Acquire);
        let mut spins = 0u32;
        while seq == last && !shared.quit.load(Ordering::Relaxed) {
            if spins < budget {
                spins += 1;
                std::hint::spin_loop();
            } else {
                // The stream has gone quiet; give the core back.
                let guard = shared.lock.lock().unwrap_or_else(|e| e.into_inner());
                if shared.seq.load(Ordering::Acquire) == last
                    && !shared.quit.load(Ordering::Relaxed)
                {
                    let _unused = shared.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                spins = 0;
            }
            seq = shared.seq.load(Ordering::Acquire);
        }
        if shared.quit.load(Ordering::Relaxed) {
            return;
        }
        last = seq;

        // SAFETY: the publisher wrote `job` before the Release bump of `seq` we just acquired,
        // and cannot publish another until this worker bumps `done` below.
        let job = unsafe {
            (*shared.job.get()).expect("a latched sequence number always has a job behind it")
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            IN_JOB.with(|f| f.set(true));
            // SAFETY: same window as above; the publisher's frame outlives this call.
            unsafe { (*job)(ith, nth) };
        }));
        IN_JOB.with(|f| f.set(false));
        if result.is_err() {
            shared.panicked.store(true, Ordering::Relaxed);
        }
        shared.done.fetch_add(1, Ordering::Release);
    }
}

/// Whether the pool may build itself. `XN_THREADPOOL=0` reverts to a rayon fork/join per
/// operator, which is the escape hatch if the resident workers ever fight with something else
/// for cores. It does not reject a pool an embedder supplies through [`start_pool_with`].
fn use_own_threads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XN_THREADPOOL").as_deref(), Ok("0")))
}

/// The threshold [`set_min_parallel_work`] holds. `0`, the default, fans out every size.
static MIN_PARALLEL_WORK: AtomicUsize = AtomicUsize::new(0);

/// The smallest operator, in multiply-adds or elements, that [`threads_for`] and
/// [`dispatch_work`] will fan out. `0`, the default, fans out every size.
///
/// A threshold is only worth setting where a fan-out is expensive relative to the work. This
/// pool's own dispatch costs microseconds, so once it is running every size is worth
/// splitting; it is the fallback path, a rayon fork/join, that can cost far more than a small
/// operator takes. An embedder on such a runtime sets this, and lowers it once its pool is
/// running.
pub fn set_min_parallel_work(work: usize) {
    MIN_PARALLEL_WORK.store(work, Ordering::Relaxed);
}

/// Threads worth putting on an operator of `work` multiply-adds (or elements).
///
/// [`size`] once `work` reaches [`set_min_parallel_work`]'s threshold, which by default is
/// every operator; `1` below it, since a fan-out that costs more than the operator is a loss.
pub(crate) fn threads_for(work: usize) -> usize {
    if work < MIN_PARALLEL_WORK.load(Ordering::Relaxed) {
        return 1;
    }
    size()
}

/// Holds the right to drive the pool, releasing it even if the dispatch unwinds.
struct PublishGuard(&'static Pool);

impl PublishGuard {
    fn claim(pool: &'static Pool) -> Option<Self> {
        // `then_some` would be wrong here: it builds its argument eagerly, so a *failed* claim
        // would still construct a guard, and dropping it would release the token whichever
        // other thread is currently holding -- two publishers, one job slot.
        if pool.publish.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
        {
            Some(Self(pool))
        } else {
            None
        }
    }
}

impl Drop for PublishGuard {
    fn drop(&mut self) {
        self.0.publish.store(false, Ordering::Release);
    }
}

/// Threads available to put on one operator, including the calling thread.
///
/// Fixed at first use with the pool, like rayon's global pool: [`crate::set_num_threads`]
/// after that point changes how work is chunked but cannot grow it. Without the pool it is
/// rayon's width, read afresh.
pub fn size() -> usize {
    match pool() {
        Some(pool) => pool.size,
        None => rayon::current_num_threads().max(1),
    }
}

/// [`dispatch`] for an operator whose size the caller knows, in multiply-adds or elements.
///
/// Runs it on [`threads_for`] threads, so with no threshold set this is [`dispatch`].
pub fn dispatch_work<F: Fn(usize, usize) + Sync>(work: usize, f: F) {
    let nth = threads_for(work);
    if nth <= 1 {
        f(0, 1);
        return;
    }
    if pool().is_some() {
        dispatch(f);
        return;
    }
    use rayon::prelude::*;
    (0..nth).into_par_iter().for_each(|ith| f(ith, nth));
}

/// Run `f(ith, nth)` on every participant and return once all of them have finished.
///
/// `nth` is always [`size`]; the caller is participant 0 and runs its share inline. Splitting
/// the work is the callee's job, and a share that comes out empty is free -- that is cheaper
/// than varying the participant count, which would let a lagging worker read a job that has
/// already been replaced.
///
/// `f` must give each `ith` a disjoint slice of the output: the pool provides the barrier, not
/// mutual exclusion.
pub fn dispatch<F: Fn(usize, usize) + Sync>(f: F) {
    let Some(pool) = pool() else {
        // Fan out through rayon, the way this used to work. A nested dispatch fans out
        // again here rather than collapsing to `(0, 1)`; rayon copes. The width is `size()`,
        // rayon's own, which no threshold bends: a caller that knows how much work it has
        // says so through `dispatch_work` and is gated there.
        let nth = size();
        use rayon::prelude::*;
        (0..nth).into_par_iter().for_each(|ith| f(ith, nth));
        return;
    };
    // A job that dispatches again would wait on workers already inside this job.
    if IN_JOB.with(|f| f.get()) {
        f(0, 1);
        return;
    }
    let nth = pool.size;
    if nth <= 1 {
        f(0, 1);
        return;
    }
    let Some(_publish) = PublishGuard::claim(pool) else {
        // Another thread owns the pool; see the field's documentation.
        f(0, 1);
        return;
    };
    let shared = pool.shared;

    let borrowed: &(dyn Fn(usize, usize) + Sync + '_) = &f;
    // Erase the lifetime. Sound because this frame does not return until every worker has
    // bumped `done`, which happens strictly after its last use of the reference.
    let job: *const Job = unsafe {
        std::mem::transmute::<*const (dyn Fn(usize, usize) + Sync + '_), *const Job>(borrowed)
    };

    // SAFETY: no worker can be reading `job` here -- the previous dispatch did not return until
    // every worker had acknowledged it, and this one is not published yet.
    unsafe { *shared.job.get() = Some(job) };
    shared.done.store(0, Ordering::Relaxed);
    shared.panicked.store(false, Ordering::Relaxed);
    // Release: everything above is visible to any worker that observes the new sequence number.
    shared.seq.fetch_add(1, Ordering::Release);
    // Only costs anything if a worker actually parked.
    {
        let _guard = shared.lock.lock().unwrap_or_else(|e| e.into_inner());
        shared.wake.notify_all();
    }

    let own = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, nth)));

    let want = nth - 1;
    let mut spins = 0u32;
    while shared.done.load(Ordering::Acquire) < want {
        if spins < spin_budget() {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    // Clear before returning so a stale pointer is never observable.
    // SAFETY: every worker has acknowledged; nobody is reading `job`.
    unsafe { *shared.job.get() = None };

    if let Err(payload) = own {
        std::panic::resume_unwind(payload);
    }
    if shared.panicked.load(Ordering::Relaxed) {
        panic!("an xn worker panicked while running a parallel operator");
    }
}

/// Run `f(lo, hi)` over `n` items in pieces of `unit`, on the pool when an operator of
/// `work` is worth splitting and on the calling thread otherwise.
///
/// The piece is the caller's unit of disjointness -- output channels, column blocks -- so it
/// says how to cut the work and this decides whether to.
pub fn par_units_by<F: Fn(usize, usize) + Sync>(work: usize, n: usize, unit: usize, f: F) {
    let units = n.div_ceil(unit.max(1));
    if threads_for(work).min(units) <= 1 {
        f(0, n);
        return;
    }
    par_units(units, |u| f(u * unit, ((u + 1) * unit).min(n)));
}

/// Run `f(i)` for every `i` in `0..n`, handing indices out on demand.
///
/// The counterpart to [`dispatch`], which splits statically. A static split is only right when
/// every participant runs at the same speed, and on a hybrid CPU it never does: an efficiency
/// core takes several times longer over an equal share, and the barrier waits for it. That
/// shows up as a long tail rather than a slow mean -- p95 and max frame times blow out while
/// p50 barely moves. Pulling from a shared counter lets the fast cores absorb the difference,
/// which is what ggml's `current_chunk` does for its own matmuls.
///
/// Indices are handed out in small consecutive runs so the atomic is touched a handful of
/// times per worker rather than once per index, while still leaving enough runs to balance.
pub fn par_units<F: Fn(usize) + Sync>(n: usize, f: F) {
    if n == 0 {
        return;
    }
    let nth = size();
    if n == 1 || nth == 1 {
        (0..n).for_each(&f);
        return;
    }
    // Enough runs to rebalance (a straggler costs at most one run), few enough that the
    // counter is not the bottleneck.
    let runs = (nth * 8).min(n);
    let batch = n.div_ceil(runs);
    let next = AtomicUsize::new(0);
    dispatch(|_, _| {
        loop {
            let start = next.fetch_add(batch, Ordering::Relaxed);
            if start >= n {
                break;
            }
            for i in start..(start + batch).min(n) {
                f(i);
            }
        }
    });
}

/// Split `dst` into `chunk`-sized pieces and run `f(index, piece)` on all of them.
///
/// The pool replacement for `dst.par_chunks_mut(chunk).enumerate().for_each(..)`. Each worker
/// takes a contiguous run of pieces, so a worker's writes stay in one region.
pub fn par_chunks_mut<T: Send + Sync, F>(dst: &mut [T], chunk: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    par_chunks_mut_by(dst.len(), dst, chunk, f)
}

/// [`par_chunks_mut`] for a call whose output elements cost more than a pass over them.
///
/// `dst.len()` says how to cut the output, `work` says how much there is to do: a matmul row
/// of `n` columns `k` deep is `n * k`, not `n`. They differ only for an embedder that set
/// [`set_min_parallel_work`], which weighs the latter.
pub fn par_chunks_mut_by<T: Send + Sync, F>(work: usize, dst: &mut [T], chunk: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    if chunk == 0 {
        return;
    }
    let len = dst.len();
    let nchunks = len.div_ceil(chunk);
    if nchunks <= 1 || threads_for(work) == 1 {
        for (i, d) in dst.chunks_mut(chunk).enumerate() {
            f(i, d);
        }
        return;
    }
    let base = dst.as_mut_ptr() as usize;
    par_units(nchunks, |c| {
        let start = c * chunk;
        let end = ((c + 1) * chunk).min(len);
        // SAFETY: each index is handed to exactly one worker, so the pieces do not overlap,
        // and every piece lies inside the original slice.
        let piece =
            unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(start), end - start) };
        f(c, piece);
    });
}

/// As [`par_chunks_mut`], but walking a read-only slice in step with the output.
///
/// The pool replacement for `src.par_chunks(chunk).zip(dst.par_chunks_mut(chunk)).for_each(..)`.
/// `src` is chunked by `src_chunk`, which lets the two sides have different row widths.
pub fn par_chunks_zip<T: Send + Sync, U: Sync, F>(
    dst: &mut [T],
    chunk: usize,
    src: &[U],
    src_chunk: usize,
    f: F,
) where
    F: Fn(usize, &mut [T], &[U]) + Sync,
{
    if chunk == 0 {
        return;
    }
    let len = dst.len();
    let src_len = src.len();
    let nchunks = len.div_ceil(chunk);
    if nchunks <= 1 || threads_for(len) == 1 {
        for (i, d) in dst.chunks_mut(chunk).enumerate() {
            let s = (i * src_chunk).min(src_len);
            let e = ((i + 1) * src_chunk).min(src_len);
            f(i, d, &src[s..e]);
        }
        return;
    }
    let base = dst.as_mut_ptr() as usize;
    let sbase = src.as_ptr() as usize;
    par_units(nchunks, |c| {
        let start = c * chunk;
        let end = ((c + 1) * chunk).min(len);
        let sstart = (c * src_chunk).min(src_len);
        let send = ((c + 1) * src_chunk).min(src_len);
        // SAFETY: output pieces go to exactly one worker each and lie inside `dst`; the input
        // pieces lie inside `src` and are only read.
        let (piece, spiece) = unsafe {
            (
                std::slice::from_raw_parts_mut((base as *mut T).add(start), end - start),
                std::slice::from_raw_parts((sbase as *const U).add(sstart), send - sstart),
            )
        };
        f(c, piece, spiece);
    });
}

/// Run `f(i)` for every `i` in `0..n`, split into contiguous runs across the pool.
///
/// The pool replacement for `(0..n).into_par_iter().for_each(..)`.
pub fn par_range<F: Fn(usize) + Sync>(n: usize, f: F) {
    if n <= 1 || threads_for(n) == 1 {
        (0..n).for_each(&f);
        return;
    }
    par_units(n, f);
}
