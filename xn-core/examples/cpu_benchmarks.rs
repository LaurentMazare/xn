use clap::{Parser, Subcommand};
use rayon::prelude::*;
use xn::Result;
use xn::quantized::GgmlType;
use xn::quantized::k_quants::{BlockQ8_0, QK8_0};

type Tensor = xn::Tensor<f32, xn::CpuDevice>;

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

struct MatMul;
impl Benchmark for MatMul {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::zeros((125, 4096), &xn::CPU)?;
        let rhs = Tensor::zeros((4096, 1024), &xn::CPU)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 5;
}

// Shared dimensions for the q8_0 matmul benchmarks. `K` must be a multiple of
// QK8_0 (32) since q8_0 packs 32 elements per block.
const QM: usize = 125;
const QK: usize = 4096;
const QN: usize = 1024;

// Number of distinct weight matrices to rotate through. With QN=1024,
// QK=4096, q8_0, each weight is ~4.25 MiB; 24 of them (~102 MiB) is well
// past typical L3 caches so each iteration pays the cost of streaming the
// weight from RAM, matching real LLM inference behaviour.
const Q_WEIGHTS: usize = 24;

// Pre-quantized q8_0 lhs (m × k_blocks) and `Q_WEIGHTS` q8_0 rhs matrices
// (n × k_blocks each). Both `QMatMul` and `QMatMulSgemm` reuse this so the
// f32→q8_0 conversion is not timed. The atomic counter cycles through the
// weight matrices to defeat L3 caching.
struct QData {
    lhs: Vec<BlockQ8_0>,
    rhs: Vec<Vec<BlockQ8_0>>,
    counter: std::sync::atomic::AtomicUsize,
}

fn q_preprocess() -> Result<QData> {
    let k_blocks = QK / QK8_0;
    let lhs = vec![BlockQ8_0::zeros(); QM * k_blocks];
    let rhs = (0..Q_WEIGHTS).map(|_| vec![BlockQ8_0::zeros(); QN * k_blocks]).collect();
    Ok(QData { lhs, rhs, counter: std::sync::atomic::AtomicUsize::new(0) })
}

fn q_pick_rhs(d: &QData) -> &[BlockQ8_0] {
    let idx = d.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % d.rhs.len();
    &d.rhs[idx]
}

// Existing per-row mul-vec path: for each output row, parallelise over
// columns and call `vec_dot` (matches the inner loop of `k_quants::matmul`,
// minus the lhs quantization step that we pre-do in `q_preprocess`).
struct QMatMul;
impl Benchmark for QMatMul {
    type PreProcessData = QData;
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        q_preprocess()
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let k_blocks = QK / QK8_0;
        let rhs = q_pick_rhs(d);
        let mut dst = vec![0f32; QM * QN];
        for row_idx in 0..QM {
            let lhs_row = &d.lhs[row_idx * k_blocks..(row_idx + 1) * k_blocks];
            let dst_row = &mut dst[row_idx * QN..(row_idx + 1) * QN];
            let result: Result<Vec<_>> = dst_row
                .into_par_iter()
                .enumerate()
                .with_min_len(128)
                .with_max_len(512)
                .map(|(col_idx, dst)| {
                    let rhs_col = &rhs[col_idx * k_blocks..(col_idx + 1) * k_blocks];
                    BlockQ8_0::vec_dot(QK, rhs_col, lhs_row).map(|value| *dst = value)
                })
                .collect();
            result?;
        }
        Ok(dst)
    }

    const ITERS: usize = Q_WEIGHTS;
}

// New blocked sgemm path: q8_0 × q8_0 → f32. Dispatches at compile time to
// `neon::sgemm_q8_0_q8_0` on aarch64 and `avx::sgemm_q8_0_q8_0` on x86. Both
// kernels are single-threaded — the existing matmul uses rayon over output
// columns, so expect the gap to shrink on multi-core machines.
#[cfg(any(target_feature = "neon", target_feature = "avx"))]
struct QMatMulSgemm;
#[cfg(any(target_feature = "neon", target_feature = "avx"))]
impl Benchmark for QMatMulSgemm {
    type PreProcessData = QData;
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        q_preprocess()
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let k_blocks = QK / QK8_0;
        let rhs = q_pick_rhs(d);
        // sgemm output is column-major with stride `ldc`; here we use ldc = QM
        // so the buffer is tightly packed.
        let mut dst = vec![0f32; QM * QN];
        #[cfg(target_feature = "neon")]
        xn::quantized::neon::sgemm_q8_0_q8_0(
            QM, QN, k_blocks, &d.lhs, k_blocks, rhs, k_blocks, &mut dst, QM, 0, 1,
        )?;
        #[cfg(all(target_feature = "avx", not(target_feature = "neon")))]
        xn::quantized::avx::sgemm_q8_0_q8_0(
            QM, QN, k_blocks, &d.lhs, k_blocks, rhs, k_blocks, &mut dst, QM, 0, 1,
        )?;
        Ok(dst)
    }

    const ITERS: usize = Q_WEIGHTS;
}

struct MatVec;
impl Benchmark for MatVec {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::zeros((1024 * 4, 1024 * 4), &xn::CPU)?;
        let rhs = Tensor::zeros((1024 * 4, 1), &xn::CPU)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 100;
}

fn run<B: Benchmark>(iters: Option<usize>) -> Result<()> {
    use std::hint::black_box;

    let iters = iters.unwrap_or(B::ITERS);
    let d = B::preprocess()?;
    let start = std::time::Instant::now();
    for _iter in 0..iters {
        let _res = black_box(B::run_one(black_box(&d))?);
    }
    println!("{:?}", start.elapsed() / iters as u32);
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Matmul,
    Matvec,
    Qmatmul,
    #[cfg(any(target_feature = "neon", target_feature = "avx"))]
    QmatmulSgemm,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,

    #[arg(long)]
    iters: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Matmul => {
            for _ in 0..20 {
                run::<MatMul>(args.iters)?
            }
        }
        Task::Matvec => {
            for _ in 0..20 {
                run::<MatVec>(args.iters)?
            }
        }
        Task::Qmatmul => {
            for _ in 0..20 {
                run::<QMatMul>(args.iters)?
            }
        }
        #[cfg(any(target_feature = "neon", target_feature = "avx"))]
        Task::QmatmulSgemm => {
            for _ in 0..20 {
                run::<QMatMulSgemm>(args.iters)?
            }
        }
    }
    Ok(())
}
