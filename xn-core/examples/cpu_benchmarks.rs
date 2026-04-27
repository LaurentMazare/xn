use clap::{Parser, Subcommand};
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

// Existing per-row mul-vec path: f32 lhs × q8_0 rhs (rhs already transposed).
// `k_quants::matmul` quantizes the lhs to q8_0 internally and parallelises
// across output columns with rayon.
struct QMatMul;
impl Benchmark for QMatMul {
    type PreProcessData = (Vec<f32>, Vec<BlockQ8_0>);
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = vec![0f32; QM * QK];
        let rhs = vec![BlockQ8_0::zeros(); QN * QK / QK8_0];
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let mut dst = vec![0f32; QM * QN];
        xn::quantized::k_quants::matmul((QM, QK, QN), &d.0, &d.1, &mut dst)?;
        Ok(dst)
    }

    const ITERS: usize = 5;
}

// New blocked sgemm path: q8_0 × q8_0 → f32 via `neon::sgemm_q8_0_q8_0`. The
// lhs is quantized inline each iteration so the comparison includes the same
// f32→q8_0 conversion that `QMatMul` performs internally. Single-threaded —
// the existing matmul uses rayon over output columns, so expect the gap to
// shrink on multi-core machines.
#[cfg(target_feature = "neon")]
struct QMatMulSgemm;
#[cfg(target_feature = "neon")]
impl Benchmark for QMatMulSgemm {
    type PreProcessData = (Vec<f32>, Vec<BlockQ8_0>);
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = vec![0f32; QM * QK];
        let rhs = vec![BlockQ8_0::zeros(); QN * QK / QK8_0];
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let k_blocks = QK / QK8_0;
        let mut lhs_q = vec![BlockQ8_0::zeros(); QM * k_blocks];
        for row in 0..QM {
            BlockQ8_0::from_float(
                &d.0[row * QK..(row + 1) * QK],
                &mut lhs_q[row * k_blocks..(row + 1) * k_blocks],
            )?;
        }
        // sgemm output is column-major with stride `ldc`; here we use ldc = QM
        // so the buffer is tightly packed.
        let mut dst = vec![0f32; QM * QN];
        xn::quantized::neon::sgemm_q8_0_q8_0(
            QM, QN, k_blocks, &lhs_q, k_blocks, &d.1, k_blocks, &mut dst, QM, 0, 1,
        )?;
        Ok(dst)
    }

    const ITERS: usize = 5;
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
    #[cfg(target_feature = "neon")]
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
        #[cfg(target_feature = "neon")]
        Task::QmatmulSgemm => {
            for _ in 0..20 {
                run::<QMatMulSgemm>(args.iters)?
            }
        }
    }
    Ok(())
}
