// Wasm-only benchmark for the q8_0 simd128 sgemm kernel. Mirrors the
// `qmatmul`/`qmatmul-sgemm` paths from `cpu_benchmarks.rs` but compiles on
// `wasm32-wasip1` (no clap, no example-only deps) and dispatches directly to
// `xn::quantized::simd128::sgemm_q8_0_q8_0`. Run with:
//
//     cargo build --target wasm32-wasip1 --release --example wasm_benchmarks
//     wasmtime run target/wasm32-wasip1/release/examples/wasm_benchmarks.wasm \
//         qmatmul-sgemm
//
// Tasks: `qmatmul` (per-row vec_dot) | `qmatmul-sgemm` (blocked sgemm) |
// `matmul-f32` (per-row f32 dot, same dims as `qmatmul`).

#[cfg(target_arch = "wasm32")]
mod bench {
    use rayon::prelude::*;
    use xn::Result;
    use xn::quantized::GgmlType;
    use xn::quantized::k_quants::{BlockQ8_0, QK8_0};

    trait Benchmark {
        type PreProcessData;
        type RunResult;

        fn preprocess() -> Result<Self::PreProcessData>;
        fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

        const ITERS: usize;
    }

    // Shared dimensions for the q8_0 matmul benchmarks. `K` must be a multiple
    // of QK8_0 (32) since q8_0 packs 32 elements per block.
    const QM: usize = 125;
    const QK: usize = 4096;
    const QN: usize = 1024;

    // Number of distinct weight matrices to rotate through. With QN=1024,
    // QK=4096, q8_0, each weight is ~4.25 MiB; 24 of them (~102 MiB) is well
    // past typical L3 caches so each iteration pays the cost of streaming the
    // weight from RAM, matching real LLM inference behaviour.
    const Q_WEIGHTS: usize = 24;

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

    type FTensor = xn::Tensor<f32, xn::CpuDevice>;

    struct FData {
        lhs: FTensor,
        rhs: Vec<FTensor>,
        counter: std::sync::atomic::AtomicUsize,
    }

    fn f_preprocess() -> Result<FData> {
        let lhs = FTensor::zeros((QM, QK), &xn::CPU)?;
        let rhs = (0..Q_WEIGHTS)
            .map(|_| FTensor::zeros((QK, QN), &xn::CPU))
            .collect::<Result<Vec<_>>>()?;
        Ok(FData { lhs, rhs, counter: std::sync::atomic::AtomicUsize::new(0) })
    }

    fn f_pick_rhs(d: &FData) -> &FTensor {
        let idx = d.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % d.rhs.len();
        &d.rhs[idx]
    }

    struct MatMulF32;
    impl Benchmark for MatMulF32 {
        type PreProcessData = FData;
        type RunResult = FTensor;
        fn preprocess() -> Result<Self::PreProcessData> {
            f_preprocess()
        }

        fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
            d.lhs.matmul(f_pick_rhs(d))
        }

        const ITERS: usize = Q_WEIGHTS;
    }

    struct QMatMulSgemm;
    impl Benchmark for QMatMulSgemm {
        type PreProcessData = QData;
        type RunResult = Vec<f32>;
        fn preprocess() -> Result<Self::PreProcessData> {
            q_preprocess()
        }

        fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
            let k_blocks = QK / QK8_0;
            let rhs = q_pick_rhs(d);
            let mut dst = vec![0f32; QM * QN];
            xn::quantized::simd128::sgemm_q8_0_q8_0(
                QM, QN, k_blocks, &d.lhs, k_blocks, rhs, k_blocks, &mut dst, QM, 0, 1,
            )?;
            Ok(dst)
        }

        const ITERS: usize = Q_WEIGHTS;
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

    pub fn entry() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let mut task: Option<String> = None;
        let mut iters: Option<usize> = None;
        let mut i = 1;
        while i < args.len() {
            let a = &args[i];
            if a == "--iters" {
                i += 1;
                iters = Some(args[i].parse().expect("invalid --iters value"));
            } else if let Some(v) = a.strip_prefix("--iters=") {
                iters = Some(v.parse().expect("invalid --iters value"));
            } else if task.is_none() {
                task = Some(a.clone());
            }
            i += 1;
        }
        let task = task.unwrap_or_else(|| "qmatmul-sgemm".to_string());
        match task.as_str() {
            "qmatmul" => {
                for _ in 0..20 {
                    run::<QMatMul>(iters)?
                }
            }
            "qmatmul-sgemm" => {
                for _ in 0..20 {
                    run::<QMatMulSgemm>(iters)?
                }
            }
            "matmul-f32" => {
                for _ in 0..20 {
                    run::<MatMulF32>(iters)?
                }
            }
            other => panic!("unknown task: {other}"),
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn main() -> xn::Result<()> {
    bench::entry()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("wasm_benchmarks is only supported on wasm32 targets");
    std::process::exit(1);
}
