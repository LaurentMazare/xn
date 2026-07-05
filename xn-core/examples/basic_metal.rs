//! Basic Metal example demonstrating tensor operations on the GPU.
//!
//! Run with: cargo run --release --no-default-features --features metal --example basic_metal

use xn::{Backend, Result, Tensor, metal_backend::Device};

fn main() -> Result<()> {
    let device = Device::new(0)?;
    println!("Metal device initialized: {}", device.name());

    // A (2x3) @ B (3x2) = C (2x2)
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;
    let c = a.matmul(&b)?;
    println!("A @ B shape: {:?}", c.dims());
    println!("A @ B data:  {:?}", c.to_vec()?); // [22, 28, 49, 64]

    // Matmul benchmarks: a decode-shaped GEMV (m=1) and square GEMMs.
    for (m, n, k) in [(1usize, 4096usize, 4096usize), (512, 512, 512), (2048, 2048, 2048)] {
        let a: Tensor<f32, Device> = Tensor::from_vec(
            (0..m * k).map(|i| (i % 127) as f32 * 0.01).collect(),
            (m, k),
            &device,
        )?;
        let b: Tensor<f32, Device> = Tensor::from_vec(
            (0..k * n).map(|i| (i % 113) as f32 * 0.01).collect(),
            (k, n),
            &device,
        )?;
        let _ = a.matmul(&b)?;
        device.synchronize()?;
        let iters = 50;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _c = a.matmul(&b)?;
        }
        device.synchronize()?;
        let elapsed = start.elapsed();
        let flops = 2.0 * (m * n * k) as f64 * iters as f64;
        println!(
            "matmul {m}x{k} @ {k}x{n}: {:.1} us/iter, {:.1} GFLOP/s",
            elapsed.as_micros() as f64 / iters as f64,
            flops / elapsed.as_secs_f64() / 1e9,
        );
    }
    Ok(())
}
