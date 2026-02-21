//! Basic CUDA example demonstrating tensor operations on GPU.
//!
//! Run with: cargo run --release --example basic_cuda

use xn::{Backend, Result, Tensor, cuda_backend::Device};

fn main() -> Result<()> {
    println!("Initializing CUDA device...");
    let device = Device::new(0)?;
    println!("CUDA device initialized successfully!");

    // Create two matrices for multiplication
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;

    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]  (3x2)
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;

    println!("A shape: {:?}", a.dims());
    println!("B shape: {:?}", b.dims());

    // C = A @ B = [[22, 28],
    //              [49, 64]]  (2x2)
    let c = a.matmul(&b)?;

    println!("C = A @ B");
    println!("C shape: {:?}", c.dims());
    println!("C data: {:?}", c.to_vec()?);
    // Benchmark matmul with bs=32, m=1, n=11264, k=2048
    run_mm_benchmark(32, 1, 11264, 2048, &device)?;
    run_mm_benchmark(1, 32, 11264, 2048, &device)?;
    Ok(())
}

fn run_mm_benchmark(bs: usize, m: usize, n: usize, k: usize, device: &Device) -> Result<()> {
    println!("\nBenchmarking matmul ({bs}x{m}x{k}) @ (1x{k}x{n})...");
    let a_data: Vec<half::bf16> =
        (0..bs * m * k).map(|i| half::bf16::from_f32((i % 127) as f32 * 0.01)).collect();
    let b_data: Vec<half::bf16> =
        (0..k * n).map(|i| half::bf16::from_f32((i % 113) as f32 * 0.01)).collect();
    let a: Tensor<half::bf16, Device> = Tensor::from_vec(a_data, (bs, m, k), device)?;
    let b: Tensor<half::bf16, Device> = Tensor::from_vec(b_data, (k, n), device)?;

    // Warmup
    let _warmup = a.matmul(&b)?;
    device.synchronize()?;

    let num_iters = 1000;
    let start = std::time::Instant::now();
    for _ in 0..num_iters {
        let _c = a.matmul(&b)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_iters as f64;
    let flops = 2.0 * bs as f64 * m as f64 * n as f64 * k as f64;
    let tflops = flops * num_iters as f64 / elapsed.as_secs_f64() / 1e12;
    println!("{num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)");

    Ok(())
}
