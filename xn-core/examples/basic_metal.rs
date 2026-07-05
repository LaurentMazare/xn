//! Basic Metal example demonstrating tensor operations on the GPU.
//!
//! Run with: cargo run --release --no-default-features --features metal --example basic_metal

use xn::{Backend, Result, Tensor, metal_backend::Device};

/// GEMV decode-path benchmark: y = x @ W^T with a row-major W (n, k), the
/// bandwidth-bound matmul_t shape used per token by LLM decoding.
fn bench_gemv<T: xn::WithDTypeF>(name: &str, device: &Device, n: usize, k: usize) -> Result<()> {
    let x: Tensor<T, Device> = Tensor::from_vec(
        (0..k).map(|i| T::from_f32((i % 127) as f32 * 0.01)).collect(),
        (1, k),
        device,
    )?;
    let w: Tensor<T, Device> = Tensor::from_vec(
        (0..n * k).map(|i| T::from_f32((i % 113) as f32 * 0.01)).collect(),
        (n, k),
        device,
    )?;
    let _ = x.matmul_t(&w)?;
    device.synchronize()?;
    let iters = 200;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _y = x.matmul_t(&w)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();
    let bytes = (n * k * T::BYTE_SIZE * iters) as f64;
    println!(
        "gemv {name} 1x{k} @ ({n}x{k})^T: {:.1} us/iter, {:.1} GB/s",
        elapsed.as_micros() as f64 / iters as f64,
        bytes / elapsed.as_secs_f64() / 1e9,
    );
    Ok(())
}

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

    // Decode-shaped GEMVs (TinyLlama's projection shapes plus a square case).
    for (n, k) in [(4096usize, 4096usize), (5632, 2048), (2048, 2048), (2048, 5632), (32000, 2048)]
    {
        bench_gemv::<f32>("f32", &device, n, k)?;
        bench_gemv::<half::f16>("f16", &device, n, k)?;
    }
    Ok(())
}
