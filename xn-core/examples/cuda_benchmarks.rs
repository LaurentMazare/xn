//! Benchmark: BF16 matmul vs FP8 matmul (per-tensor and per-token scaling),
//! including mixed variants (quantize-on-the-fly + FP8 matmul).
//!
//! Computes C = A × B^T where A is [M, K] and B is [N, K].
//!
//! Run with: cargo run --release --features cuda --example cuda_benchmarks

use std::time::Instant;
use xn::cuda_backend::quantization::Fp8Tensor;
use xn::{Backend, Result, Tensor, cuda_backend::Device};

const M: usize = 32;
const K: usize = 2048;
const N: usize = 11264;
const WARMUP: usize = 50;
const ITERS: usize = 2000;

fn random_bf16_vec(n: usize, seed: u64) -> Vec<half::bf16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            half::bf16::from_f32((s & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0)
        })
        .collect()
}

fn main() -> Result<()> {
    let device = Device::new(0)?;

    let (cc_major, cc_minor) = device.compute_cap()?;
    println!("GPU compute capability: {cc_major}.{cc_minor}");
    println!("Matrix dimensions: A=[{M}, {K}], B=[{N}, {K}] -> C=[{M}, {N}]");
    println!("Warmup: {WARMUP}  |  Iterations: {ITERS}");

    let a_bf16: Tensor<half::bf16, Device> =
        Tensor::from_vec(random_bf16_vec(M * K, 42), (M, K), &device)?;
    let b_bf16: Tensor<half::bf16, Device> =
        Tensor::from_vec(random_bf16_vec(N * K, 123), (N, K), &device)?;

    // =====================================================================
    // 1. BF16 × BF16 -> BF16  (using Tensor::matmul_t)
    // =====================================================================
    println!("\n=== BF16 x BF16 -> BF16 ===");

    for _ in 0..WARMUP {
        let _c = a_bf16.matmul_t(&b_bf16)?;
    }
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _c = a_bf16.matmul_t(&b_bf16)?;
    }
    device.synchronize()?;
    let bf16_elapsed = t0.elapsed();
    let bf16_us = bf16_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let bf16_tflops = flops * ITERS as f64 / bf16_elapsed.as_secs_f64() / 1e12;
    println!(
        "  {ITERS} iters in {:.3} ms  |  {bf16_us:.1} us/iter  |  {bf16_tflops:.2} TFLOP/s",
        bf16_elapsed.as_secs_f64() * 1e3,
    );

    // Keep a reference result for accuracy comparison.
    let c_bf16 = a_bf16.matmul_t(&b_bf16)?;
    device.synchronize()?;

    // =====================================================================
    // 2. FP8 × FP8 -> BF16  (both pre-quantized)
    // =====================================================================
    println!("\n=== FP8 x FP8 -> BF16 (both pre-quantized) ===");

    let a_fp8 = Fp8Tensor::quantize(&a_bf16)?;
    let b_fp8 = Fp8Tensor::quantize(&b_bf16)?;

    for _ in 0..WARMUP {
        let _c = a_fp8.matmul_t(&b_fp8)?;
    }
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _c = a_fp8.matmul_t(&b_fp8)?;
    }
    device.synchronize()?;
    let fp8_elapsed = t0.elapsed();
    let fp8_us = fp8_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    let fp8_tflops = flops * ITERS as f64 / fp8_elapsed.as_secs_f64() / 1e12;
    println!(
        "  {ITERS} iters in {:.3} ms  |  {fp8_us:.1} us/iter  |  {fp8_tflops:.2} TFLOP/s",
        fp8_elapsed.as_secs_f64() * 1e3,
    );

    let c_fp8 = a_fp8.matmul_t(&b_fp8)?;
    device.synchronize()?;

    // =====================================================================
    // 3. Mixed: quantize A on-the-fly + FP8 matmul with pre-quantized B
    // =====================================================================
    println!("\n=== Mixed: quantize(A) on-the-fly + FP8 matmul (B pre-quantized) ===");

    for _ in 0..WARMUP {
        let a_q = Fp8Tensor::quantize(&a_bf16)?;
        let _c = a_q.matmul_t(&b_fp8)?;
    }
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let a_q = Fp8Tensor::quantize(&a_bf16)?;
        let _c = a_q.matmul_t(&b_fp8)?;
    }
    device.synchronize()?;
    let mixed_elapsed = t0.elapsed();
    let mixed_us = mixed_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    let mixed_tflops = flops * ITERS as f64 / mixed_elapsed.as_secs_f64() / 1e12;
    println!(
        "  {ITERS} iters in {:.3} ms  |  {mixed_us:.1} us/iter  |  {mixed_tflops:.2} TFLOP/s",
        mixed_elapsed.as_secs_f64() * 1e3,
    );

    // =====================================================================
    // 4 & 5. Per-token scaling variants (require compute capability >= 9.0)
    // =====================================================================
    let per_token = if cc_major >= 9 {
        // 4. FP8 × FP8 -> BF16 with per-token scaling (both pre-quantized)
        println!("\n=== FP8 x FP8 -> BF16 per-token (both pre-quantized) ===");

        let a_fp8_pt = Fp8Tensor::quantize_per_token(&a_bf16)?;
        let b_fp8_pt = Fp8Tensor::quantize_per_token(&b_bf16)?;

        for _ in 0..WARMUP {
            let _c = a_fp8_pt.matmul_t(&b_fp8_pt)?;
        }
        device.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _c = a_fp8_pt.matmul_t(&b_fp8_pt)?;
        }
        device.synchronize()?;
        let fp8_pt_elapsed = t0.elapsed();
        let fp8_pt_us = fp8_pt_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let fp8_pt_tflops = flops * ITERS as f64 / fp8_pt_elapsed.as_secs_f64() / 1e12;
        println!(
            "  {ITERS} iters in {:.3} ms  |  {fp8_pt_us:.1} us/iter  |  {fp8_pt_tflops:.2} TFLOP/s",
            fp8_pt_elapsed.as_secs_f64() * 1e3,
        );

        let c_fp8_pt = a_fp8_pt.matmul_t(&b_fp8_pt)?;
        device.synchronize()?;

        // 5. Per-token mixed: quantize_per_token(A) on-the-fly + FP8 matmul
        println!(
            "\n=== Mixed per-token: quantize_per_token(A) on-the-fly + FP8 matmul (B pre-quantized per-token) ==="
        );

        for _ in 0..WARMUP {
            let a_q = Fp8Tensor::quantize_per_token(&a_bf16)?;
            let _c = a_q.matmul_t(&b_fp8_pt)?;
        }
        device.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let a_q = Fp8Tensor::quantize_per_token(&a_bf16)?;
            let _c = a_q.matmul_t(&b_fp8_pt)?;
        }
        device.synchronize()?;
        let mixed_pt_elapsed = t0.elapsed();
        let mixed_pt_us = mixed_pt_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let mixed_pt_tflops = flops * ITERS as f64 / mixed_pt_elapsed.as_secs_f64() / 1e12;
        println!(
            "  {ITERS} iters in {:.3} ms  |  {mixed_pt_us:.1} us/iter  |  {mixed_pt_tflops:.2} TFLOP/s",
            mixed_pt_elapsed.as_secs_f64() * 1e3,
        );

        Some((fp8_pt_us, mixed_pt_us, c_fp8_pt))
    } else {
        println!(
            "\n=== Skipping per-token FP8 matmul benchmarks (requires compute capability >= 9.0, got {cc_major}.{cc_minor}) ==="
        );
        None
    };

    // =====================================================================
    // Comparison
    // =====================================================================
    println!("\n=== Comparison ===");
    println!("  FP8 vs BF16 speedup:             {:.2}x", bf16_us / fp8_us);
    println!("  Mixed vs BF16 speedup:           {:.2}x", bf16_us / mixed_us);
    if let Some((fp8_pt_us, mixed_pt_us, _)) = per_token.as_ref() {
        println!("  FP8 per-token vs BF16 speedup:   {:.2}x", bf16_us / fp8_pt_us);
        println!("  Mixed per-token vs BF16 speedup: {:.2}x", bf16_us / mixed_pt_us);
        println!("  FP8 per-token vs FP8 per-tensor: {:.2}x", fp8_us / fp8_pt_us);
    }

    // Accuracy: compare FP8 results against BF16 reference.
    let c_bf16_vec = c_bf16.to_vec()?;
    let c_fp8_vec = c_fp8.to_vec()?;

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    for (&a, &b) in c_bf16_vec.iter().zip(c_fp8_vec.iter()) {
        let d = (a.to_f32() - b.to_f32()).abs();
        max_diff = max_diff.max(d);
        sum_diff += d as f64;
    }
    println!("\n  FP8 per-tensor vs BF16 accuracy:");
    println!("    Max abs diff:  {max_diff:.4}");
    println!("    Mean abs diff: {:.4}", sum_diff / (M * N) as f64);

    if let Some((_, _, c_fp8_pt)) = per_token.as_ref() {
        let c_fp8_pt_vec = c_fp8_pt.to_vec()?;
        let mut max_diff: f32 = 0.0;
        let mut sum_diff: f64 = 0.0;
        for (&a, &b) in c_bf16_vec.iter().zip(c_fp8_pt_vec.iter()) {
            let d = (a.to_f32() - b.to_f32()).abs();
            max_diff = max_diff.max(d);
            sum_diff += d as f64;
        }
        println!("\n  FP8 per-token vs BF16 accuracy:");
        println!("    Max abs diff:  {max_diff:.4}");
        println!("    Mean abs diff: {:.4}", sum_diff / (M * N) as f64);
    }

    Ok(())
}
