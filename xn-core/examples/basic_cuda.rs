//! Basic CUDA example demonstrating tensor operations on GPU.
//!
//! Run with: cargo run --release --example basic_cuda

use xn::{Result, Tensor, cuda_backend::Device};

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

    Ok(())
}
