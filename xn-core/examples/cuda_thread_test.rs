//! Minimal reproduction of CUDA threading issue.
//! This test fails intermittently when run with many threads doing different operations.
//!
//! Run with: cargo run --release --example cuda_thread_test

use std::sync::{Arc, Barrier};
use std::thread;
use xn::{Result, Tensor, cuda_backend::Device};

fn main() -> Result<()> {
    // Use a barrier to ensure all 32 threads start simultaneously
    let barrier = Arc::new(Barrier::new(32));

    for run in 0..10 {
        let mut handles = vec![];

        for i in 0..32 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || -> std::result::Result<(), String> {
                b.wait();

                let device = Device::new(0).unwrap();

                // Different operations per thread - this is key to triggering the race
                match i % 6 {
                    0 => {
                        let a: Tensor<f32, Device> =
                            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device).unwrap();
                        let b: Tensor<f32, Device> =
                            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device).unwrap();
                        let c = a.add(&b).unwrap();
                        let r = c.to_vec().unwrap();
                        if r != vec![6.0, 8.0, 10.0, 12.0] {
                            return Err(format!("add: {:?}", r));
                        }
                    }
                    1 => {
                        let a: Tensor<f32, Device> =
                            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device).unwrap();
                        let b: Tensor<f32, Device> =
                            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device).unwrap();
                        let c = a.sub(&b).unwrap();
                        let r = c.to_vec().unwrap();
                        if r != vec![4.0, 4.0, 4.0, 4.0] {
                            return Err(format!("sub: {:?}", r));
                        }
                    }
                    2 => {
                        let a: Tensor<f32, Device> =
                            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device).unwrap();
                        let b: Tensor<f32, Device> =
                            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device).unwrap();
                        let c = a.mul(&b).unwrap();
                        let r = c.to_vec().unwrap();
                        if r != vec![5.0, 12.0, 21.0, 32.0] {
                            return Err(format!("mul: {:?}", r));
                        }
                    }
                    3 => {
                        let a: Tensor<f32, Device> = Tensor::from_vec(
                            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            vec![2, 3],
                            &device,
                        )
                        .unwrap();
                        let b: Tensor<f32, Device> = Tensor::from_vec(
                            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            vec![3, 2],
                            &device,
                        )
                        .unwrap();
                        let c = a.matmul(&b).unwrap();
                        let r = c.to_vec().unwrap();
                        if r != vec![22.0, 28.0, 49.0, 64.0] {
                            return Err(format!("matmul: {:?}", r));
                        }
                    }
                    4 => {
                        let x: Tensor<f32, Device> =
                            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &device).unwrap();
                        let y = x.softmax().unwrap();
                        let r = y.to_vec().unwrap();
                        let sum: f32 = r.iter().sum();
                        if (sum - 1.0).abs() > 1e-4 {
                            return Err(format!("softmax sum: {}", sum));
                        }
                    }
                    _ => {
                        let a: Tensor<f32, Device> = Tensor::from_vec(
                            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            vec![2, 3],
                            &device,
                        )
                        .unwrap();
                        let b = a.transpose(0, 1).unwrap();
                        let r = b.contiguous().unwrap().to_vec().unwrap();
                        if r != vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0] {
                            return Err(format!("transpose: {:?}", r));
                        }
                    }
                }
                Ok(())
            }));
        }

        let mut failures = 0;
        for (i, h) in handles.into_iter().enumerate() {
            if let Err(e) = h.join().unwrap() {
                eprintln!("[run {} thread {}] {}", run, i, e);
                failures += 1;
            }
        }

        if failures > 0 {
            eprintln!("Run {}: {} failures", run, failures);
        } else {
            eprintln!("Run {}: OK", run);
        }
    }

    Ok(())
}
