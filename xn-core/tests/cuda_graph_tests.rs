#![cfg(feature = "cuda")]

//! CUDA-graph capture tests. These exercise the record/capture/replay flow of
//! `Device::capture_record_begin` / `capture_begin` / `CaptureGraph::launch`
//! and the host-upload cache backing it, and only run on CUDA-enabled hosts.

use xn::{D, Result, Tensor, cuda_backend::Device};

fn get_device() -> Device {
    Device::new(0).expect("Failed to initialize CUDA device")
}

const B: usize = 2;
const R: usize = 3;
const C: usize = 4;

/// The workload under capture: reads `input` (B, R, C), writes `out_sum`
/// (B, C, 1) and `out_idx` (B, C) in place. It deliberately uses ops whose
/// CUDA kernels upload host data internally — the strided copy behind
/// `contiguous`, the shape descriptors of the reductions and of the broadcast
/// — plus an explicit in-body `from_vec` constant. Those uploads are what the
/// record/replay cache turns into capture-legal device-to-device copies.
fn body(
    input: &Tensor<f32, Device>,
    out_sum: &Tensor<f32, Device>,
    out_idx: &Tensor<i64, Device>,
    bias: &[f32],
    device: &Device,
) -> Result<()> {
    let x = input.transpose(1, 2)?.contiguous()?; // (B, C, R)
    let bias = Tensor::from_vec(bias.to_vec(), R, device)?;
    let x = x.broadcast_add(&bias)?;
    out_sum.copy_(&x.sum_keepdim([2])?)?;
    out_idx.copy_(&x.argmax(D::Minus1)?)?;
    Ok(())
}

/// Host-side reference for [`body`].
fn expected(input: &[f32], bias: &[f32]) -> (Vec<f32>, Vec<i64>) {
    let mut sums = vec![0f32; B * C];
    let mut idxs = vec![0i64; B * C];
    for b in 0..B {
        for c in 0..C {
            let mut sum = 0f32;
            let (mut best, mut best_r) = (f32::NEG_INFINITY, 0i64);
            for r in 0..R {
                let v = input[b * R * C + r * C + c] + bias[r];
                sum += v;
                if v > best {
                    best = v;
                    best_r = r as i64;
                }
            }
            sums[b * C + c] = sum;
            idxs[b * C + c] = best_r;
        }
    }
    (sums, idxs)
}

fn check(
    out_sum: &Tensor<f32, Device>,
    out_idx: &Tensor<i64, Device>,
    input: &[f32],
    bias: &[f32],
) -> Result<()> {
    let (sums, idxs) = expected(input, bias);
    assert_eq!(out_sum.to_vec()?, sums);
    assert_eq!(out_idx.to_vec()?, idxs);
    Ok(())
}

#[test]
fn test_capture_replay() -> Result<()> {
    let device = get_device();
    let bias = [0.25f32, -1.5, 3.0];
    let input_v: Vec<f32> = (0..B * R * C).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let input = Tensor::from_vec(input_v.clone(), (B, R, C), &device)?;
    let out_sum: Tensor<f32, Device> = Tensor::zeros((B, C, 1), &device)?;
    let out_idx: Tensor<i64, Device> = Tensor::zeros((B, C), &device)?;

    // Record pass: runs normally and populates the upload cache.
    device.capture_record_begin()?;
    let res = body(&input, &out_sum, &out_idx, &bias, &device);
    device.capture_record_end()?;
    res?;
    assert!(device.capture_cache_len() > 0, "record pass cached no uploads");
    check(&out_sum, &out_idx, &input_v, &bias)?;

    // Capture pass: records the graph without executing it.
    out_sum.fill_(0.0)?;
    out_idx.fill_(0)?;
    device.capture_begin()?;
    let res = body(&input, &out_sum, &out_idx, &bias, &device);
    if res.is_err() {
        device.capture_abort();
    }
    res?;
    let graph = device.capture_end()?;
    assert!(out_sum.to_vec()?.iter().all(|&x| x == 0.0), "capture must not execute");

    // Replay computes.
    graph.launch()?;
    check(&out_sum, &out_idx, &input_v, &bias)?;

    // Replays observe in-place refreshes of the input buffer.
    for step in 1..4 {
        let input_v: Vec<f32> =
            (0..B * R * C).map(|i| ((i * step) as f32) * 0.25 - step as f32).collect();
        input.copy_(&Tensor::from_vec(input_v.clone(), (B, R, C), &device)?)?;
        graph.launch()?;
        check(&out_sum, &out_idx, &input_v, &bias)?;
    }
    Ok(())
}

#[test]
fn test_capture_unrecorded_upload_fails() -> Result<()> {
    let device = get_device();
    let input_v: Vec<f32> = (0..B * R * C).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_v, (B, R, C), &device)?;
    let out_sum: Tensor<f32, Device> = Tensor::zeros((B, C, 1), &device)?;
    let out_idx: Tensor<i64, Device> = Tensor::zeros((B, C), &device)?;

    device.capture_record_begin()?;
    let res = body(&input, &out_sum, &out_idx, &[1.0, 2.0, 3.0], &device);
    device.capture_record_end()?;
    res?;

    // The bias differs from the record pass, so its `from_vec` content was
    // never recorded and the capture must fail rather than fall back to a
    // pageable host copy.
    device.capture_begin()?;
    let res = body(&input, &out_sum, &out_idx, &[4.0, 5.0, 6.0], &device);
    device.capture_abort();
    let err = res.expect_err("capturing an unrecorded upload should fail");
    assert!(err.to_string().contains("unrecorded content"), "unexpected error: {err}");

    // The device stays usable after the aborted capture.
    let t = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device)?;
    assert_eq!(t.add(&t)?.to_vec()?, [2.0, 4.0]);
    Ok(())
}

#[test]
fn test_capture_mode_bookkeeping() -> Result<()> {
    let device = get_device();
    device.capture_record_begin()?;
    assert!(device.capture_record_begin().is_err(), "nested record must fail");
    let _: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], 3, &device)?;
    device.capture_record_end()?;
    assert!(device.capture_record_end().is_err(), "unbalanced record end must fail");
    assert!(device.capture_cache_len() > 0);
    device.capture_cache_clear();
    assert_eq!(device.capture_cache_len(), 0);
    Ok(())
}
