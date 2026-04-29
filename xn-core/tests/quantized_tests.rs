use xn::quantized::{GgmlDType, QLinear};
use xn::{CpuDevice, ModuleT, Result, Tensor};

#[test]
fn qlinear_vs_linear_no_bias() -> Result<()> {
    let dev = CpuDevice;
    let in_features = 64;
    let out_features = 32;
    let batch = 4;

    // Create a random weight and input.
    let dummy: Tensor<f32, _> = Tensor::zeros((), &dev)?;
    let weight = dummy.randn((out_features, in_features), 0.0, 1.0)?;
    let xs = dummy.randn((batch, in_features), 0.0, 1.0)?;

    // Reference: standard linear (no bias).
    let linear = xn::nn::Linear::new(weight);
    let ref_out = linear.forward(&xs)?;

    // Quantized linear from the same linear layer.
    let qlinear = QLinear::from_linear(linear, GgmlDType::Q8_0)?;
    let q_out = ModuleT::forward(&qlinear, &xs)?;

    // Compare element-wise.
    let ref_v = ref_out.to_vec()?;
    let q_v = q_out.to_vec()?;
    assert_eq!(ref_v.len(), q_v.len());
    let max_err = ref_v.iter().zip(q_v.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // Q8_0 should be very accurate.
    assert!(max_err < 0.3, "max error too large: {max_err}");
    Ok(())
}

#[test]
fn qlinear_vs_linear_with_bias() -> Result<()> {
    let dev = CpuDevice;
    let in_features = 64;
    let out_features = 32;
    let batch = 4;

    let dummy: Tensor<f32, _> = Tensor::zeros((), &dev)?;
    let weight = dummy.randn((out_features, in_features), 0.0, 1.0)?;
    let bias = dummy.randn((out_features,), 0.0, 1.0)?;
    let xs = dummy.randn((batch, in_features), 0.0, 1.0)?;

    let linear = xn::nn::Linear::new(weight).with_bias(bias);
    let ref_out = linear.forward(&xs)?;

    let qlinear = QLinear::from_linear(linear, GgmlDType::Q8_0)?;
    let q_out = ModuleT::forward(&qlinear, &xs)?;

    let ref_v = ref_out.to_vec()?;
    let q_v = q_out.to_vec()?;
    assert_eq!(ref_v.len(), q_v.len());
    let max_err = ref_v.iter().zip(q_v.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_err < 0.3, "max error too large: {max_err}");
    Ok(())
}

#[test]
fn qlinear_3d_input() -> Result<()> {
    let dev = CpuDevice;
    let in_features = 64;
    let out_features = 32;
    let batch = 2;
    let seq_len = 3;

    let dummy: Tensor<f32, _> = Tensor::zeros((), &dev)?;
    let weight = dummy.randn((out_features, in_features), 0.0, 1.0)?;
    let xs = dummy.randn((batch, seq_len, in_features), 0.0, 1.0)?;

    let linear = xn::nn::Linear::new(weight);
    let ref_out = linear.forward(&xs)?;

    let qlinear = QLinear::from_linear(linear, GgmlDType::Q8_0)?;
    let q_out = ModuleT::forward(&qlinear, &xs)?;

    assert_eq!(q_out.dims(), &[batch, seq_len, out_features]);
    let ref_v = ref_out.to_vec()?;
    let q_v = q_out.to_vec()?;
    let max_err = ref_v.iter().zip(q_v.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_err < 0.3, "max error too large: {max_err}");
    Ok(())
}

// Exercises the SIMD `sgemm_q8_0_q8_0` kernels by computing a row-major
// reference with `BlockQ8_0::vec_dot` and comparing it against the
// column-major output of sgemm.
#[cfg(any(target_feature = "neon", target_feature = "avx", target_feature = "simd128"))]
#[allow(clippy::type_complexity)]
fn check_sgemm_q8_0_matches_vec_dot(
    sgemm: fn(
        usize,
        usize,
        usize,
        &[xn::quantized::k_quants::BlockQ8_0],
        usize,
        &[xn::quantized::k_quants::BlockQ8_0],
        usize,
        &mut [f32],
        usize,
        usize,
        usize,
    ) -> Result<()>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use xn::quantized::GgmlType;
    use xn::quantized::k_quants::{BlockQ8_0, QK8_0};

    assert!(k.is_multiple_of(QK8_0), "k={k} not a multiple of {QK8_0}");

    let dev = CpuDevice;
    let dummy: Tensor<f32, _> = Tensor::zeros((), &dev)?;
    let lhs = dummy.randn((m, k), 0.0, 1.0)?.to_vec()?;
    let rhs = dummy.randn((n, k), 0.0, 1.0)?.to_vec()?;

    let k_blocks = k / QK8_0;
    let mut lhs_q = vec![BlockQ8_0::zeros(); m * k_blocks];
    let mut rhs_q = vec![BlockQ8_0::zeros(); n * k_blocks];
    for i in 0..m {
        BlockQ8_0::from_float(
            &lhs[i * k..(i + 1) * k],
            &mut lhs_q[i * k_blocks..(i + 1) * k_blocks],
        )?;
    }
    for j in 0..n {
        BlockQ8_0::from_float(
            &rhs[j * k..(j + 1) * k],
            &mut rhs_q[j * k_blocks..(j + 1) * k_blocks],
        )?;
    }

    // Reference: per-cell vec_dot, row-major (`dst[i * n + j]`).
    let mut ref_dst = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let lhs_row = &lhs_q[i * k_blocks..(i + 1) * k_blocks];
            let rhs_row = &rhs_q[j * k_blocks..(j + 1) * k_blocks];
            ref_dst[i * n + j] = BlockQ8_0::vec_dot(k, lhs_row, rhs_row)?;
        }
    }

    // sgemm: column-major output with `ldc = m` (`c[m * j + i]`).
    let mut sg_dst = vec![0f32; m * n];
    sgemm(m, n, k_blocks, &lhs_q, k_blocks, &rhs_q, k_blocks, &mut sg_dst, m, 0, 1)?;

    let mut max_err = 0f32;
    for i in 0..m {
        for j in 0..n {
            let r = ref_dst[i * n + j];
            let s = sg_dst[m * j + i];
            max_err = max_err.max((r - s).abs());
        }
    }
    // Tolerance bounded by k·ε·|result|: simd128 sgemm uses
    // `f32x4_relaxed_madd` (one rounding) which can drift sub-ULP per
    // block from the reference's mul-then-add (two roundings). The
    // x86/aarch64 paths still match bit-for-bit at this tolerance.
    let tol = 1e-4;
    assert!(max_err <= tol, "({m}, {k}, {n}): max error {max_err} > tol {tol}");
    Ok(())
}

// A spread of shapes that exercises every tile case in the cpp-style
// dispatch (1×1 up to 4×2 / 2×4) plus odd remainders, so both the inner
// kernel and the recursive `mnpack` walk are covered.
#[cfg(any(target_feature = "neon", target_feature = "avx", target_feature = "simd128"))]
const SGEMM_TEST_SHAPES: &[(usize, usize, usize)] =
    &[(1, 32, 1), (3, 32, 5), (7, 64, 11), (5, 128, 8), (8, 32, 3), (16, 96, 17), (32, 64, 33)];

#[cfg(target_feature = "neon")]
#[test]
fn sgemm_q8_0_neon_matches_vec_dot() -> Result<()> {
    for &(m, k, n) in SGEMM_TEST_SHAPES {
        check_sgemm_q8_0_matches_vec_dot(xn::quantized::neon::sgemm_q8_0_q8_0, m, k, n)?;
    }
    Ok(())
}

#[cfg(target_feature = "avx")]
#[test]
fn sgemm_q8_0_avx_matches_vec_dot() -> Result<()> {
    for &(m, k, n) in SGEMM_TEST_SHAPES {
        check_sgemm_q8_0_matches_vec_dot(xn::quantized::avx::sgemm_q8_0_q8_0, m, k, n)?;
    }
    Ok(())
}

#[cfg(target_feature = "simd128")]
#[test]
fn sgemm_q8_0_simd128_matches_vec_dot() -> Result<()> {
    for &(m, k, n) in SGEMM_TEST_SHAPES {
        check_sgemm_q8_0_matches_vec_dot(xn::quantized::simd128::sgemm_q8_0_q8_0, m, k, n)?;
    }
    Ok(())
}
