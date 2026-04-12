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
