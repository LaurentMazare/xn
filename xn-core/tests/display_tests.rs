use std::sync::Mutex;
use xn::{CPU, CpuTensor, display};

// Tests that modify global print options must hold this lock
static PRINT_OPTS_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_debug_scalar() {
    let t: CpuTensor<f32> = CpuTensor::from_vec(vec![3.25], vec![1], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[3.25; F32]");
}

#[test]
fn test_debug_1d_small() {
    let t: CpuTensor<f32> = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[1, 2, 3; F32]");
}

#[test]
fn test_debug_1d_large() {
    let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
    let t: CpuTensor<f32> = CpuTensor::from_vec(data, vec![20], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[dims 20; F32]");
}

#[test]
fn test_debug_2d() {
    let t: CpuTensor<f32> =
        CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[dims 2, 2; F32]");
}

#[test]
fn test_debug_i64() {
    let t: CpuTensor<i64> = CpuTensor::from_vec(vec![1, 2, 3], vec![3], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[1, 2, 3; I64]");
}

#[test]
fn test_debug_u8() {
    let t: CpuTensor<u8> = CpuTensor::from_vec(vec![10, 20, 30], vec![3], &CPU).unwrap();
    let s = format!("{:?}", t);
    assert_eq!(s, "Tensor[10, 20, 30; U8]");
}

#[test]
fn test_display_1d_f32_integers() {
    let t: CpuTensor<f32> = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1., 2., 3.]
Tensor[[3], F32]"
    );
}

#[test]
fn test_display_1d_f32_floats() {
    let t: CpuTensor<f32> = CpuTensor::from_vec(vec![1.5, 2.5, 3.5], vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1.5000, 2.5000, 3.5000]
Tensor[[3], F32]"
    );
}

#[test]
fn test_display_2d_f32() {
    let t: CpuTensor<f32> =
        CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[[1., 2., 3.],
 [4., 5., 6.]]
Tensor[[2, 3], F32]"
    );
}

#[test]
fn test_display_i64() {
    let t: CpuTensor<i64> = CpuTensor::from_vec(vec![1, 2, 3, 4], vec![4], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1, 2, 3, 4]
Tensor[[4], I64]"
    );
}

#[test]
fn test_display_u8() {
    let t: CpuTensor<u8> = CpuTensor::from_vec(vec![10, 20, 30], vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[10, 20, 30]
Tensor[[3], U8]"
    );
}

#[test]
fn test_display_f16() {
    let data: Vec<half::f16> = vec![1.0, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t: CpuTensor<half::f16> = CpuTensor::from_vec(data, vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1., 2., 3.]
Tensor[[3], F16]"
    );
}

#[test]
fn test_display_bf16() {
    let data: Vec<half::bf16> = vec![1.0, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t: CpuTensor<half::bf16> = CpuTensor::from_vec(data, vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1., 2., 3.]
Tensor[[3], BF16]"
    );
}

#[test]
fn test_display_summarize_large_1d() {
    let _lock = PRINT_OPTS_LOCK.lock().unwrap();
    display::set_threshold(10);
    display::set_edge_items(3);
    let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
    let t: CpuTensor<f32> = CpuTensor::from_vec(data, vec![100], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[ 0.,  1.,  2., ..., 97., 98., 99.]
Tensor[[100], F32]"
    );
    display::set_print_options_default();
}

#[test]
fn test_display_summarize_large_2d() {
    let _lock = PRINT_OPTS_LOCK.lock().unwrap();
    display::set_threshold(10);
    display::set_edge_items(2);
    let data: Vec<f32> = (0..36).map(|x| x as f32).collect();
    let t: CpuTensor<f32> = CpuTensor::from_vec(data, vec![6, 6], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[[ 0.,  1., ...,  4.,  5.],
 [ 6.,  7., ..., 10., 11.],
 ...
 [24., 25., ..., 28., 29.],
 [30., 31., ..., 34., 35.]]
Tensor[[6, 6], F32]"
    );
    display::set_print_options_default();
}

#[test]
fn test_sci_mode() {
    let _lock = PRINT_OPTS_LOCK.lock().unwrap();
    display::set_sci_mode(Some(true));
    let t: CpuTensor<f32> = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[1.0000e0, 2.0000e0, 3.0000e0]
Tensor[[3], F32]"
    );
    display::set_print_options_default();
}

#[test]
fn test_display_3d() {
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let t: CpuTensor<f32> = CpuTensor::from_vec(data, vec![2, 2, 2], &CPU).unwrap();
    let s = format!("{}", t);
    assert_eq!(
        s,
        "\
[[[1., 2.],
  [3., 4.]],
 [[5., 6.],
  [7., 8.]]]
Tensor[[2, 2, 2], F32]"
    );
}

#[test]
fn test_print_options_short() {
    let _lock = PRINT_OPTS_LOCK.lock().unwrap();
    display::set_print_options_short();
    let po = display::print_options().lock().unwrap();
    assert_eq!(po.precision, 2);
    assert_eq!(po.edge_items, 2);
    drop(po);
    display::set_print_options_default();
}

#[test]
fn test_print_options_full() {
    let _lock = PRINT_OPTS_LOCK.lock().unwrap();
    display::set_print_options_full();
    let po = display::print_options().lock().unwrap();
    assert_eq!(po.threshold, usize::MAX);
    drop(po);
    display::set_print_options_default();
}
