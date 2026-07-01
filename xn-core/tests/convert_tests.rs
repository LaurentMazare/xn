use half::{bf16, f16};
use xn::{CPU, CpuTensor, Tensor};

// =============================================================================
// Vec/slice -> Tensor
// =============================================================================

#[test]
fn vec_to_tensor() {
    let t: CpuTensor<f32> = vec![1.0f32, 2.0, 3.0].try_into().unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn slice_to_tensor() {
    let t: CpuTensor<i64> = [4i64, 5, 6, 7].as_slice().try_into().unwrap();
    assert_eq!(t.dims(), &[4]);
    assert_eq!(t.to_vec().unwrap(), vec![4, 5, 6, 7]);
}

#[test]
fn empty_vec_to_tensor() {
    let t: CpuTensor<f32> = Vec::<f32>::new().try_into().unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.to_vec().unwrap(), Vec::<f32>::new());
}

// =============================================================================
// Tensor -> Vec (1d/2d/3d), both by-ref and owned
// =============================================================================

#[test]
fn tensor_to_vec1() {
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], 3, &CPU).unwrap();
    let by_ref: Vec<f32> = (&t).try_into().unwrap();
    assert_eq!(by_ref, vec![1.0, 2.0, 3.0]);
    let owned: Vec<f32> = t.try_into().unwrap();
    assert_eq!(owned, vec![1.0, 2.0, 3.0]);
}

#[test]
fn tensor_to_vec2() {
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &CPU).unwrap();
    let by_ref: Vec<Vec<f32>> = (&t).try_into().unwrap();
    assert_eq!(by_ref, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let owned: Vec<Vec<f32>> = t.try_into().unwrap();
    assert_eq!(owned, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

#[test]
fn tensor_to_vec3() {
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let t: CpuTensor<f32> = Tensor::from_vec(data, (2, 2, 2), &CPU).unwrap();
    let by_ref: Vec<Vec<Vec<f32>>> = (&t).try_into().unwrap();
    assert_eq!(
        by_ref,
        vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]], vec![vec![4.0, 5.0], vec![6.0, 7.0]],]
    );
}

#[test]
fn to_vec_rank_mismatch_errors() {
    // A 2x2 tensor cannot be converted to a 1d Vec.
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &CPU).unwrap();
    let r: Result<Vec<f32>, _> = (&t).try_into();
    assert!(r.is_err());
}

// =============================================================================
// Round-trips through every supported dtype
// =============================================================================

#[test]
fn roundtrip_all_dtypes() {
    let t_f32: CpuTensor<f32> = vec![1.5f32, -2.5, 3.0].try_into().unwrap();
    assert_eq!(Vec::<f32>::try_from(&t_f32).unwrap(), vec![1.5, -2.5, 3.0]);

    let t_f16: CpuTensor<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)].try_into().unwrap();
    assert_eq!(Vec::<f16>::try_from(&t_f16).unwrap(), vec![f16::from_f32(1.0), f16::from_f32(2.0)]);

    let t_bf16: CpuTensor<bf16> =
        vec![bf16::from_f32(1.0), bf16::from_f32(2.0)].try_into().unwrap();
    assert_eq!(
        Vec::<bf16>::try_from(&t_bf16).unwrap(),
        vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]
    );

    let t_i64: CpuTensor<i64> = vec![-1i64, 0, 42].try_into().unwrap();
    assert_eq!(Vec::<i64>::try_from(&t_i64).unwrap(), vec![-1, 0, 42]);

    let t_u8: CpuTensor<u8> = vec![0u8, 128, 255].try_into().unwrap();
    assert_eq!(Vec::<u8>::try_from(&t_u8).unwrap(), vec![0, 128, 255]);
}

// =============================================================================
// Scalar conversions (rank-0 tensors)
// =============================================================================

#[test]
fn scalar_to_tensor_and_back() {
    let t: CpuTensor<f32> = 42.0f32.try_into().unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.elem_count(), 1);
    let by_ref: f32 = (&t).try_into().unwrap();
    assert_eq!(by_ref, 42.0);
    let owned: f32 = t.try_into().unwrap();
    assert_eq!(owned, 42.0);
}

#[test]
fn scalar_all_dtypes() {
    let t_f16: CpuTensor<f16> = f16::from_f32(1.5).try_into().unwrap();
    assert_eq!(f16::try_from(&t_f16).unwrap(), f16::from_f32(1.5));

    let t_bf16: CpuTensor<bf16> = bf16::from_f32(2.5).try_into().unwrap();
    assert_eq!(bf16::try_from(&t_bf16).unwrap(), bf16::from_f32(2.5));

    let t_i64: CpuTensor<i64> = 123i64.try_into().unwrap();
    assert_eq!(i64::try_from(&t_i64).unwrap(), 123);

    let t_u8: CpuTensor<u8> = 7u8.try_into().unwrap();
    assert_eq!(u8::try_from(&t_u8).unwrap(), 7);
}

#[test]
fn to_scalar_on_non_scalar_errors() {
    let t: CpuTensor<f32> = vec![1.0f32, 2.0].try_into().unwrap();
    let r: Result<f32, _> = (&t).try_into();
    assert!(r.is_err());
}

// =============================================================================
// write_bytes
// =============================================================================

#[test]
fn write_bytes_f32() {
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &CPU).unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    let mut expected = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf, expected);
}

#[test]
fn write_bytes_i64() {
    let t: CpuTensor<i64> = Tensor::from_vec(vec![-1i64, 256, 1000], 3, &CPU).unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    let mut expected = Vec::new();
    for v in [-1i64, 256, 1000] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf, expected);
}

#[test]
fn write_bytes_u8() {
    let t: CpuTensor<u8> = Tensor::from_vec(vec![0u8, 128, 255], 3, &CPU).unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    assert_eq!(buf, vec![0u8, 128, 255]);
}

#[test]
fn write_bytes_f16() {
    let vals = [f16::from_f32(1.0), f16::from_f32(-2.0)];
    let t: CpuTensor<f16> = vals.to_vec().try_into().unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    let mut expected = Vec::new();
    for v in vals {
        expected.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    assert_eq!(buf, expected);
}

#[test]
fn write_bytes_bf16() {
    let vals = [bf16::from_f32(1.0), bf16::from_f32(-2.0)];
    let t: CpuTensor<bf16> = vals.to_vec().try_into().unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    let mut expected = Vec::new();
    for v in vals {
        expected.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    assert_eq!(buf, expected);
}

#[test]
fn write_bytes_flattens_multi_dim() {
    // write_bytes should serialize in row-major order regardless of rank.
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &CPU).unwrap();
    let mut buf = Vec::new();
    t.write_bytes(&mut buf).unwrap();
    assert_eq!(buf.len(), 4 * std::mem::size_of::<f32>());
    assert_eq!(&buf[0..4], &1.0f32.to_le_bytes());
}
