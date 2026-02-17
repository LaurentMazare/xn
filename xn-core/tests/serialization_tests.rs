use std::collections::HashMap;
use xn::{CPU, CpuTensor, DType, Tensor, TypedTensor, safetensors};

#[test]
fn save_and_load_single_tensor() {
    let path = std::env::temp_dir().join("test_single.safetensors");
    let t: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), &CPU).unwrap();
    t.save_safetensors("t", &path).unwrap();

    let tensors = safetensors::load_from_file(&path, &CPU).unwrap();
    assert_eq!(tensors.len(), 1);
    let loaded = match tensors.get("t").unwrap() {
        TypedTensor::F32(t) => t,
        _ => panic!("expected F32"),
    };
    assert_eq!(loaded.dims(), &[2, 2]);
    assert_eq!(loaded.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn save_and_load_multiple_tensors() {
    let path = std::env::temp_dir().join("test_multi.safetensors");
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), &CPU).unwrap();
    let b: CpuTensor<i64> = Tensor::from_vec(vec![10, 20], (2,), &CPU).unwrap();

    let mut map: HashMap<String, TypedTensor<_>> = HashMap::new();
    map.insert("a".to_string(), TypedTensor::F32(a));
    map.insert("b".to_string(), TypedTensor::I64(b));
    safetensors::save(&map, &path).unwrap();

    let loaded = safetensors::load_from_file(&path, &CPU).unwrap();
    assert_eq!(loaded.len(), 2);

    let la = match loaded.get("a").unwrap() {
        TypedTensor::F32(t) => t,
        _ => panic!("expected F32"),
    };
    assert_eq!(la.dims(), &[1, 3]);
    assert_eq!(la.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

    let lb = match loaded.get("b").unwrap() {
        TypedTensor::I64(t) => t,
        _ => panic!("expected I64"),
    };
    assert_eq!(lb.dims(), &[2]);
    assert_eq!(lb.to_vec().unwrap(), vec![10, 20]);

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn save_with_str_keys() {
    let path = std::env::temp_dir().join("test_str_keys.safetensors");
    let t: CpuTensor<f32> = Tensor::from_vec(vec![5.0, 6.0], (2,), &CPU).unwrap();

    let map: HashMap<&str, TypedTensor<_>> = [("x", TypedTensor::F32(t))].into_iter().collect();
    safetensors::save(&map, &path).unwrap();

    let loaded = safetensors::load_from_file(&path, &CPU).unwrap();
    let lx = match loaded.get("x").unwrap() {
        TypedTensor::F32(t) => t,
        _ => panic!("expected F32"),
    };
    assert_eq!(lx.to_vec().unwrap(), vec![5.0, 6.0]);

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn roundtrip_preserves_dtypes() {
    let path = std::env::temp_dir().join("test_dtypes.safetensors");

    let f16_t: CpuTensor<half::f16> =
        Tensor::from_vec(vec![half::f16::from_f32(1.0)], (1,), &CPU).unwrap();
    let bf16_t: CpuTensor<half::bf16> =
        Tensor::from_vec(vec![half::bf16::from_f32(2.0)], (1,), &CPU).unwrap();
    let u8_t: CpuTensor<u8> = Tensor::from_vec(vec![42], (1,), &CPU).unwrap();

    let mut map: HashMap<String, TypedTensor<_>> = HashMap::new();
    map.insert("f16".to_string(), TypedTensor::F16(f16_t));
    map.insert("bf16".to_string(), TypedTensor::BF16(bf16_t));
    map.insert("u8".to_string(), TypedTensor::U8(u8_t));
    safetensors::save(&map, &path).unwrap();

    let loaded = safetensors::load_from_file(&path, &CPU).unwrap();

    assert_eq!(loaded.get("f16").unwrap().dtype(), DType::F16);
    assert_eq!(loaded.get("bf16").unwrap().dtype(), DType::BF16);
    assert_eq!(loaded.get("u8").unwrap().dtype(), DType::U8);

    let lf16 = match loaded.get("f16").unwrap() {
        TypedTensor::F16(t) => t,
        _ => panic!("expected F16"),
    };
    assert_eq!(lf16.to_vec().unwrap(), vec![half::f16::from_f32(1.0)]);

    let lu8 = match loaded.get("u8").unwrap() {
        TypedTensor::U8(t) => t,
        _ => panic!("expected U8"),
    };
    assert_eq!(lu8.to_vec().unwrap(), vec![42]);

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn save_load_buffer_roundtrip() {
    let path = std::env::temp_dir().join("test_buffer.safetensors");
    let t: CpuTensor<f32> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &CPU).unwrap();
    t.save_safetensors("t", &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let loaded = safetensors::load_from_buffer(&bytes, &CPU).unwrap();
    let lt = match loaded.get("t").unwrap() {
        TypedTensor::F32(t) => t,
        _ => panic!("expected F32"),
    };
    assert_eq!(lt.dims(), &[2, 3]);
    assert_eq!(lt.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    std::fs::remove_file(&path).unwrap();
}
