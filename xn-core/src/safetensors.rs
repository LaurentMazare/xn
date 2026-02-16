use crate::tensor::TypedTensor;
use crate::{Backend, Result, Shape, Tensor, WithDType};
use std::collections::HashMap;

fn load_tensor<T: WithDType, B: Backend>(
    data: &[u8],
    shape: Shape,
    device: &B,
) -> Result<Tensor<T, B>> {
    let vec = T::vec_from_le_bytes(data);
    Tensor::from_vec(vec, shape, device)
}

fn tensors_from_safetensors<B: Backend>(
    st: &safetensors::SafeTensors<'_>,
    device: &B,
) -> Result<HashMap<String, TypedTensor<B>>> {
    let mut map = HashMap::new();
    for (name, tensor) in st.iter() {
        let shape: Shape = tensor.shape().into();
        let data = tensor.data();
        let typed = match tensor.dtype() {
            safetensors::Dtype::F16 => {
                TypedTensor::F16(load_tensor::<half::f16, B>(data, shape, device)?)
            }
            safetensors::Dtype::BF16 => {
                TypedTensor::BF16(load_tensor::<half::bf16, B>(data, shape, device)?)
            }
            safetensors::Dtype::F32 => {
                TypedTensor::F32(load_tensor::<f32, B>(data, shape, device)?)
            }
            safetensors::Dtype::I64 => {
                TypedTensor::I64(load_tensor::<i64, B>(data, shape, device)?)
            }
            safetensors::Dtype::U8 => TypedTensor::U8(load_tensor::<u8, B>(data, shape, device)?),
            _ => continue,
        };
        map.insert(name.to_string(), typed);
    }
    Ok(map)
}

/// Load all tensors from a safetensors byte buffer.
/// Tensors with unhandled data types are silently discarded.
pub fn load_from_buffer<B: Backend>(
    buffer: &[u8],
    device: &B,
) -> Result<HashMap<String, TypedTensor<B>>> {
    let st = safetensors::SafeTensors::deserialize(buffer)?;
    tensors_from_safetensors(&st, device)
}

/// Load all tensors from a safetensors file.
/// Tensors with unhandled data types are silently discarded.
pub fn load_from_file<B: Backend>(
    path: impl AsRef<std::path::Path>,
    device: &B,
) -> Result<HashMap<String, TypedTensor<B>>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let st = safetensors::SafeTensors::deserialize(&mmap)?;
    tensors_from_safetensors(&st, device)
}
