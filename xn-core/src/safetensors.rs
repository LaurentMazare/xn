use crate::tensor::TypedTensor;
use crate::{Backend, DType, Result, Shape, Tensor, WithDType};
use std::borrow::Cow;
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

fn dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::F16 => safetensors::Dtype::F16,
        DType::BF16 => safetensors::Dtype::BF16,
        DType::F32 => safetensors::Dtype::F32,
        DType::I64 => safetensors::Dtype::I64,
        DType::U8 => safetensors::Dtype::U8,
    }
}

/// Reinterpret a `Vec<T>` as `Vec<u8>` without copying.
fn vec_to_bytes<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let byte_len = vs.len() * T::BYTE_SIZE;
    let byte_cap = vs.capacity() * T::BYTE_SIZE;
    let ptr = vs.as_mut_ptr() as *mut u8;
    std::mem::forget(vs);
    // SAFETY: Every T is at least as large as u8, so the pointer is valid for byte_len bytes.
    unsafe { Vec::from_raw_parts(ptr, byte_len, byte_cap) }
}

/// A pre-materialized view of tensor data for safetensors serialization.
struct SaveView {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

impl safetensors::tensor::View for SaveView {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn typed_tensor_to_save_view<B: Backend>(tensor: &TypedTensor<B>) -> Result<SaveView> {
    let dtype = dtype_to_safetensors(tensor.dtype());
    let shape = tensor.shape().dims().to_vec();
    let data = match tensor {
        TypedTensor::F16(t) => vec_to_bytes(t.to_vec()?),
        TypedTensor::BF16(t) => vec_to_bytes(t.to_vec()?),
        TypedTensor::F32(t) => vec_to_bytes(t.to_vec()?),
        TypedTensor::I64(t) => vec_to_bytes(t.to_vec()?),
        TypedTensor::U8(t) => t.to_vec()?,
    };
    Ok(SaveView { data, shape, dtype })
}

/// Save tensors to a safetensors file.
pub fn save<K: AsRef<str> + Ord + std::fmt::Display, B: Backend>(
    tensors: &HashMap<K, TypedTensor<B>>,
    path: impl AsRef<std::path::Path>,
) -> Result<()> {
    let views: Vec<(&K, SaveView)> = tensors
        .iter()
        .map(|(name, tensor)| Ok((name, typed_tensor_to_save_view(tensor)?)))
        .collect::<Result<_>>()?;
    Ok(safetensors::tensor::serialize_to_file(views, &None, path.as_ref())?)
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    /// Save this tensor to a safetensors file with the given name.
    pub fn save_safetensors(&self, name: &str, path: impl AsRef<std::path::Path>) -> Result<()> {
        let view = SaveView {
            data: vec_to_bytes(self.to_vec()?),
            shape: self.shape().dims().to_vec(),
            dtype: dtype_to_safetensors(T::DTYPE),
        };
        Ok(safetensors::tensor::serialize_to_file([(name, view)], &None, path.as_ref())?)
    }
}
