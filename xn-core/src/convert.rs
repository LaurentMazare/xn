//! Implement conversion traits for tensors
use crate::{Backend, CPU, CpuTensor, DType, Error, Tensor, WithDType};
use half::{bf16, f16};
use std::convert::TryFrom;

impl<T: WithDType, B: Backend> TryFrom<&Tensor<T, B>> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: &Tensor<T, B>) -> Result<Self, Self::Error> {
        tensor.to_vec1()
    }
}

impl<T: WithDType, B: Backend> TryFrom<&Tensor<T, B>> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: &Tensor<T, B>) -> Result<Self, Self::Error> {
        tensor.to_vec2()
    }
}

impl<T: WithDType, B: Backend> TryFrom<&Tensor<T, B>> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: &Tensor<T, B>) -> Result<Self, Self::Error> {
        tensor.to_vec3()
    }
}

impl<T: WithDType, B: Backend> TryFrom<Tensor<T, B>> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: Tensor<T, B>) -> Result<Self, Self::Error> {
        Vec::<T>::try_from(&tensor)
    }
}

impl<T: WithDType, B: Backend> TryFrom<Tensor<T, B>> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: Tensor<T, B>) -> Result<Self, Self::Error> {
        Vec::<Vec<T>>::try_from(&tensor)
    }
}

impl<T: WithDType, B: Backend> TryFrom<Tensor<T, B>> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: Tensor<T, B>) -> Result<Self, Self::Error> {
        Vec::<Vec<Vec<T>>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<&[T]> for CpuTensor<T> {
    type Error = Error;
    fn try_from(v: &[T]) -> Result<Self, Self::Error> {
        Tensor::from_vec(v.to_vec(), v.len(), &CPU)
    }
}

impl<T: WithDType> TryFrom<Vec<T>> for CpuTensor<T> {
    type Error = Error;
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        let len = v.len();
        Tensor::from_vec(v, len, &CPU)
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl<B: Backend> TryFrom<&Tensor<$typ, B>> for $typ {
            type Error = Error;

            fn try_from(tensor: &Tensor<$typ, B>) -> Result<Self, Self::Error> {
                tensor.to_scalar()
            }
        }

        impl<B: Backend> TryFrom<Tensor<$typ, B>> for $typ {
            type Error = Error;

            fn try_from(tensor: Tensor<$typ, B>) -> Result<Self, Self::Error> {
                $typ::try_from(&tensor)
            }
        }

        impl TryFrom<$typ> for CpuTensor<$typ> {
            type Error = Error;

            fn try_from(v: $typ) -> Result<Self, Self::Error> {
                Tensor::from_vec(vec![v], (), &CPU)
            }
        }
    };
}

from_tensor!(f32);
from_tensor!(f16);
from_tensor!(bf16);
from_tensor!(i64);
from_tensor!(u8);

/// Reinterpret a `Vec<T>` as a `Vec<U>`.
///
/// # Safety
/// Callers must only invoke this when `T` and `U` are the very same type (which is
/// guaranteed here by matching on `T::DTYPE` before selecting `U`). The `Vec` layout
/// is independent of the element type, so the reinterpretation is a no-op.
fn cast_vec<T, U>(v: Vec<T>) -> Vec<U> {
    unsafe { std::mem::transmute::<Vec<T>, Vec<U>>(v) }
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    /// Write the tensor data to `f` as little-endian bytes, in row-major order.
    pub fn write_bytes<W: std::io::Write>(&self, f: &mut W) -> crate::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let vs = self.flatten_all()?.to_vec1()?;
        match T::DTYPE {
            DType::BF16 => {
                for v in cast_vec::<T, bf16>(vs) {
                    f.write_u16::<LittleEndian>(v.to_bits())?
                }
            }
            DType::F16 => {
                for v in cast_vec::<T, f16>(vs) {
                    f.write_u16::<LittleEndian>(v.to_bits())?
                }
            }
            DType::F32 => {
                for v in cast_vec::<T, f32>(vs) {
                    f.write_f32::<LittleEndian>(v)?
                }
            }
            DType::I64 => {
                for v in cast_vec::<T, i64>(vs) {
                    f.write_i64::<LittleEndian>(v)?
                }
            }
            DType::U8 => {
                let vs = cast_vec::<T, u8>(vs);
                f.write_all(&vs)?;
            }
        }
        Ok(())
    }
}
