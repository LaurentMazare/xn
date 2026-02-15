#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

// Strided index computation (same as cuda_utils.cuh)
__device__ __forceinline__ unsigned int get_strided_index_layout(
    unsigned int idx,
    const unsigned int num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

// Copy from strided source to contiguous destination.
// info contains [dims..., src_strides...] packed into a single array.
template <typename T>
__device__ void copy_strided(
    const size_t numel,
    const unsigned int num_dims,
    const size_t *info,
    const unsigned int src_offset,
    const T *src,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    unsigned int src_idx = src_offset + get_strided_index_layout(idx, num_dims, dims, strides);
    dst[idx] = src[src_idx];
}

#define COPY_STRIDED_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void copy_strided_##RUST_NAME( \
    const size_t numel, \
    const unsigned int num_dims, \
    const size_t *info, \
    const unsigned int src_offset, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { copy_strided<TYPENAME>(numel, num_dims, info, src_offset, src, dst); }

#if __CUDA_ARCH__ >= 800
COPY_STRIDED_OP(__nv_bfloat16, bf16)
#endif
#if __CUDA_ARCH__ >= 530
COPY_STRIDED_OP(__half, f16)
#endif
COPY_STRIDED_OP(uint8_t, u8)
COPY_STRIDED_OP(int64_t, i64)
COPY_STRIDED_OP(float, f32)
COPY_STRIDED_OP(double, f64)

template <typename T>
__device__ void transpose(const size_t numel, const uint32_t d1,
                          const uint32_t d2, const uint32_t d_i,
                          const uint32_t d_j, const uint32_t d_k, const T *src,
                          T *dst) {
  const size_t dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= numel)
    return;

  // The implementation below is very slow as it computes lots of divisions and
  // multiplications.
  // TODO: Replace it with an optimized implementation and/or process data by
  // blocks.
  size_t dst_idx2 = dst_idx;
  const size_t i = dst_idx2 / (d2 * d_j * d1 * d_k);
  dst_idx2 -= i * d2 * d_j * d1 * d_k;
  const size_t a2 = dst_idx2 / (d_j * d1 * d_k);
  dst_idx2 -= a2 * d_j * d1 * d_k;
  const size_t j = dst_idx2 / (d1 * d_k);
  dst_idx2 -= j * d1 * d_k;
  const size_t a1 = dst_idx2 / d_k;
  dst_idx2 -= a1 * d_k;
  const size_t k = dst_idx2;
  const size_t src_idx = i * d1 * d_j * d2 * d_k + a1 * d_j * d2 * d_k +
                         j * d2 * d_k + a2 * d_k + k;

  dst[dst_idx] = src[src_idx];
}

#define OPS(TYPENAME, RUST_NAME)                                               \
  extern "C" __global__ void transpose_##RUST_NAME(                            \
      const size_t numel, const uint32_t d1, const uint32_t d2,                \
      const uint32_t d_i, const uint32_t d_j, const uint32_t d_k,              \
      const TYPENAME *src, TYPENAME *dst) {                                    \
    transpose<TYPENAME>(numel, d1, d2, d_i, d_j, d_k, src, dst);               \
  }

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(uint8_t, u8)
OPS(int64_t, i64)
OPS(float, f32)
OPS(double, f64)
