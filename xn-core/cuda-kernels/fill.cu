#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

template <typename T>
__device__ void fill(T *dst, const T value, const size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;
  dst[idx] = value;
}

#define FILL_OP(TYPENAME, RUST_NAME)                                           \
  extern "C" __global__ void fill_##RUST_NAME(                                 \
      TYPENAME *dst, const TYPENAME value, const size_t numel) {               \
    fill<TYPENAME>(dst, value, numel);                                         \
  }

template <typename T>
__device__ void copy2d(const T *src, T *dst, uint32_t d1, uint32_t d2,
                       uint32_t src_s, uint32_t dst_s) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  dst[idx1 * dst_s + idx2] = src[idx1 * src_s + idx2];
}

#define COPY2D_OP(TYPENAME, FNNAME)                                            \
  extern "C" __global__ void FNNAME(const TYPENAME *src, TYPENAME *dst,        \
                                    uint32_t d1, uint32_t d2, uint32_t src_s,  \
                                    uint32_t dst_s) {                          \
    copy2d(src, dst, d1, d2, src_s, dst_s);                                    \
  }

#if __CUDA_ARCH__ >= 800
FILL_OP(__nv_bfloat16, bf16)
COPY2D_OP(__nv_bfloat16, copy2d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
FILL_OP(__half, f16)
COPY2D_OP(__half, copy2d_f16)
#endif

FILL_OP(float, f32)
FILL_OP(double, f64)
FILL_OP(int64_t, i64)
FILL_OP(uint8_t, u8)
COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(int64_t, copy2d_i64)
COPY2D_OP(uint8_t, copy2d_u8)
