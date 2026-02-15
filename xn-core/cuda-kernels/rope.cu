#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void ropei(const T * cos, const T * sin, const T * src, T * dst, const uint32_t bh, const uint32_t td) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t rope_idx = idx % (td / 2);
    T c = cos[rope_idx];
    T s = sin[rope_idx];

    T src1 = src[2 * idx];
    T src2 = src[2 * idx + 1];

    dst[2 * idx] = src1 * c - src2 * s;
    dst[2 * idx + 1] = src1 * s + src2 * c;
}

template <typename T>
__device__ void rope(const T * cos, const T * sin, const T * src, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bh = idx / (td / 2);
    uint32_t i_td = idx - (td / 2) * i_bh;
    uint32_t i_t = i_td / (d / 2);
    uint32_t i_d = i_td - (d / 2) * i_t;
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    T c = cos[i_cs];
    T s = sin[i_cs];
    T src1 = src[i1];
    T src2 = src[i2];

    dst[i1] = src1 * c - src2 * s;
    dst[i2] = src1 * s + src2 * c;
}

template <typename T>
__device__ void rope_thd(
    const T * cos,
    const T * sin,
    const T * src,
    T * dst,
    const uint32_t b,
    const uint32_t t,
    const uint32_t h,
    const uint32_t d
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= b * t * h * d) return;

    uint32_t i_bth = idx / (d / 2);
    uint32_t i_d = idx - (d / 2) * i_bth;
    uint32_t i_t = (i_bth / h) % t;
    uint32_t i1 = i_bth * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    T c = cos[i_cs];
    T s = sin[i_cs];
    T src1 = src[i1];
    T src2 = src[i2];

    dst[i1] = src1 * c - src2 * s;
    dst[i2] = src1 * s + src2 * c;
}

#define ROPE_OP(TYPENAME, FN_NAME, FN_NAME_I, FN_NAME_THD) \
  extern "C" __global__ void FN_NAME_I( \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      const TYPENAME *src, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td) { \
    ropei<TYPENAME>(cos, sin, src, dst, bh, td); \
  } \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      const TYPENAME *src, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t d) { \
    rope<TYPENAME>(cos, sin, src, dst, bh, td, d); \
  } \
  extern "C" __global__ void FN_NAME_THD( \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      const TYPENAME *src, \
      TYPENAME *dst, \
      const uint32_t b, \
      const uint32_t t, \
      const uint32_t h, \
      const uint32_t d) { \
    rope_thd<TYPENAME>(cos, sin, src, dst, b, t, h, d); \
  } \

#if __CUDA_ARCH__ >= 800
ROPE_OP(__nv_bfloat16, rope_bf16, rope_i_bf16, rope_thd_bf16)
#endif

#if __CUDA_ARCH__ >= 530
ROPE_OP(__half, rope_f16, rope_i_f16, rope_thd_f16)
#endif

ROPE_OP(float, rope_f32, rope_i_f32, rope_thd_f32)
ROPE_OP(double, rope_f64, rope_i_f64, rope_thd_f64)
