#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>
#include<math.h>

// ============================================================================
// Helper functions for type conversions and math
// ============================================================================

template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float(float v) { return v; }
template<> __device__ __forceinline__ float to_float(double v) { return (float)v; }
template<> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ float from_float(float v) { return v; }
template<> __device__ __forceinline__ double from_float(float v) { return (double)v; }
template<> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
#endif

// ============================================================================
// Scale-add operation (out-of-place: dst = src * scale + add)
// ============================================================================

template <typename T>
__device__ void scale_add_op(const size_t numel, const T * src, T * dst, const T scale, const T add) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = src[idx] * scale + add;
}

// ============================================================================
// Binary operations (out-of-place: dst = lhs op rhs)
// ============================================================================

template <typename T>
__device__ void binary_add(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = lhs[idx] + rhs[idx];
}

template <typename T>
__device__ void binary_sub(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = lhs[idx] - rhs[idx];
}

template <typename T>
__device__ void binary_mul(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = lhs[idx] * rhs[idx];
}

template <typename T>
__device__ void binary_div(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = lhs[idx] / rhs[idx];
}

template <typename T>
__device__ void binary_maximum(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T l = lhs[idx];
    T r = rhs[idx];
    dst[idx] = (l > r) ? l : r;
}

template <typename T>
__device__ void binary_minimum(const size_t numel, const T * lhs, const T * rhs, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T l = lhs[idx];
    T r = rhs[idx];
    dst[idx] = (l < r) ? l : r;
}

// ============================================================================
// Binary assign operations (in-place: dst op= src)
// ============================================================================

template <typename T>
__device__ void assign_add(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] += src[idx];
}

template <typename T>
__device__ void assign_sub(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] -= src[idx];
}

template <typename T>
__device__ void assign_mul(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] *= src[idx];
}

template <typename T>
__device__ void assign_div(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] /= src[idx];
}

template <typename T>
__device__ void assign_maximum(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T d = dst[idx];
    T s = src[idx];
    dst[idx] = (d > s) ? d : s;
}

template <typename T>
__device__ void assign_minimum(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T d = dst[idx];
    T s = src[idx];
    dst[idx] = (d < s) ? d : s;
}

// ============================================================================
// Unary operations (out-of-place: dst = op(src))
// ============================================================================

template <typename T>
__device__ void unary_cos(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(cosf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_sin(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(sinf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_sqr(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T v = src[idx];
    dst[idx] = v * v;
}

template <typename T>
__device__ void unary_sqrt(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(sqrtf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_rsqrt(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(rsqrtf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_abs(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(fabsf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_gelu_erf(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(src[idx]);
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    float result = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void unary_elu(const size_t numel, const T * src, T * dst, float alpha) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(src[idx]);
    float result = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void unary_relu(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(src[idx]);
    dst[idx] = from_float<T>((x > 0.0f) ? x : 0.0f);
}

template <typename T>
__device__ void unary_silu(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(src[idx]);
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float result = x / (1.0f + expf(-x));
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void unary_tanh(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(tanhf(to_float(src[idx])));
}

template <typename T>
__device__ void unary_sigmoid(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(src[idx]);
    dst[idx] = from_float<T>(1.0f / (1.0f + expf(-x)));
}

// ============================================================================
// Inplace unary operations (dst = op(dst))
// ============================================================================

template <typename T>
__device__ void inplace_cos(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(cosf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_sin(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(sinf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_sqr(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    T v = dst[idx];
    dst[idx] = v * v;
}

template <typename T>
__device__ void inplace_sqrt(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(sqrtf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_rsqrt(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(rsqrtf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_abs(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(fabsf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_gelu_erf(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(dst[idx]);
    float result = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void inplace_elu(const size_t numel, T * dst, float alpha) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(dst[idx]);
    float result = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void inplace_relu(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(dst[idx]);
    dst[idx] = from_float<T>((x > 0.0f) ? x : 0.0f);
}

template <typename T>
__device__ void inplace_silu(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(dst[idx]);
    float result = x / (1.0f + expf(-x));
    dst[idx] = from_float<T>(result);
}

template <typename T>
__device__ void inplace_tanh(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = from_float<T>(tanhf(to_float(dst[idx])));
}

template <typename T>
__device__ void inplace_sigmoid(const size_t numel, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = to_float(dst[idx]);
    dst[idx] = from_float<T>(1.0f / (1.0f + expf(-x)));
}

// ============================================================================
// Kernel definitions macro
// ============================================================================

#define BINARY_OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void binary_add_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_add<TYPENAME>(numel, lhs, rhs, dst); \
  } \
  extern "C" __global__ void binary_sub_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_sub<TYPENAME>(numel, lhs, rhs, dst); \
  } \
  extern "C" __global__ void binary_mul_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_mul<TYPENAME>(numel, lhs, rhs, dst); \
  } \
  extern "C" __global__ void binary_div_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_div<TYPENAME>(numel, lhs, rhs, dst); \
  } \
  extern "C" __global__ void binary_maximum_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_maximum<TYPENAME>(numel, lhs, rhs, dst); \
  } \
  extern "C" __global__ void binary_minimum_##RUST_NAME( \
      const size_t numel, const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst) { \
    binary_minimum<TYPENAME>(numel, lhs, rhs, dst); \
  } \

#define ASSIGN_OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void assign_add_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_add<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void assign_sub_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_sub<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void assign_mul_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_mul<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void assign_div_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_div<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void assign_maximum_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_maximum<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void assign_minimum_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    assign_minimum<TYPENAME>(numel, src, dst); \
  } \

#define UNARY_OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void unary_cos_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_cos<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_sin_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_sin<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_sqr_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_sqr<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_sqrt_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_sqrt<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_rsqrt_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_rsqrt<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_abs_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_abs<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_gelu_erf_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_gelu_erf<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_elu_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst, float alpha) { \
    unary_elu<TYPENAME>(numel, src, dst, alpha); \
  } \
  extern "C" __global__ void unary_relu_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_relu<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_silu_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_silu<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_tanh_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_tanh<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void unary_sigmoid_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst) { \
    unary_sigmoid<TYPENAME>(numel, src, dst); \
  } \

#define INPLACE_UNARY_OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void inplace_cos_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_cos<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_sin_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_sin<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_sqr_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_sqr<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_sqrt_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_sqrt<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_rsqrt_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_rsqrt<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_abs_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_abs<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_gelu_erf_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_gelu_erf<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_elu_##RUST_NAME( \
      const size_t numel, TYPENAME *dst, float alpha) { \
    inplace_elu<TYPENAME>(numel, dst, alpha); \
  } \
  extern "C" __global__ void inplace_relu_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_relu<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_silu_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_silu<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_tanh_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_tanh<TYPENAME>(numel, dst); \
  } \
  extern "C" __global__ void inplace_sigmoid_##RUST_NAME( \
      const size_t numel, TYPENAME *dst) { \
    inplace_sigmoid<TYPENAME>(numel, dst); \
  } \

#define SCALE_ADD_OP(TYPENAME, RUST_NAME) \
  extern "C" __global__ void scale_add_##RUST_NAME( \
      const size_t numel, const TYPENAME *src, TYPENAME *dst, const TYPENAME scale, const TYPENAME add) { \
    scale_add_op<TYPENAME>(numel, src, dst, scale, add); \
  } \

#define ALL_OPS(TYPENAME, RUST_NAME) \
  BINARY_OPS(TYPENAME, RUST_NAME) \
  ASSIGN_OPS(TYPENAME, RUST_NAME) \
  UNARY_OPS(TYPENAME, RUST_NAME) \
  INPLACE_UNARY_OPS(TYPENAME, RUST_NAME) \
  SCALE_ADD_OP(TYPENAME, RUST_NAME) \

#if __CUDA_ARCH__ >= 800
ALL_OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
ALL_OPS(__half, f16)
#endif

ALL_OPS(float, f32)
ALL_OPS(double, f64)

