#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>
#include<math.h>

// ============================================================================
// Helper functions for type conversions
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
// Binary operation functors
// ============================================================================

template<typename T>
struct AddOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct SubOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a - b; }
};

template<typename T>
struct MulOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template<typename T>
struct DivOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a / b; }
};

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return (a > b) ? a : b; }
};

template<typename T>
struct MinOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return (a < b) ? a : b; }
};

// ============================================================================
// Strided index calculation
// ============================================================================

__device__ __forceinline__ unsigned int get_strided_index(
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

// ============================================================================
// General strided broadcast binary kernel
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_strided(
    const size_t numel,
    const unsigned int num_dims,
    const size_t *info, // [dims..., lhs_strides..., rhs_strides...]
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    const size_t *dims = info;
    const size_t *lhs_strides = info + num_dims;
    const size_t *rhs_strides = info + 2 * num_dims;

    unsigned int lhs_idx = get_strided_index(idx, num_dims, dims, lhs_strides);
    unsigned int rhs_idx = get_strided_index(idx, num_dims, dims, rhs_strides);

    dst[idx] = op(lhs[lhs_idx], rhs[rhs_idx]);
}

// ============================================================================
// Optimized kernel: both operands contiguous (no broadcast)
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_contiguous(
    const size_t numel,
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = op(lhs[idx], rhs[idx]);
}

// ============================================================================
// Optimized kernel: rhs broadcast along first dim (rhs_strides = [0, 1])
// lhs is contiguous, rhs has shape [..., dim1] broadcast to [dim0, dim1]
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_rhs_row(
    const size_t numel,
    const size_t dim1,
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned int rhs_idx = idx % dim1;
    dst[idx] = op(lhs[idx], rhs[rhs_idx]);
}

// ============================================================================
// Optimized kernel: rhs broadcast along second dim (rhs_strides = [1, 0])
// lhs is contiguous, rhs has shape [dim0, ...] broadcast to [dim0, dim1]
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_rhs_col(
    const size_t numel,
    const size_t dim1,
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned int rhs_idx = idx / dim1;
    dst[idx] = op(lhs[idx], rhs[rhs_idx]);
}

// ============================================================================
// Optimized kernel: lhs broadcast along first dim (lhs_strides = [0, 1])
// rhs is contiguous, lhs has shape [..., dim1] broadcast to [dim0, dim1]
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_lhs_row(
    const size_t numel,
    const size_t dim1,
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned int lhs_idx = idx % dim1;
    dst[idx] = op(lhs[lhs_idx], rhs[idx]);
}

// ============================================================================
// Optimized kernel: lhs broadcast along second dim (lhs_strides = [1, 0])
// rhs is contiguous, lhs has shape [dim0, ...] broadcast to [dim0, dim1]
// ============================================================================

template<typename T, typename Op>
__device__ void broadcast_binary_lhs_col(
    const size_t numel,
    const size_t dim1,
    const T *lhs,
    const T *rhs,
    T *dst,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned int lhs_idx = idx / dim1;
    dst[idx] = op(lhs[lhs_idx], rhs[idx]);
}

// ============================================================================
// Kernel instantiation macros
// ============================================================================

#define BROADCAST_STRIDED_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_strided_##RUST_NAME( \
    const size_t numel, \
    const unsigned int num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_strided<TYPENAME, OP_TYPE<TYPENAME>>(numel, num_dims, info, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_CONTIGUOUS_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_contiguous_##RUST_NAME( \
    const size_t numel, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_contiguous<TYPENAME, OP_TYPE<TYPENAME>>(numel, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_RHS_ROW_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_rhs_row_##RUST_NAME( \
    const size_t numel, \
    const size_t dim1, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_rhs_row<TYPENAME, OP_TYPE<TYPENAME>>(numel, dim1, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_RHS_COL_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_rhs_col_##RUST_NAME( \
    const size_t numel, \
    const size_t dim1, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_rhs_col<TYPENAME, OP_TYPE<TYPENAME>>(numel, dim1, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_LHS_ROW_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_lhs_row_##RUST_NAME( \
    const size_t numel, \
    const size_t dim1, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_lhs_row<TYPENAME, OP_TYPE<TYPENAME>>(numel, dim1, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_LHS_COL_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
extern "C" __global__ void broadcast_##OP_NAME##_lhs_col_##RUST_NAME( \
    const size_t numel, \
    const size_t dim1, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *dst \
) { \
    broadcast_binary_lhs_col<TYPENAME, OP_TYPE<TYPENAME>>(numel, dim1, lhs, rhs, dst, OP_TYPE<TYPENAME>()); \
}

#define BROADCAST_ALL_VARIANTS(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_STRIDED_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_CONTIGUOUS_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_RHS_ROW_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_RHS_COL_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_LHS_ROW_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME) \
    BROADCAST_LHS_COL_KERNEL(OP_NAME, OP_TYPE, TYPENAME, RUST_NAME)

#define BROADCAST_ALL_OPS(TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(add, AddOp, TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(sub, SubOp, TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(mul, MulOp, TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(div, DivOp, TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(max, MaxOp, TYPENAME, RUST_NAME) \
    BROADCAST_ALL_VARIANTS(min, MinOp, TYPENAME, RUST_NAME)

// ============================================================================
// Instantiate for all supported types
// ============================================================================

#if __CUDA_ARCH__ >= 800
BROADCAST_ALL_OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
BROADCAST_ALL_OPS(__half, f16)
#endif

BROADCAST_ALL_OPS(float, f32)
BROADCAST_ALL_OPS(int64_t, i64)
BROADCAST_ALL_OPS(uint8_t, u8)
