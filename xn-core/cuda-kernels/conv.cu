#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>
#include<math.h>

// ============================================================================
// Helper functions for type conversions
// ============================================================================

template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float(float v) { return v; }
template<> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ float from_float(float v) { return v; }
template<> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
#endif

// ============================================================================
// Im2Col1D kernel
// Transforms input from [B, C_in, L_in] to [B, L_out, C_in * K] for conv1d
// ============================================================================

template <typename T>
__device__ void im2col1d_kernel(
    const size_t numel,       // Total output elements (B * L_out * C_in * K)
    const size_t batch,
    const size_t c_in,
    const size_t l_in,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const T *src,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // dst layout: [B, L_out, C_in, K] flattened
    const size_t k_idx = idx % k_size;
    const size_t c_idx = (idx / k_size) % c_in;
    const size_t l_idx = (idx / (k_size * c_in)) % l_out;
    const size_t b_idx = idx / (k_size * c_in * l_out);

    // Compute source position
    size_t src_l = l_idx * stride + k_idx * dilation;

    if (src_l < padding || src_l >= l_in + padding) {
        dst[idx] = static_cast<T>(0);
    } else {
        src_l -= padding;
        const size_t src_idx = b_idx * c_in * l_in + c_idx * l_in + src_l;
        dst[idx] = src[src_idx];
    }
}

// ============================================================================
// Col2Im1D kernel
// Transforms col data from [B, L_in, C_out * K] to [B, C_out, L_out]
// ============================================================================

template <typename T>
__device__ void col2im1d_kernel(
    const size_t dst_numel,   // Total output elements (B * C_out * L_out)
    const size_t batch,
    const size_t l_in,        // Input length (before transpose conv)
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const T *src,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_numel) return;

    // dst layout: [B, C_out, L_out]
    const size_t l_out_idx = idx % l_out;
    const size_t c_idx = (idx / l_out) % c_out;
    const size_t b_idx = idx / (l_out * c_out);

    // src layout: [B, L_in, C_out, K]
    const size_t src_s0 = l_in * c_out * k_size;
    const size_t src_s1 = c_out * k_size;
    const size_t src_s2 = k_size;

    T sum = static_cast<T>(0);

    // Find all (l_in_idx, k_idx) pairs that contribute to this output position
    // l_out_idx = l_in_idx * stride + k_idx
    int l_in_idx = l_out_idx / stride;
    int k_start = l_out_idx - l_in_idx * stride;

    for (; k_start < (int)k_size && l_in_idx >= 0; k_start += stride, --l_in_idx) {
        if (l_in_idx < (int)l_in) {
            const size_t src_idx = b_idx * src_s0 + l_in_idx * src_s1 + c_idx * src_s2 + k_start;
            sum += src[src_idx];
        }
    }

    dst[idx] = sum;
}

// ============================================================================
// Direct Conv1D kernel (naive implementation for fallback)
// src: [B, C_in, L_in], kernel: [C_out, C_in/groups, K], dst: [B, C_out, L_out]
// ============================================================================

template <typename T>
__device__ void conv1d_direct_kernel(
    const size_t dst_numel,
    const size_t batch,
    const size_t c_in,
    const size_t l_in,
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t groups,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_numel) return;

    // dst layout: [B, C_out, L_out]
    const size_t l_idx = idx % l_out;
    const size_t c_out_idx = (idx / l_out) % c_out;
    const size_t b_idx = idx / (l_out * c_out);

    const size_t c_in_per_group = c_in / groups;
    const size_t c_out_per_group = c_out / groups;
    const size_t g = c_out_idx / c_out_per_group;
    const size_t c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (size_t k = 0; k < k_size; ++k) {
        size_t src_l = l_idx * stride + k * dilation;
        if (src_l < padding || src_l >= l_in + padding) {
            continue;
        }
        src_l -= padding;

        for (size_t c = 0; c < c_in_per_group; ++c) {
            const size_t src_c = c_in_start + c;
            const size_t src_idx = b_idx * c_in * l_in + src_c * l_in + src_l;
            // kernel layout: [C_out, C_in/groups, K]
            const size_t k_idx = c_out_idx * c_in_per_group * k_size + c * k_size + k;
            sum += to_float(src[src_idx]) * to_float(kernel[k_idx]);
        }
    }

    dst[idx] = from_float<T>(sum);
}

// ============================================================================
// Direct Conv Transpose 1D kernel (naive implementation for fallback)
// src: [B, C_in, L_in], kernel: [C_in, C_out/groups, K], dst: [B, C_out, L_out]
// ============================================================================

template <typename T>
__device__ void conv_transpose1d_direct_kernel(
    const size_t dst_numel,
    const size_t batch,
    const size_t c_in,
    const size_t l_in,
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t groups,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_numel) return;

    // dst layout: [B, C_out, L_out]
    const size_t l_out_idx = idx % l_out;
    const size_t c_out_idx = (idx / l_out) % c_out;
    const size_t b_idx = idx / (l_out * c_out);

    const size_t c_in_per_group = c_in / groups;
    const size_t c_out_per_group = c_out / groups;
    const size_t g = c_out_idx / c_out_per_group;
    const size_t c_in_start = g * c_in_per_group;
    const size_t c_out_in_group = c_out_idx % c_out_per_group;

    float sum = 0.0f;

    for (int k = 0; k < (int)k_size; ++k) {
        // out_l = in_l * stride + k * dilation - padding
        // We need: in_l = (out_l + padding - k * dilation) / stride
        int in_l_offset = (int)l_out_idx + (int)padding - k * (int)dilation;
        if (in_l_offset < 0 || in_l_offset % (int)stride != 0) {
            continue;
        }
        int in_l = in_l_offset / (int)stride;
        if (in_l >= (int)l_in) {
            continue;
        }

        for (size_t c = 0; c < c_in_per_group; ++c) {
            const size_t src_c = c_in_start + c;
            const size_t src_idx = b_idx * c_in * l_in + src_c * l_in + in_l;
            // kernel layout: [C_in, C_out/groups, K]
            const size_t k_idx = src_c * c_out_per_group * k_size + c_out_in_group * k_size + k;
            sum += to_float(src[src_idx]) * to_float(kernel[k_idx]);
        }
    }

    dst[idx] = from_float<T>(sum);
}

// ============================================================================
// Transpose kernel: [B, L, C] -> [B, C, L]
// ============================================================================

template <typename T>
__device__ void transpose_blc_to_bcl(
    const size_t numel,
    const size_t batch,
    const size_t length,
    const size_t channels,
    const T *src,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // src layout: [B, L, C]
    const size_t c_idx = idx % channels;
    const size_t l_idx = (idx / channels) % length;
    const size_t b_idx = idx / (channels * length);

    // dst layout: [B, C, L]
    const size_t dst_idx = b_idx * channels * length + c_idx * length + l_idx;
    dst[dst_idx] = src[idx];
}

// ============================================================================
// Transpose kernel: [B, C, L] -> [B, L, C]
// ============================================================================

template <typename T>
__device__ void transpose_bcl_to_blc(
    const size_t numel,
    const size_t batch,
    const size_t channels,
    const size_t length,
    const T *src,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // src layout: [B, C, L]
    const size_t l_idx = idx % length;
    const size_t c_idx = (idx / length) % channels;
    const size_t b_idx = idx / (length * channels);

    // dst layout: [B, L, C]
    const size_t dst_idx = b_idx * length * channels + l_idx * channels + c_idx;
    dst[dst_idx] = src[idx];
}

// ============================================================================
// Kernel instantiation macros
// ============================================================================

#define IM2COL1D_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void im2col1d_##RUST_NAME( \
    const size_t numel, \
    const size_t batch, \
    const size_t c_in, \
    const size_t l_in, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    im2col1d_kernel<TYPENAME>(numel, batch, c_in, l_in, l_out, k_size, stride, padding, dilation, src, dst); \
}

#define COL2IM1D_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void col2im1d_##RUST_NAME( \
    const size_t dst_numel, \
    const size_t batch, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    col2im1d_kernel<TYPENAME>(dst_numel, batch, l_in, c_out, l_out, k_size, stride, src, dst); \
}

#define CONV1D_DIRECT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void conv1d_direct_##RUST_NAME( \
    const size_t dst_numel, \
    const size_t batch, \
    const size_t c_in, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t groups, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
    conv1d_direct_kernel<TYPENAME>(dst_numel, batch, c_in, l_in, c_out, l_out, k_size, stride, padding, dilation, groups, src, kernel, dst); \
}

#define CONV_TRANSPOSE1D_DIRECT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void conv_transpose1d_direct_##RUST_NAME( \
    const size_t dst_numel, \
    const size_t batch, \
    const size_t c_in, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t groups, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
    conv_transpose1d_direct_kernel<TYPENAME>(dst_numel, batch, c_in, l_in, c_out, l_out, k_size, stride, padding, out_padding, dilation, groups, src, kernel, dst); \
}

#define TRANSPOSE_BLC_BCL_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void transpose_blc_bcl_##RUST_NAME( \
    const size_t numel, \
    const size_t batch, \
    const size_t length, \
    const size_t channels, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    transpose_blc_to_bcl<TYPENAME>(numel, batch, length, channels, src, dst); \
}

#define TRANSPOSE_BCL_BLC_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void transpose_bcl_blc_##RUST_NAME( \
    const size_t numel, \
    const size_t batch, \
    const size_t channels, \
    const size_t length, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    transpose_bcl_to_blc<TYPENAME>(numel, batch, channels, length, src, dst); \
}

#define ALL_CONV_OPS(TYPENAME, RUST_NAME) \
    IM2COL1D_OP(TYPENAME, RUST_NAME) \
    COL2IM1D_OP(TYPENAME, RUST_NAME) \
    CONV1D_DIRECT_OP(TYPENAME, RUST_NAME) \
    CONV_TRANSPOSE1D_DIRECT_OP(TYPENAME, RUST_NAME) \
    TRANSPOSE_BLC_BCL_OP(TYPENAME, RUST_NAME) \
    TRANSPOSE_BCL_BLC_OP(TYPENAME, RUST_NAME)

// ============================================================================
// Instantiate for all supported types
// ============================================================================

#if __CUDA_ARCH__ >= 800
ALL_CONV_OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
ALL_CONV_OPS(__half, f16)
#endif

ALL_CONV_OPS(float, f32)
