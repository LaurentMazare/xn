// Kernels extracted from:
// https://github.com/vllm-project/vllm/blob/fafca38adc1ce65d2c9e2857138c3c0d65b0905e/csrc/quantization/w8a8/fp8/common.cu#L1
#include "cuda_utils.cuh"
#include "vectorization_utils.cuh"
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <tuple>

// ---------------------------------------------------------------------------
// FP8 helpers
// ---------------------------------------------------------------------------

// We target E4M3 (fp8_e4m3fn) for weights/activations quantization.
using fp8_e4m3 = __nv_fp8_e4m3;

// Max representable value for FP8 E4M3: 448.0
template <typename T> struct quant_type_max;
template <> struct quant_type_max<fp8_e4m3> {
    static constexpr float value = 448.0f;
};
template <typename T>
inline constexpr float quant_type_max_v = quant_type_max<T>::value;

// Minimum scaling factor to avoid division by zero / denorm issues.
template <typename T> struct min_scaling_factor;
template <> struct min_scaling_factor<fp8_e4m3> {
    __device__ __host__ static constexpr float val() { return 1.0f / (FP8_E4M3_MAX * 512.0f); }
    static constexpr float FP8_E4M3_MAX = 448.0f;
};

// Convert a float value to FP8 with a scaling factor.
// If `is_inverse_scale` is true, `scale` is 1/real_scale (i.e. multiply by it).
// Otherwise, `scale` is the real scale (divide by it).
template <bool is_inverse_scale, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float val, float scale) {
    float x;
    if constexpr (is_inverse_scale) {
        x = val * scale;
    } else {
        x = val / scale;
    }
    // Clamp to FP8 range before conversion.
    x = fminf(fmaxf(x, -quant_type_max_v<fp8_type>), quant_type_max_v<fp8_type>);
    return static_cast<fp8_type>(x);
}

// atomicMax for float (used in per-tensor absmax reduction).
__device__ __forceinline__ void atomicMaxFloat(float *addr, float val) {
    if (val <= 0.0f) return;
    // Use integer atomicMax on positive floats (IEEE 754 positive floats
    // have the same ordering as their integer bit patterns).
    atomicMax(reinterpret_cast<int *>(addr), __float_as_int(val));
}

// Simple max functor for cub::BlockReduce.
struct CubMaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <typename scalar_t, typename fp8_type, bool STRIDE_I_ZERO,
          bool STRIDE_J_ZERO>
__device__ void scaled_fp8_quant_kernel_strided_group_shape(
    fp8_type *__restrict__ out, const scalar_t *__restrict__ input,
    const float *__restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride, int group_m, int group_n, int64_t scale_stride_i,
    int64_t scale_stride_j) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t *token_in = input + token_idx * in_row_stride;
  fp8_type *token_out = out + token_idx * out_row_stride;

  // Precompute row-level base offset for scale access (compile-time eliminated
  // when STRIDE_I_ZERO)
  const int64_t scale_row_base =
      STRIDE_I_ZERO ? 0
                    : static_cast<int>(token_idx) / group_m * scale_stride_i;

  auto get_inv_scale = [&](int gj) {
    return 1.0f / scale[scale_row_base + gj * scale_stride_j];
  };

  int cached_gj = -1;
  float cached_inv_scale = 0.0f;
  auto get_inv_scale_cached = [&](int gj) {
    if (gj != cached_gj) {
      cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];
      cached_gj = gj;
    }
    return cached_inv_scale;
  };

  constexpr int VEC_SIZE = 16; // FP8 so vectorize to 128 bits
  auto scaled_fp8_conversion_vectorized = [&](const scalar_t *in, fp8_type *out,
                                              int size, float inv_scale) {
    vectorize_with_alignment<VEC_SIZE>(
        in, out, size, tid, blockDim.x,
        [=] __device__(fp8_type & dst, const scalar_t &src) {
          dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                      inv_scale);
        });
  };

  if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
    // Per-tensor or per-token: single scale per row, vectorize full row
    scaled_fp8_conversion_vectorized(token_in, token_out, hidden_size,
                                     get_inv_scale(0));
  } else if (group_n % VEC_SIZE == 0) {
    // Multiple column groups with vectorization
    const int num_groups_n = hidden_size / group_n;

    for (int gj = 0; gj < num_groups_n; gj++) {
      scaled_fp8_conversion_vectorized(token_in + gj * group_n,
                                       token_out + gj * group_n, group_n,
                                       get_inv_scale(gj));
    }
  } else {
    // Scalar path for small column groups (group_n < VEC_SIZE)
    for (int n = tid; n < hidden_size; n += blockDim.x) {
      const int gj = n / group_n;
      token_out[n] = scaled_fp8_conversion<true, fp8_type>(
          static_cast<float>(token_in[n]), get_inv_scale_cached(gj));
    }
  }
}

template <typename scalar_t, typename fp8_type>
__device__ void segmented_max_reduction_strided(
    float *__restrict__ scale, const scalar_t *__restrict__ input,
    int hidden_size, int64_t in_row_stride, int64_t num_tokens) {
  __shared__ float cache[256];
  const int tid = threadIdx.x;
  int64_t token_idx = blockIdx.x;

  // one block per token. Guard in case gridDim.x > num_tokens.
  if (token_idx >= num_tokens) {
    return;
  }

  const scalar_t *row_ptr = input + token_idx * in_row_stride;

  // each thread scans elements of the row in a strided fashion.
  float thread_max = 0.0f;
  for (int e = tid; e < hidden_size; e += blockDim.x) {
    float v = fabsf(static_cast<float>(row_ptr[e]));
    thread_max = fmaxf(thread_max, v);
  }

  cache[tid] = thread_max;
  __syncthreads();

  // parallel reduction to find row max.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      cache[tid] = fmaxf(cache[tid], cache[tid + offset]);
    }
    __syncthreads();
  }

  // thread 0 updates global scale (per-tensor) atomically.
  if (tid == 0) {
    atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t, typename fp8_type>
__device__ void scaled_fp8_quant_kernel_strided_dynamic(
    fp8_type *__restrict__ out, const scalar_t *__restrict__ input,
    const float *__restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t *token_in = input + token_idx * in_row_stride;
  fp8_type *token_out = out + token_idx * out_row_stride;

  const float reciprocal_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t &src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    reciprocal_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__device__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type *__restrict__ out, float *__restrict__ scale,
    const scalar_t *__restrict__ input, const float *__restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
  int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
  const scalar_t *token_in = input + in_offset;
  fp8_type *token_out = out + out_offset;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max =
      BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t &src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

// ---------------------------------------------------------------------------
// extern "C" instantiations for bf16 -> fp8_e4m3
// ---------------------------------------------------------------------------

// Per-tensor absmax reduction: launch with one block per row, 256 threads.
// After this kernel, scale[0] = max(absmax_per_row) / FP8_E4M3_MAX.
// The caller must zero-initialize `scale` before launching.
extern "C" __global__ void segmented_max_reduction_bf16(
    float *__restrict__ scale,
    const __nv_bfloat16 *__restrict__ input,
    int hidden_size,
    long long in_row_stride,
    long long num_tokens
) {
    segmented_max_reduction_strided<__nv_bfloat16, fp8_e4m3>(
        scale, input, hidden_size, in_row_stride, num_tokens);
}

// Per-tensor quantization: launch with one block per row, 256 threads.
// `scale` should already contain the computed scale from the reduction kernel.
extern "C" __global__ void scaled_fp8_quant_dynamic_bf16(
    __nv_fp8_e4m3 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ input,
    const float *__restrict__ scale,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    scaled_fp8_quant_kernel_strided_dynamic<__nv_bfloat16, fp8_e4m3>(
        out, input, scale, hidden_size, in_row_stride, out_row_stride);
}

// Dequantize FP8 E4M3 -> bf16: out[i] = (float)fp8[i] * scale
extern "C" __global__ void fp8_dequant_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel
) {
    const float s = *scale;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        float v = static_cast<float>(input[i]) * s;
        out[i] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------------------
// extern "C" instantiations for f16 -> fp8_e4m3
// ---------------------------------------------------------------------------

extern "C" __global__ void segmented_max_reduction_f16(
    float *__restrict__ scale,
    const __half *__restrict__ input,
    int hidden_size,
    long long in_row_stride,
    long long num_tokens
) {
    segmented_max_reduction_strided<__half, fp8_e4m3>(
        scale, input, hidden_size, in_row_stride, num_tokens);
}

extern "C" __global__ void scaled_fp8_quant_dynamic_f16(
    __nv_fp8_e4m3 *__restrict__ out,
    const __half *__restrict__ input,
    const float *__restrict__ scale,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    scaled_fp8_quant_kernel_strided_dynamic<__half, fp8_e4m3>(
        out, input, scale, hidden_size, in_row_stride, out_row_stride);
}

extern "C" __global__ void fp8_dequant_f16(
    __half *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel
) {
    const float s = *scale;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        float v = static_cast<float>(input[i]) * s;
        out[i] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// extern "C" instantiations for f32 -> fp8_e4m3
// ---------------------------------------------------------------------------

extern "C" __global__ void segmented_max_reduction_f32(
    float *__restrict__ scale,
    const float *__restrict__ input,
    int hidden_size,
    long long in_row_stride,
    long long num_tokens
) {
    segmented_max_reduction_strided<float, fp8_e4m3>(
        scale, input, hidden_size, in_row_stride, num_tokens);
}

extern "C" __global__ void scaled_fp8_quant_dynamic_f32(
    __nv_fp8_e4m3 *__restrict__ out,
    const float *__restrict__ input,
    const float *__restrict__ scale,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    scaled_fp8_quant_kernel_strided_dynamic<float, fp8_e4m3>(
        out, input, scale, hidden_size, in_row_stride, out_row_stride);
}

// Dequantize FP8 E4M3 -> f32: out[i] = (float)fp8[i] * scale
extern "C" __global__ void fp8_dequant_f32(
    float *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel
) {
    const float s = *scale;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        out[i] = static_cast<float>(input[i]) * s;
    }
}

// ---------------------------------------------------------------------------
// Per-token quantization: extern "C" instantiations
// ---------------------------------------------------------------------------

extern "C" __global__ void dynamic_per_token_scaled_fp8_quant_bf16(
    __nv_fp8_e4m3 *__restrict__ out,
    float *__restrict__ scale,
    const __nv_bfloat16 *__restrict__ input,
    const float *__restrict__ scale_ub,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    dynamic_per_token_scaled_fp8_quant_kernel_strided<__nv_bfloat16, fp8_e4m3>(
        out, scale, input, scale_ub, hidden_size, in_row_stride, out_row_stride);
}

extern "C" __global__ void dynamic_per_token_scaled_fp8_quant_f16(
    __nv_fp8_e4m3 *__restrict__ out,
    float *__restrict__ scale,
    const __half *__restrict__ input,
    const float *__restrict__ scale_ub,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    dynamic_per_token_scaled_fp8_quant_kernel_strided<__half, fp8_e4m3>(
        out, scale, input, scale_ub, hidden_size, in_row_stride, out_row_stride);
}

extern "C" __global__ void dynamic_per_token_scaled_fp8_quant_f32(
    __nv_fp8_e4m3 *__restrict__ out,
    float *__restrict__ scale,
    const float *__restrict__ input,
    const float *__restrict__ scale_ub,
    int hidden_size,
    long long in_row_stride,
    long long out_row_stride
) {
    dynamic_per_token_scaled_fp8_quant_kernel_strided<float, fp8_e4m3>(
        out, scale, input, scale_ub, hidden_size, in_row_stride, out_row_stride);
}

// ---------------------------------------------------------------------------
// Per-token dequantization
// ---------------------------------------------------------------------------

extern "C" __global__ void fp8_dequant_per_token_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel,
    const int hidden_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int token_idx = i / hidden_size;
        float v = static_cast<float>(input[i]) * scale[token_idx];
        out[i] = __float2bfloat16(v);
    }
}

extern "C" __global__ void fp8_dequant_per_token_f16(
    __half *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel,
    const int hidden_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int token_idx = i / hidden_size;
        float v = static_cast<float>(input[i]) * scale[token_idx];
        out[i] = __float2half(v);
    }
}

extern "C" __global__ void fp8_dequant_per_token_f32(
    float *__restrict__ out,
    const __nv_fp8_e4m3 *__restrict__ input,
    const float *__restrict__ scale,
    const unsigned int numel,
    const int hidden_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        int token_idx = i / hidden_size;
        out[i] = static_cast<float>(input[i]) * scale[token_idx];
    }
}
