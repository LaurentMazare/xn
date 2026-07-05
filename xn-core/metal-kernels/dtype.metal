// Shared prelude selecting the storage scalar type. This file is prepended to
// every kernel source at library-build time (see `metal_backend/mod.rs`).
//   (default)  f32:  SCALAR = float,  LOAD/STORE are identity
//   USE_F16    f16:  SCALAR = half,   hardware converts on load/store
//   USE_BF16   bf16: SCALAR = ushort, bf16 is the top half of an f32, so
//              LOAD shifts the bits up and STORE rounds to nearest-even
//              (matching `half::bf16::from_f32` on the CPU side).
//   USE_I64    i64:  SCALAR = long,   integer compute (no float roundtrip)
//   USE_U8     u8:   SCALAR = uchar,  integer compute
// The float variants compute in f32; ACC is the arithmetic type LOAD produces
// (f32 for the float variants, the native type for the integer ones — only
// the dtype-generic kernels are compiled for those, see KERNEL_SRCS_INT).
// Reads must use LOAD(x) and arithmetic writes STORE(v); direct
// SCALAR-to-SCALAR assignment (pure copies) works in every variant.
#include <metal_stdlib>
using namespace metal;

#if defined(USE_F16)
typedef half SCALAR;
typedef half4 SCALAR4;
typedef float ACC;
#define LOAD(x) float(x)
#define LOAD4(x) float4(x)
#define STORE(v) SCALAR(v)
#elif defined(USE_BF16)
typedef ushort SCALAR;
typedef ushort4 SCALAR4;
typedef float ACC;
inline float bf16_load(ushort x) {
    return as_type<float>(uint(x) << 16);
}
inline float4 bf16_load4(ushort4 x) {
    return float4(bf16_load(x.x), bf16_load(x.y), bf16_load(x.z), bf16_load(x.w));
}
#define LOAD4(x) bf16_load4(x)
inline ushort bf16_store(float v) {
    uint b = as_type<uint>(v);
    // NaN check on the bit pattern (robust under fast-math): quieten instead
    // of rounding, as rounding a NaN's mantissa can carry into the exponent
    // and produce inf/garbage.
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return ushort((b >> 16) | 0x40u);
    }
    // Round to nearest, ties to even.
    return ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
#define LOAD(x) bf16_load(x)
#define STORE(v) bf16_store(v)
#elif defined(USE_I64)
typedef long SCALAR;
typedef long ACC;
#define LOAD(x) (x)
#define STORE(v) (v)
#elif defined(USE_U8)
typedef uchar SCALAR;
typedef uchar ACC;
#define LOAD(x) (x)
#define STORE(v) (v)
#else
typedef float SCALAR;
typedef float4 SCALAR4;
typedef float ACC;
#define LOAD(x) (x)
#define LOAD4(x) (x)
#define STORE(v) (v)
#endif
