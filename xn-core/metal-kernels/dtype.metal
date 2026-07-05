// Shared prelude selecting the storage scalar type. This file is prepended to
// every kernel source at library-build time (see `metal_backend/mod.rs`).
// Compute is always done in f32; only the buffer element type changes:
//   (default)  f32:  SCALAR = float,  LOAD/STORE are identity
//   USE_F16    f16:  SCALAR = half,   hardware converts on load/store
//   USE_BF16   bf16: SCALAR = ushort, bf16 is the top half of an f32, so
//              LOAD shifts the bits up and STORE rounds to nearest-even
//              (matching `half::bf16::from_f32` on the CPU side).
// Reads must use LOAD(x) and float writes STORE(v); direct SCALAR-to-SCALAR
// assignment (pure copies) works in all three variants.
#include <metal_stdlib>
using namespace metal;

#if defined(USE_F16)
typedef half SCALAR;
#define LOAD(x) float(x)
#define STORE(v) SCALAR(v)
#elif defined(USE_BF16)
typedef ushort SCALAR;
inline float bf16_load(ushort x) {
    return as_type<float>(uint(x) << 16);
}
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
#else
typedef float SCALAR;
#define LOAD(x) (x)
#define STORE(v) (v)
#endif
