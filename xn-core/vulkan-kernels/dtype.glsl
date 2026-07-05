// Shared prelude selecting the storage scalar type. Include this right after
// `#version 450`. Compute is always done in f32; only the SSBO element type
// changes:
//   (default)  f32:  SCALAR = float,     LOAD/STORE are identity
//   -DUSE_F16  f16:  SCALAR = float16_t, hardware converts on load/store
//   -DUSE_BF16 bf16: SCALAR = uint16_t,  bf16 is the top half of an f32, so
//              LOAD shifts the bits up and STORE rounds to nearest-even
//              (matching `half::bf16::from_f32` on the CPU side).
// Reads must use LOAD(x) and float writes STORE(v); direct SCALAR-to-SCALAR
// assignment (pure copies) works in all three variants.
#ifdef USE_F16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define SCALAR float16_t
#define LOAD(x) float(x)
#define STORE(v) float16_t(v)
#elif defined(USE_BF16)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#define SCALAR uint16_t
float bf16_load(uint16_t x) {
    return uintBitsToFloat(uint(x) << 16);
}
uint16_t bf16_store(float v) {
    uint b = floatBitsToUint(v);
    if (isnan(v)) {
        // Quieten instead of rounding: rounding a NaN's mantissa can carry
        // into the exponent and produce inf/garbage.
        return uint16_t((b >> 16) | 0x40u);
    }
    // Round to nearest, ties to even.
    return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
#define LOAD(x) bf16_load(x)
#define STORE(v) bf16_store(v)
#else
#define SCALAR float
#define LOAD(x) (x)
#define STORE(v) (v)
#endif
