// Shared prelude selecting the storage scalar type. Include this right after
// `#version 450`. Compute is always done in f32; only the SSBO element type
// changes. With -DUSE_F16 buffers hold float16_t (half the bandwidth),
// otherwise float. Reads use float(x), writes use SCALAR(x) — identity casts
// in the f32 variant.
#ifdef USE_F16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define SCALAR float16_t
#else
#define SCALAR float
#endif
