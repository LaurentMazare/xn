// Dtype conversion kernels between the float storage types, so `to_dtype`
// stays on the GPU instead of forcing a host synchronization. This file is
// concatenated into every library variant; the guard makes only the f32
// variant define the kernels (they name their types explicitly).
#if !defined(USE_F16) && !defined(USE_BF16)

struct CastPc {
    uint n;
};

// Same bf16 <-> f32 bit conversions as the dtype.metal prelude.
inline float cast_bf16_load(ushort x) {
    return as_type<float>(uint(x) << 16);
}
inline ushort cast_bf16_store(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return ushort((b >> 16) | 0x40u);
    }
    return ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

#define XN_CAST(name, S, D, EXPR)                                     \
kernel void name(                                                     \
    device const S *src [[buffer(0)]],                                \
    device D *dst [[buffer(1)]],                                      \
    constant CastPc &pc [[buffer(2)]],                                \
    uint i [[thread_position_in_grid]]                                \
) {                                                                   \
    if (i >= pc.n) return;                                            \
    S v = src[i];                                                     \
    dst[i] = (EXPR);                                                  \
}

XN_CAST(cast_f32_f16, float, half, half(v))
XN_CAST(cast_f16_f32, half, float, float(v))
XN_CAST(cast_f32_bf16, float, ushort, cast_bf16_store(v))
XN_CAST(cast_bf16_f32, ushort, float, cast_bf16_load(v))
XN_CAST(cast_f16_bf16, half, ushort, cast_bf16_store(float(v)))
XN_CAST(cast_bf16_f16, ushort, half, half(cast_bf16_load(v)))

#endif
