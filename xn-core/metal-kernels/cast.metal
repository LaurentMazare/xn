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

// Float -> integer casts follow Rust `as` semantics: truncate toward zero,
// saturate out-of-range values, NaN -> 0 (checked on the bit pattern so
// fast-math cannot elide it).
inline long cast_f32_to_i64(float v) {
    if ((as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u) {
        return 0;
    }
    if (v >= 9223372036854775808.0f) {
        return 0x7fffffffffffffff;
    }
    if (v <= -9223372036854775808.0f) {
        return as_type<long>(0x8000000000000000);
    }
    return long(v);
}
inline uchar cast_f32_to_u8(float v) {
    if ((as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u) {
        return 0;
    }
    return uchar(clamp(v, 0.0f, 255.0f));
}

XN_CAST(cast_f32_f16, float, half, half(v))
XN_CAST(cast_f16_f32, half, float, float(v))
XN_CAST(cast_f32_bf16, float, ushort, cast_bf16_store(v))
XN_CAST(cast_bf16_f32, ushort, float, cast_bf16_load(v))
XN_CAST(cast_f16_bf16, half, ushort, cast_bf16_store(float(v)))
XN_CAST(cast_bf16_f16, ushort, half, half(cast_bf16_load(v)))

XN_CAST(cast_f32_i64, float, long, cast_f32_to_i64(v))
XN_CAST(cast_f32_u8, float, uchar, cast_f32_to_u8(v))
XN_CAST(cast_f16_i64, half, long, cast_f32_to_i64(float(v)))
XN_CAST(cast_f16_u8, half, uchar, cast_f32_to_u8(float(v)))
XN_CAST(cast_bf16_i64, ushort, long, cast_f32_to_i64(cast_bf16_load(v)))
XN_CAST(cast_bf16_u8, ushort, uchar, cast_f32_to_u8(cast_bf16_load(v)))

XN_CAST(cast_i64_f32, long, float, float(v))
XN_CAST(cast_i64_f16, long, half, half(float(v)))
XN_CAST(cast_i64_bf16, long, ushort, cast_bf16_store(float(v)))
XN_CAST(cast_u8_f32, uchar, float, float(v))
XN_CAST(cast_u8_f16, uchar, half, half(float(v)))
XN_CAST(cast_u8_bf16, uchar, ushort, cast_bf16_store(float(v)))

XN_CAST(cast_i64_u8, long, uchar, uchar(v))
XN_CAST(cast_u8_i64, uchar, long, long(v))

#endif
