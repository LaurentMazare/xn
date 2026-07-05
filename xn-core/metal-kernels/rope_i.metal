// Rotary position embedding, interleaved.

struct RopeIPc {
    uint bh;
    uint td;
    uint h;
    uint cs_stride_b;
    uint cos_off;
    uint sin_off;
};

kernel void rope_i(
    device const SCALAR *cosb [[buffer(0)]],
    device const SCALAR *sinb [[buffer(1)]],
    device const SCALAR *src [[buffer(2)]],
    device SCALAR *dst [[buffer(3)]],
    constant RopeIPc &pc [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (2u * idx >= pc.bh * pc.td) return;

    uint half_td = pc.td / 2u;
    uint i_bh = idx / half_td;
    uint cos_idx = idx % half_td;
    if (pc.cs_stride_b > 0u) cos_idx += (i_bh / pc.h) * pc.cs_stride_b;

    float c = LOAD(cosb[pc.cos_off + cos_idx]);
    float s = LOAD(sinb[pc.sin_off + cos_idx]);
    float a = LOAD(src[2u * idx]);
    float b = LOAD(src[2u * idx + 1u]);
    dst[2u * idx] = STORE(a * c - b * s);
    dst[2u * idx + 1u] = STORE(a * s + b * c);
}
