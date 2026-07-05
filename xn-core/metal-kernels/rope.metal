// Rotary position embedding, non-interleaved (GPT-NeoX style).

struct RopePc {
    uint bh;
    uint td;
    uint d;
    uint h;
    uint cs_stride_b;
    uint cos_off;
    uint sin_off;
};

kernel void rope(
    device const SCALAR *cosb [[buffer(0)]],
    device const SCALAR *sinb [[buffer(1)]],
    device const SCALAR *src [[buffer(2)]],
    device SCALAR *dst [[buffer(3)]],
    constant RopePc &pc [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (2u * idx >= pc.bh * pc.td) return;

    uint half_td = pc.td / 2u;
    uint half_d = pc.d / 2u;
    uint i_bh = idx / half_td;
    uint i_td = idx - half_td * i_bh;
    uint i_t = i_td / half_d;
    uint i_d = i_td - half_d * i_t;
    uint i1 = i_bh * pc.td + i_t * pc.d + i_d;
    uint i2 = i1 + half_d;
    uint i_cs = i_t * half_d + i_d;
    if (pc.cs_stride_b > 0u) i_cs += (i_bh / pc.h) * pc.cs_stride_b;

    float c = LOAD(cosb[pc.cos_off + i_cs]);
    float s = LOAD(sinb[pc.sin_off + i_cs]);
    float a = LOAD(src[i1]);
    float b = LOAD(src[i2]);
    dst[i1] = STORE(a * c - b * s);
    dst[i2] = STORE(a * s + b * c);
}
