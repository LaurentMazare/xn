// dst = src * scale + add   (elementwise)

struct ScaleAddPc {
    uint n;
    float scale;
    float add;
};

kernel void scale_add(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    constant ScaleAddPc &pc [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= pc.n) return;
    dst[i] = STORE(LOAD(src[i]) * pc.scale + pc.add);
}
