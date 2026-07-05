// 2D strided copy: dst[dst_o + i*dst_s + j] = src[src_o + i*src_s + j]
// for i in [0,d1), j in [0,d2).

struct Copy2dPc {
    uint d1;
    uint d2;
    uint src_s;
    uint dst_s;
    uint src_o;
    uint dst_o;
};

kernel void copy2d(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    constant Copy2dPc &pc [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= pc.d1 * pc.d2) return;
    uint i1 = idx / pc.d2;
    uint i2 = idx - pc.d2 * i1;
    dst[pc.dst_o + i1 * pc.dst_s + i2] = src[pc.src_o + i1 * pc.src_s + i2];
}
