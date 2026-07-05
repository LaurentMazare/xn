// Transpose two dims of a tensor whose shape factors as
// (d_i, d1, d_j, d2, d_k), swapping d1 and d2. Mirrors layout.cu.

struct TransposePc {
    uint numel;
    uint d1;
    uint d2;
    uint d_i;
    uint d_j;
    uint d_k;
};

kernel void transpose(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    constant TransposePc &pc [[buffer(2)]],
    uint dst_idx [[thread_position_in_grid]]
) {
    if (dst_idx >= pc.numel) return;

    uint rem = dst_idx;
    uint i = rem / (pc.d2 * pc.d_j * pc.d1 * pc.d_k);
    rem -= i * (pc.d2 * pc.d_j * pc.d1 * pc.d_k);
    uint a2 = rem / (pc.d_j * pc.d1 * pc.d_k);
    rem -= a2 * (pc.d_j * pc.d1 * pc.d_k);
    uint j = rem / (pc.d1 * pc.d_k);
    rem -= j * (pc.d1 * pc.d_k);
    uint a1 = rem / pc.d_k;
    rem -= a1 * pc.d_k;
    uint k = rem;

    uint src_idx = i * pc.d1 * pc.d_j * pc.d2 * pc.d_k
                 + a1 * pc.d_j * pc.d2 * pc.d_k
                 + j * pc.d2 * pc.d_k
                 + a2 * pc.d_k
                 + k;
    dst[dst_idx] = src[src_idx];
}
