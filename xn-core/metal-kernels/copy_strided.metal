// Copy from a strided source to a contiguous destination.
// `info` packs [dims (num_dims), src_strides (num_dims)].

struct CopyStridedPc {
    uint numel;
    uint num_dims;
    uint src_offset;
};

kernel void copy_strided(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    device const uint *info [[buffer(2)]],
    constant CopyStridedPc &pc [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= pc.numel) return;
    uint nd = pc.num_dims;
    uint si = 0u;
    uint rem = idx;
    for (uint d = 0u; d < nd; d++) {
        uint di = nd - 1u - d;
        uint dimv = info[di];
        si += (rem % dimv) * info[nd + di];
        rem /= dimv;
    }
    dst[idx] = src[pc.src_offset + si];
}
