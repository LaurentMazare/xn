// Apply causal mask in place: set dst to -inf where idx2 > offset + idx1.
// Linear index decomposes as (b, idx1, idx2) over (bh, t1, t2).

struct CausalityMaskPc {
    uint bh;
    uint t1;
    uint t2;
    uint offset;
};

kernel void causality_mask(
    device SCALAR *dst [[buffer(0)]],
    constant CausalityMaskPc &pc [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = pc.bh * pc.t1 * pc.t2;
    if (idx >= total) return;
    uint idx2 = idx % pc.t2;
    uint tmp = idx / pc.t2;
    uint idx1 = tmp % pc.t1;
    if (idx2 > pc.offset + idx1) {
        dst[idx] = STORE(as_type<float>(0xff800000u)); // -inf
    }
}
