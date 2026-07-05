// General strided broadcast binary op. `info` packs, per dimension:
//   [dims (num_dims), lhs_strides (num_dims), rhs_strides (num_dims)]
// A broadcast dimension has stride 0. Destination is contiguous row-major.

struct BroadcastPc {
    uint numel;
    uint num_dims;
    uint op;
};

kernel void broadcast(
    device const SCALAR *lhs [[buffer(0)]],
    device const SCALAR *rhs [[buffer(1)]],
    device SCALAR *dst [[buffer(2)]],
    device const uint *info [[buffer(3)]],
    constant BroadcastPc &pc [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= pc.numel) return;
    uint nd = pc.num_dims;
    uint li = 0u;
    uint ri = 0u;
    uint rem = idx;
    for (uint d = 0u; d < nd; d++) {
        uint di = nd - 1u - d;
        uint dimv = info[di];
        uint coord = rem % dimv;
        rem /= dimv;
        li += coord * info[nd + di];
        ri += coord * info[2u * nd + di];
    }
    float a = LOAD(lhs[li]);
    float b = LOAD(rhs[ri]);
    float r;
    switch (pc.op) {
        case 0u: r = a + b; break;
        case 1u: r = a - b; break;
        case 2u: r = a * b; break;
        case 3u: r = a / b; break;
        case 4u: r = max(a, b); break;
        case 5u: r = min(a, b); break;
        default: r = a;
    }
    dst[idx] = STORE(r);
}
