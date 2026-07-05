// Reduction over one dimension. One threadgroup per output; f32 accumulation.
// Iteration shape (outer, inner, dim); physical layout (outer, dim, inner).
//   op: 0 = sum, 1 = max, 2 = min

struct ReducePc {
    uint num_outputs;
    uint dim_size;
    uint inner_size;
    uint op;
};

kernel void reduce(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    constant ReducePc &pc [[buffer(2)]],
    uint o [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[256];
    uint a_inner = o % pc.inner_size;
    uint a_outer = o / pc.inner_size;
    uint outer_base = a_outer * pc.dim_size * pc.inner_size + a_inner;

    float acc = pc.op == 0u ? 0.0 : (pc.op == 1u ? -3.402823466e+38 : 3.402823466e+38);
    for (uint k = tid; k < pc.dim_size; k += 256u) {
        float v = LOAD(src[outer_base + k * pc.inner_size]);
        if (pc.op == 0u) acc += v;
        else if (pc.op == 1u) acc = max(acc, v);
        else acc = min(acc, v);
    }
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            if (pc.op == 0u) sh[tid] += sh[tid + s];
            else if (pc.op == 1u) sh[tid] = max(sh[tid], sh[tid + s]);
            else sh[tid] = min(sh[tid], sh[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) dst[o] = STORE(sh[0]);
}
