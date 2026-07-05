// Arg-reduction over one dimension (-> i64 index). One threadgroup per output.
// The output buffer is an i64 array, viewed here as pairs of uint
// (little-endian): low word = index, high word = 0.
//   op: 0 = argmin, 1 = argmax

struct ReduceArgPc {
    uint num_outputs;
    uint dim_size;
    uint inner_size;
    uint op;
};

kernel void reduce_arg(
    device const SCALAR *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    constant ReduceArgPc &pc [[buffer(2)]],
    uint o [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float shv[256];
    threadgroup uint shi[256];
    uint a_inner = o % pc.inner_size;
    uint a_outer = o / pc.inner_size;
    uint outer_base = a_outer * pc.dim_size * pc.inner_size + a_inner;

    float best = pc.op == 1u ? -3.402823466e+38 : 3.402823466e+38;
    uint bidx = 0u;
    bool is_set = false;
    for (uint k = tid; k < pc.dim_size; k += 256u) {
        float v = LOAD(src[outer_base + k * pc.inner_size]);
        bool better = pc.op == 1u ? v > best : v < best;
        if (!is_set || better) {
            best = v;
            bidx = k;
            is_set = true;
        }
    }
    shv[tid] = best;
    shi[tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            bool better = pc.op == 1u ? shv[tid + s] > shv[tid] : shv[tid + s] < shv[tid];
            bool tie_lower = shv[tid + s] == shv[tid] && shi[tid + s] < shi[tid];
            if (better || tie_lower) {
                shv[tid] = shv[tid + s];
                shi[tid] = shi[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        dst[2u * o] = shi[0];
        dst[2u * o + 1u] = 0u;
    }
}
