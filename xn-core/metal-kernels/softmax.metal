// Row-wise softmax. One threadgroup per row; `ncols` elements per row.
// Reductions accumulate in f32.

struct SoftmaxPc {
    uint ncols;
};

kernel void softmax(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    constant SoftmaxPc &pc [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[256];
    uint base = row * pc.ncols;

    float m = -3.402823466e+38;
    for (uint c = tid; c < pc.ncols; c += 256u) m = max(m, LOAD(src[base + c]));
    sh[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxv = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0;
    for (uint c = tid; c < pc.ncols; c += 256u) {
        float e = exp(LOAD(src[base + c]) - maxv);
        dst[base + c] = STORE(e);
        sum += e;
    }
    sh[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < pc.ncols; c += 256u) {
        dst[base + c] = STORE(LOAD(dst[base + c]) * inv);
    }
}
