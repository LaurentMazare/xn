// Row-wise RMSNorm, f32 accumulation. One threadgroup per row.
// dst = x * rsqrt(mean(x^2) + eps) * alpha

struct RmsNormPc {
    uint ncols;
    float eps;
};

kernel void rmsnorm(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    device const SCALAR *alpha [[buffer(2)]],
    constant RmsNormPc &pc [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[256];
    uint base = row * pc.ncols;

    float acc = 0.0;
    for (uint c = tid; c < pc.ncols; c += 256u) {
        float x = LOAD(src[base + c]);
        acc += x * x;
    }
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sh[0] / float(pc.ncols);
    float scale = rsqrt(mean + pc.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < pc.ncols; c += 256u) {
        dst[base + c] = STORE(scale * LOAD(src[base + c]) * LOAD(alpha[c]));
    }
}
