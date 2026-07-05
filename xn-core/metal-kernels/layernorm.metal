// Row-wise LayerNorm, f32 accumulation. One threadgroup per row.
// remove_mean == 1: y = (x - mean) / sqrt(var + eps) * weight + bias
// remove_mean == 0: y =  x        / sqrt(var + eps) * weight + bias

struct LayerNormPc {
    uint ncols;
    float eps;
    uint remove_mean;
};

kernel void layernorm(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    device const SCALAR *weight [[buffer(2)]],
    device const SCALAR *bias [[buffer(3)]],
    constant LayerNormPc &pc [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float sh_sum[256];
    threadgroup float sh_sq[256];
    uint base = row * pc.ncols;

    float s1 = 0.0;
    float s2 = 0.0;
    for (uint c = tid; c < pc.ncols; c += 256u) {
        float x = LOAD(src[base + c]);
        s1 += x;
        s2 += x * x;
    }
    sh_sum[tid] = s1;
    sh_sq[tid] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh_sum[tid] += sh_sum[tid + s];
            sh_sq[tid] += sh_sq[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sh_sum[0] / float(pc.ncols);
    float var = sh_sq[0] / float(pc.ncols) - mean * mean;
    float inv_std = rsqrt(var + pc.eps);
    float mean_off = pc.remove_mean != 0u ? mean : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < pc.ncols; c += 256u) {
        float lhs = (LOAD(src[base + c]) - mean_off) * inv_std;
        dst[base + c] = STORE(lhs * LOAD(weight[c]) + LOAD(bias[c]));
    }
}
