// GEMV: the m == 1 case of the batched GEMM (the LLM-decode hot path).
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// One threadgroup per output column j (per batch b); the threadgroup's threads
// cooperatively reduce over the k dimension in f32. For weight matrices stored
// row-major with `rhs_rs == 1` (the matmul_t case), each threadgroup reads a
// contiguous weight row, giving coalesced, bandwidth-bound reads.
// Grid: (n, batch, 1). Push constants match the GEMM kernels.

struct GemvPc {
    uint m;
    uint n;
    uint k;
    uint batch;
    uint lhs_b_stride;
    uint rhs_b_stride;
    uint lhs_cs;
    uint lhs_rs;
    uint rhs_cs;
    uint rhs_rs;
    uint dst_rs;
    uint dst_cs;
    uint lhs_o;
    uint rhs_o;
};

kernel void gemv(
    device SCALAR *dst [[buffer(0)]],
    device const SCALAR *lhs [[buffer(1)]],
    device const SCALAR *rhs [[buffer(2)]],
    constant GemvPc &pc [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[256];
    uint j = tgid.x;
    uint b = tgid.y;
    uint tid = lid.x;

    uint lbase = pc.lhs_o + b * pc.lhs_b_stride; // row i = 0
    uint rbase = pc.rhs_o + b * pc.rhs_b_stride + j * pc.rhs_cs;

    float acc = 0.0;
    for (uint l = tid; l < pc.k; l += 256u) {
        acc += LOAD(lhs[lbase + l * pc.lhs_cs]) * LOAD(rhs[rbase + l * pc.rhs_rs]);
    }
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        dst[b * pc.n + j * pc.dst_cs] = STORE(sh[0]);
    }
}
