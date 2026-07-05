// Tiled threadgroup-memory GEMM (f32 accumulation), general strides + batch.
//   dst[b, i, j] = sum_l lhs[b, i, l] * rhs[b, l, j]
// Element offsets (same convention as the naive gemm):
//   lhs: lhs_o + b*lhs_b_stride + i*lhs_rs + l*lhs_cs
//   rhs: rhs_o + b*rhs_b_stride + l*rhs_rs + j*rhs_cs
//   dst:              b*m*n     + i*dst_rs + j*dst_cs
// Grid: (ceil(n/TILE), ceil(m/TILE), batch); threadgroup (TILE, TILE, 1).
#define TILE 16

struct GemmPc {
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

kernel void gemm_tiled(
    device SCALAR *dst [[buffer(0)]],
    device const SCALAR *lhs [[buffer(1)]],
    device const SCALAR *rhs [[buffer(2)]],
    constant GemmPc &pc [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float lt[TILE][TILE];
    threadgroup float rt[TILE][TILE];

    uint b = tgid.z;
    uint row = gid.y;
    uint col = gid.x;
    uint ly = lid.y;
    uint lx = lid.x;

    uint lhs_base = pc.lhs_o + b * pc.lhs_b_stride;
    uint rhs_base = pc.rhs_o + b * pc.rhs_b_stride;

    float acc = 0.0;
    uint num_tiles = (pc.k + uint(TILE) - 1u) / uint(TILE);
    for (uint t = 0u; t < num_tiles; t++) {
        uint l_lhs = t * uint(TILE) + lx; // k-index for the lhs tile column
        uint l_rhs = t * uint(TILE) + ly; // k-index for the rhs tile row
        lt[ly][lx] = (row < pc.m && l_lhs < pc.k)
            ? LOAD(lhs[lhs_base + row * pc.lhs_rs + l_lhs * pc.lhs_cs])
            : 0.0;
        rt[ly][lx] = (col < pc.n && l_rhs < pc.k)
            ? LOAD(rhs[rhs_base + l_rhs * pc.rhs_rs + col * pc.rhs_cs])
            : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < uint(TILE); kk++) {
            acc += lt[ly][kk] * rt[kk][lx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < pc.m && col < pc.n) {
        dst[b * pc.m * pc.n + row * pc.dst_rs + col * pc.dst_cs] = STORE(acc);
    }
}
