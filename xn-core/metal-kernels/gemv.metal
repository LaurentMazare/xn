// GEMV: the m == 1 case of the batched GEMM (the LLM-decode hot path).
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// One simdgroup per output column j; GEMV_NSG simdgroups (= one threadgroup)
// cover GEMV_NSG consecutive columns. The simdgroup's 32 lanes cooperatively
// reduce over the k dimension in f32 and combine with a single simd_sum, so
// there is no threadgroup memory or barrier.
//
// When both inputs are contiguous along k and 4-element aligned (the
// matmul_t-against-row-major-weights case: `rhs_rs == 1`, each column j is a
// contiguous weight row), lanes read SCALAR4 vectors, so a simdgroup pulls
// 32x4 consecutive elements per iteration; this roughly halves the per-byte
// instruction cost versus scalar loads and is what a bandwidth-bound GEMV
// needs. Other layouts take the scalar loop below.
// Grid: (ceil(n / GEMV_NSG), batch, 1); threadgroup (32 * GEMV_NSG, 1, 1).
// Push constants match the GEMM kernels.
#define GEMV_NSG 8

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
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint j = tgid.x * GEMV_NSG + sgid;
    uint b = tgid.y;
    if (j >= pc.n) return;

    uint lbase = pc.lhs_o + b * pc.lhs_b_stride; // row i = 0
    uint rbase = pc.rhs_o + b * pc.rhs_b_stride + j * pc.rhs_cs;

    float acc = 0.0;
    bool vec = pc.lhs_cs == 1u && pc.rhs_rs == 1u && (pc.k % 4u) == 0u
        && (lbase % 4u) == 0u && (rbase % 4u) == 0u;
    if (vec) {
        device const SCALAR4 *lv = (device const SCALAR4 *)(lhs + lbase);
        device const SCALAR4 *rv = (device const SCALAR4 *)(rhs + rbase);
        uint k4 = pc.k / 4u;
        for (uint i = lane; i < k4; i += 32u) {
            acc += dot(LOAD4(lv[i]), LOAD4(rv[i]));
        }
    } else {
        for (uint l = lane; l < pc.k; l += 32u) {
            acc += LOAD(lhs[lbase + l * pc.lhs_cs]) * LOAD(rhs[rbase + l * pc.rhs_rs]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        dst[b * pc.n + j * pc.dst_cs] = STORE(acc);
    }
}
