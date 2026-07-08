// GEMV: the m == 1 case of the batched GEMM (the LLM-decode hot path).
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// One workgroup per output column j (per batch b); the workgroup's threads
// cooperatively reduce over the k dimension in f32. Push constants match the
// GEMM kernel. Grid: (n, batch, 1).
struct Params {
    m: u32, n: u32, k: u32, batch: u32,
    lhs_b_stride: u32, rhs_b_stride: u32,
    lhs_cs: u32, lhs_rs: u32, rhs_cs: u32, rhs_rs: u32,
    dst_rs: u32, dst_cs: u32, lhs_o: u32, rhs_o: u32,
};
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j = wid.x;
    let b = wid.y;
    let tid = lid.x;

    let lbase = pc.lhs_o + b * pc.lhs_b_stride; // row i = 0
    let rbase = pc.rhs_o + b * pc.rhs_b_stride + j * pc.rhs_cs;

    var acc = 0.0;
    for (var l = tid; l < pc.k; l = l + 256u) {
        acc = acc + lhs[lbase + l * pc.lhs_cs] * rhs[rbase + l * pc.rhs_rs];
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[b * pc.n + j * pc.dst_cs] = sh[0];
    }
}
