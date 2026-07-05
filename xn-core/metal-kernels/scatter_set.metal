// scatter_set: for each src element at flat position i,
//   dst[left*dst_dim_size*right + ids[i]*right + right_idx] = src[i]
// where left = i/(right*src_dim_size), right_idx = i % right.
// ids is an i64 array viewed as uint pairs; only the low word is read.

struct ScatterSetPc {
    uint numel;
    uint right_size;
    uint src_dim_size;
    uint dst_dim_size;
};

kernel void scatter_set(
    device SCALAR *dst [[buffer(0)]],
    device const SCALAR *src [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    constant ScatterSetPc &pc [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= pc.numel) return;
    uint right = i % pc.right_size;
    uint left = i / (pc.right_size * pc.src_dim_size);
    uint idx = ids[2u * i];
    uint dst_off = left * pc.dst_dim_size * pc.right_size + idx * pc.right_size + right;
    dst[dst_off] = src[i];
}
