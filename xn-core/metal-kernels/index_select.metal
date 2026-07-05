// index_select over `dim`. ids is an i64 array viewed as uint pairs
// (little-endian): only the low word is read. An index of -1 selects zeros.
//   dst[(left*num_ids + id)*right + r] = src[(left*src_dim_size + ids[id])*right + r]

struct IndexSelectPc {
    uint left_size;
    uint num_ids;
    uint right_size;
    uint src_dim_size;
};

kernel void index_select(
    device const SCALAR *src [[buffer(0)]],
    device SCALAR *dst [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    constant IndexSelectPc &pc [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = pc.left_size * pc.num_ids * pc.right_size;
    if (gid >= total) return;

    uint r = gid % pc.right_size;
    uint tmp = gid / pc.right_size;
    uint id_i = tmp % pc.num_ids;
    uint left = tmp / pc.num_ids;

    int idx = int(ids[2u * id_i]);
    uint dst_off = (left * pc.num_ids + id_i) * pc.right_size + r;
    if (idx == -1) {
        dst[dst_off] = STORE(0.0);
        return;
    }
    uint src_off = (left * pc.src_dim_size + uint(idx)) * pc.right_size + r;
    dst[dst_off] = src[src_off];
}
