// Elementwise binary op: dst = lhs op rhs, same shape/contiguous.
// For `bin_assign` (dst = dst op s), bind lhs=dst and rhs=s.

struct BinaryPc {
    uint n;
    uint op;
};

kernel void binary(
    device const SCALAR *lhs [[buffer(0)]],
    device const SCALAR *rhs [[buffer(1)]],
    device SCALAR *dst [[buffer(2)]],
    constant BinaryPc &pc [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= pc.n) return;
    ACC a = LOAD(lhs[i]);
    ACC b = LOAD(rhs[i]);
    ACC r;
    switch (pc.op) {
        case 0u: r = a + b; break;
        case 1u: r = a - b; break;
        case 2u: r = a * b; break;
        case 3u: r = a / b; break;
        case 4u: r = max(a, b); break;
        case 5u: r = min(a, b); break;
        default: r = a;
    }
    dst[i] = STORE(r);
}
