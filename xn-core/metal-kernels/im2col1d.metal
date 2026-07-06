// Im2Col for 1D convolution (f32), groups == 1 only.
//   src: [batch, in_channels, in_len]
//   dst (col): [batch, out_length, in_channels * kernel_size]
// One thread per output element; dst is written in its natural flat order so
// gid doubles as the destination index directly.

struct Im2Col1dPc {
    uint batch;
    uint in_channels;
    uint in_len;
    uint out_length;
    uint kernel_size;
    uint stride;
    uint padding;
    uint dilation;
};

kernel void im2col1d(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant Im2Col1dPc &pc [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ck = pc.in_channels * pc.kernel_size;
    uint total = pc.batch * pc.out_length * ck;
    if (gid >= total) return;

    uint ck_idx = gid % ck;
    uint tmp = gid / ck;
    uint l = tmp % pc.out_length;
    uint b = tmp / pc.out_length;

    uint k_idx = ck_idx % pc.kernel_size;
    uint c_idx = ck_idx / pc.kernel_size;

    uint src_l_raw = l * pc.stride + k_idx * pc.dilation;
    float v = 0.0;
    if (src_l_raw >= pc.padding && src_l_raw < pc.padding + pc.in_len) {
        uint src_l = src_l_raw - pc.padding;
        v = src[(b * pc.in_channels + c_idx) * pc.in_len + src_l];
    }
    dst[gid] = v;
}
