// Direct 1D convolution (f32), gather form. One thread per output element.
//   src:    [batch, in_channels, length]
//   kernel: [out_channels, in_channels/groups, kernel_size]
//   dst:    [batch, out_channels, out_length]

struct Conv1dPc {
    uint batch;
    uint in_channels;
    uint out_channels;
    uint in_len;
    uint out_length;
    uint kernel_size;
    uint stride;
    uint padding;
    uint dilation;
    uint groups;
};

kernel void conv1d(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device const float *kern [[buffer(2)]],
    constant Conv1dPc &pc [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = pc.batch * pc.out_channels * pc.out_length;
    if (gid >= total) return;

    uint ol = gid % pc.out_length;
    uint tmp = gid / pc.out_length;
    uint oc = tmp % pc.out_channels;
    uint b = tmp / pc.out_channels;

    uint in_c_per_group = pc.in_channels / pc.groups;
    uint out_c_per_group = pc.out_channels / pc.groups;
    uint g = oc / out_c_per_group;
    uint in_c_start = g * in_c_per_group;

    float acc = 0.0;
    for (uint ko = 0u; ko < pc.kernel_size; ko++) {
        uint src_l = ol * pc.stride + ko * pc.dilation;
        if (src_l < pc.padding || src_l >= pc.padding + pc.in_len) continue;
        uint sl = src_l - pc.padding;
        for (uint ic = 0u; ic < in_c_per_group; ic++) {
            uint in_c = in_c_start + ic;
            float sv = src[(b * pc.in_channels + in_c) * pc.in_len + sl];
            float kv = kern[(oc * in_c_per_group + ic) * pc.kernel_size + ko];
            acc += sv * kv;
        }
    }
    dst[(b * pc.out_channels + oc) * pc.out_length + ol] = acc;
}
