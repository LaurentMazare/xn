// Direct 1D transposed convolution (f32), gather form. One thread per output.
//   src:    [batch, in_channels, length]
//   kernel: [in_channels, out_channels/groups, kernel_size]
//   dst:    [batch, out_channels, out_length]

struct ConvTranspose1dPc {
    uint batch;
    uint in_channels;
    uint out_channels;
    uint in_len;
    uint out_length;
    uint kernel_size;
    uint stride;
    uint padding;
    uint groups;
};

kernel void conv_transpose1d(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device const float *kern [[buffer(2)]],
    constant ConvTranspose1dPc &pc [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = pc.batch * pc.out_channels * pc.out_length;
    if (gid >= total) return;

    uint out_pos = gid % pc.out_length;
    uint tmp = gid / pc.out_length;
    uint oc = tmp % pc.out_channels;
    uint b = tmp / pc.out_channels;

    uint in_c_per_group = pc.in_channels / pc.groups;
    uint out_c_per_group = pc.out_channels / pc.groups;
    uint g = oc / out_c_per_group;
    uint oc_in_group = oc % out_c_per_group;
    uint in_c_start = g * in_c_per_group;

    uint out_pos_raw = out_pos + pc.padding;
    float acc = 0.0;
    for (uint ko = 0u; ko < pc.kernel_size; ko++) {
        if (out_pos_raw < ko) continue;
        uint num = out_pos_raw - ko;
        if (num % pc.stride != 0u) continue;
        uint il = num / pc.stride;
        if (il >= pc.in_len) continue;
        for (uint ic = 0u; ic < in_c_per_group; ic++) {
            uint in_c = in_c_start + ic;
            float sv = src[(b * pc.in_channels + in_c) * pc.in_len + il];
            float kv = kern[(in_c * out_c_per_group + oc_in_group) * pc.kernel_size + ko];
            acc += sv * kv;
        }
    }
    dst[(b * pc.out_channels + oc) * pc.out_length + out_pos] = acc;
}
