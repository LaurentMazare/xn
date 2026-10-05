# KleidiAI subset

The files under `kai/` are copied unmodified from Arm's
[KleidiAI](https://github.com/ARM-software/kleidiai) at tag **v1.24.0**, at the same paths.
They are licensed under Apache-2.0 (see `LICENSE`), and each keeps its own copyright header.

They are the parts the `kai` feature of `xn-core` needs and nothing else:

- the SME2 int8 matmul kernels, one gemm (`..._sme2_mopa`) and one gemv (`..._sme2_dot`)
- the routines that pack their operands (`pack/`)
- the shared header and the SME helper assembly (`kai_common.h`, `kai_common_sme_asm.S`)

`build.rs` compiles them only when the feature is on and the target can run them.

To update, copy the same files from a newer tag and adjust the version above. The kernel
interfaces are declared by hand in `src/quantized/kai.rs`, so check them against the new
headers.
