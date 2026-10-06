# Third-party code

## KleidiAI

`kleidiai/` is a git submodule of Arm's [KleidiAI](https://github.com/ARM-software/kleidiai),
pinned to the release tag **v1.24.0**. It is licensed under Apache-2.0
(`kleidiai/LICENSES/Apache-2.0.txt`), and each file keeps its own copyright header.

The `kai` feature of `xn-core` uses a few of its files and nothing else:

- the SME2 int8 matmul kernels, one gemm (`..._sme2_mopa`) and one gemv (`..._sme2_dot`)
- the routines that pack their operands (`pack/`)
- the shared header and the SME helper assembly (`kai_common.h`, `kai_common_sme_asm.S`)

`build.rs` compiles them only when the feature is on and the target can run them. The
`exclude` list in `xn-core/Cargo.toml` names the same files, so the published crate carries
them and the license, and no other part of KleidiAI.

After cloning xn, fetch the submodule with:

```
git submodule update --init xn-core/third_party/kleidiai
```

To update KleidiAI, check out a newer tag in `kleidiai/` and commit the submodule. If a file
was renamed or a new one is needed, change both `build.rs` and the `exclude` list. The kernel
interfaces are declared by hand in `src/quantized/kai.rs`, so check them against the new
headers.
