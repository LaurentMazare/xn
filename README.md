# xn
Yet another minimalist deep-learning framework optimized for inference

## Third-party code

The `kai` feature runs some matmuls through Arm's
[KleidiAI](https://github.com/ARM-software/kleidiai) kernels. They are licensed under
Apache-2.0 only, so a build with `kai` includes Apache-2.0 code even if you use xn under MIT.
They are a git submodule in `xn-core/third_party`, described in the README there.
