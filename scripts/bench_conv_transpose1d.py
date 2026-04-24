# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "torch",
# ]
# ///
"""Single-thread CPU benchmark for ConvTranspose1d at a few shapes used in
   the TTS model.
"""

import os

# Pin BLAS to a single thread before importing torch.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)


@dataclass
class Case:
    name: str
    cin: int
    lin: int
    cout: int
    k: int
    stride: int

    @property
    def lout(self) -> int:
        return (self.lin - 1) * self.stride + self.k


CASES = [
    Case("512x16 -> 256x102, k=12, s=6", cin=512, lin=16, cout=256, k=12, stride=6),
    Case("256x96 -> 128x485, k=10, s=5", cin=256, lin=96, cout=128, k=10, stride=5),
    Case("128x480 -> 64x1924, k=8, s=4", cin=128, lin=480, cout=64, k=8, stride=4),
]


def bench(name, fn, flops, iters, warmup=10):
    for _ in range(warmup):
        y = fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
    dt = (time.perf_counter() - t0) / iters
    gflops = flops / dt / 1e9
    print(f"  {name:34s} {dt * 1e3:8.3f} ms/iter   {gflops:6.2f} GFLOP/s   out={tuple(y.shape)}")
    return y


def make_fns(case: Case, x: torch.Tensor, w: torch.Tensor):
    B = x.shape[0]

    def native():
        return F.conv_transpose1d(x, w, stride=case.stride)

    def matmul_scatter():
        # y[b, co, t_in*s + k] += sum_ci x[b, ci, t_in] * w[ci, co, k]
        w2 = w.reshape(case.cin, case.cout * case.k)
        z = torch.matmul(x.transpose(1, 2), w2)  # (B, Lin, Cout*K)
        z = z.reshape(B, case.lin, case.cout, case.k).permute(0, 2, 3, 1).contiguous()
        out = torch.zeros(B, case.cout, case.lout)
        for kk in range(case.k):
            out[:, :, kk : kk + case.lin * case.stride : case.stride] += z[:, :, kk, :]
        return out

    module = torch.nn.ConvTranspose1d(case.cin, case.cout, case.k, stride=case.stride, bias=False)
    with torch.no_grad():
        module.weight.copy_(w)

    def mod():
        return module(x)

    return native, matmul_scatter, mod


def run_case(case: Case):
    B = 1
    x = torch.randn(B, case.cin, case.lin)
    w = torch.randn(case.cin, case.cout, case.k)
    macs = B * case.cin * case.cout * case.k * case.lin
    flops = 2 * macs
    # Scale iteration count so each measurement takes ~0.1-0.3s.
    iters = max(20, min(2000, int(2e8 / flops)))

    print(f"[{case.name}]")
    print(
        f"  x={tuple(x.shape)}  w={tuple(w.shape)}  y={(B, case.cout, case.lout)}"
        f"  stride={case.stride}"
    )
    print(
        f"  work={macs / 1e6:.2f} MMACs = {flops / 1e6:.2f} MFLOPs"
        f"   weight={w.numel() * 4 / 1024 / 1024:.2f} MB fp32"
        f"   iters={iters}"
    )

    native, matmul_scatter, mod = make_fns(case, x, w)
    y_ref = bench("F.conv_transpose1d", native, flops, iters)
    y_mm = bench("matmul + scatter-add", matmul_scatter, flops, iters)
    y_mod = bench("nn.ConvTranspose1d", mod, flops, iters)

    print(
        f"  max |native - matmul_scatter| = {(y_ref - y_mm).abs().max().item():.2e}"
        f"   max |native - nn.Module| = {(y_ref - y_mod).abs().max().item():.2e}"
    )
    print()


def main():
    print(f"threads:   intra={torch.get_num_threads()}  inter={torch.get_num_interop_threads()}")
    print()
    for case in CASES:
        run_case(case)


if __name__ == "__main__":
    main()
